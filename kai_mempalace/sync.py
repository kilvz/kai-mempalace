"""sync.py — Gitignore-aware drawer prune using FAISS/SQLite backend."""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, TypedDict

from kai_mempalace.palace import Palace

logger = logging.getLogger(__name__)
_BATCH = 1000


class SyncReport(TypedDict):
    scanned: int
    kept: int
    gitignored: int
    missing: int
    no_source: int
    out_of_scope: int
    removed_drawers: int
    removed_closets: int
    dry_run: bool
    by_source: dict[str, int]


def _resolve_project_root(source_file: Path, project_roots: list) -> Optional[Path]:
    for root in project_roots:
        try:
            source_file.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _has_git_marker(path: Path) -> bool:
    return (path / ".git").exists() or (path / ".gitignore").is_file()


def _ancestor_matchers(source_file: Path, root: Path, matcher_cache: dict) -> list:
    matchers: list = []
    try:
        parts = source_file.relative_to(root).parts
    except ValueError:
        return matchers
    cursor = root
    matcher = _load_gi_matcher(cursor, matcher_cache)
    if matcher is not None:
        matchers.append(matcher)
    for part in parts[:-1]:
        cursor = cursor / part
        matcher = _load_gi_matcher(cursor, matcher_cache)
        if matcher is not None:
            matchers.append(matcher)
    return matchers


def _load_gi_matcher(directory: Path, cache: dict):
    cache_key = str(directory.resolve())
    if cache_key in cache:
        return cache[cache_key]
    gi_file = directory / ".gitignore"
    if not gi_file.is_file():
        cache[cache_key] = None
        return None
    try:
        import pathspec
        with open(gi_file, "r") as f:
            spec = pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
        cache[cache_key] = spec
        return spec
    except ImportError:
        cache[cache_key] = None
        return None
    except Exception:
        cache[cache_key] = None
        return None


def is_gitignored(path: Path, matchers: list, is_dir: bool = False) -> bool:
    rel = str(path)
    for matcher in matchers:
        if matcher.match_file(rel):
            return True
    return False


def _is_registry_row(meta: dict, drawer_id: str) -> bool:
    if (meta or {}).get("room") == "_registry":
        return True
    if (meta or {}).get("ingest_mode") == "registry":
        return True
    if drawer_id and drawer_id.startswith("_reg_"):
        return True
    return False


def _classify_drawer(
    meta: dict, matcher_cache: dict, project_roots: list, drawer_id: str = ""
) -> str:
    if _is_registry_row(meta, drawer_id):
        return "kept"
    source_file = (meta or {}).get("source_file")
    if not source_file:
        return "no_source"
    src = Path(source_file)
    if not src.is_absolute():
        return "no_source"
    src = src.resolve(strict=False)
    root = _resolve_project_root(src, project_roots)
    if root is None:
        return "out_of_scope"
    if not src.exists():
        return "missing"
    matchers = _ancestor_matchers(src, root, matcher_cache)
    if matchers and is_gitignored(src, matchers, is_dir=False):
        return "gitignored"
    return "kept"


def _iter_drawer_metadata(palace, wing: Optional[str]):
    conn = sqlite3.connect(str(palace._base / "palace.db"))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT id, wing, room, content, metadata, source_file, created_at FROM drawers"
        params = []
        if wing:
            sql += " WHERE wing = ?"
            params.append(wing)
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        for r in rows:
            meta = json.loads(r["metadata"] or "{}")
            yield r["id"], meta
    finally:
        conn.close()


def _auto_detect_project_roots(palace, wing: Optional[str]) -> list:
    conn = sqlite3.connect(str(palace._base / "palace.db"))
    conn.row_factory = sqlite3.Row
    try:
        sql = "SELECT source_file FROM drawers"
        params = []
        if wing:
            sql += " WHERE wing = ?"
            params.append(wing)
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    roots: set = set()
    seen_sources: set = set()
    for r in rows:
        source_file = r["source_file"]
        if not source_file or source_file in seen_sources:
            continue
        seen_sources.add(source_file)
        src = Path(source_file)
        if not src.is_absolute():
            continue
        for parent in src.parents:
            if _has_git_marker(parent):
                roots.add(parent.resolve(strict=False))
                break
    return sorted(roots, key=lambda p: (-len(str(p)), str(p)))


def _normalize_project_dirs(project_dirs) -> list:
    resolved = [Path(p).resolve(strict=False) for p in project_dirs]
    return sorted(resolved, key=lambda p: (-len(str(p)), str(p)))


def sync_palace(
    palace_path: str,
    project_dirs: Optional[list] = None,
    wing: Optional[str] = None,
    dry_run: bool = True,
    batch_size: int = _BATCH,
    wal_log: Optional[Callable] = None,
) -> SyncReport:
    if not dry_run and not wing and not project_dirs:
        raise ValueError(
            "sync apply requires explicit wing= or project_dirs= so it cannot "
            "auto-prune every wing in a multi-project palace; pass --wing or "
            "a project directory"
        )
    if project_dirs is not None and not project_dirs:
        raise ValueError(
            "project_dirs was provided but is empty; pass at least one project "
            "root or pass project_dirs=None to auto-detect from drawer metadata"
        )

    counts = {
        "scanned": 0, "kept": 0, "gitignored": 0,
        "missing": 0, "no_source": 0, "out_of_scope": 0,
    }
    by_source: dict = defaultdict(int)
    removable_ids: list = []
    removable_sources: set = set()

    palace = Palace(palace_path)
    palace.init()

    if project_dirs is not None:
        roots = _normalize_project_dirs(project_dirs)
    else:
        roots = _auto_detect_project_roots(palace, wing)

    matcher_cache: dict = {}
    classification_cache: dict = {}

    for drawer_id, meta in _iter_drawer_metadata(palace, wing):
        counts["scanned"] += 1
        meta = meta or {}
        source_file = meta.get("source_file")

        if _is_registry_row(meta, drawer_id):
            bucket = "kept"
        elif source_file and source_file in classification_cache:
            bucket = classification_cache[source_file]
        else:
            bucket = _classify_drawer(meta, matcher_cache, roots, drawer_id)
            if source_file:
                classification_cache[source_file] = bucket

        counts[bucket] += 1
        if bucket in ("gitignored", "missing"):
            removable_ids.append(drawer_id)
            if source_file:
                removable_sources.add(source_file)
                by_source[source_file] += 1

    report: SyncReport = {
        **counts,
        "removed_drawers": 0,
        "removed_closets": 0,
        "dry_run": dry_run,
        "by_source": dict(by_source),
    }

    if dry_run or not removable_ids:
        return report

    for did in removable_ids:
        palace.delete_drawer(did)
    report["removed_drawers"] = len(removable_ids)

    return report


__all__ = [
    "SyncReport",
    "sync_palace",
]
