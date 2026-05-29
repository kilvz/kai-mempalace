"""convo_scanner.py — Parse Claude Code conversation directories into ProjectInfo.

Claude Code stores sessions under ``~/.claude/projects/<slug>/<id>.jsonl``.
This scanner reads one record per session to recover the accurate project name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from kai_mempalace.project_scanner import ProjectInfo

MAX_HEADER_LINES = 20


def is_claude_projects_root(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        if not (child.is_dir() and child.name.startswith("-")):
            continue
        try:
            if any(p.suffix == ".jsonl" for p in child.iterdir() if p.is_file()):
                return True
        except OSError:
            continue
    return False


def _extract_cwd_from_session(session_file: Path) -> Optional[str]:
    try:
        with open(session_file, encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= MAX_HEADER_LINES:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cwd = obj.get("cwd")
                if isinstance(cwd, str) and cwd:
                    return cwd
    except OSError:
        return None
    return None


def _decode_slug_fallback(slug: str) -> str:
    stripped = slug.lstrip("-")
    parts = [p for p in stripped.split("-") if p]
    return parts[-1] if parts else slug


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _resolve_project_name(project_dir: Path) -> str:
    sessions = sorted(
        (p for p in project_dir.iterdir() if p.is_file() and p.suffix == ".jsonl"),
        key=_safe_mtime,
        reverse=True,
    )
    for session in sessions:
        cwd = _extract_cwd_from_session(session)
        if cwd:
            return Path(cwd).name or cwd
    return _decode_slug_fallback(project_dir.name)


def scan_claude_projects(path: str | Path) -> list[ProjectInfo]:
    root = Path(path).expanduser().resolve()
    if not is_claude_projects_root(root):
        return []

    projects: dict[str, ProjectInfo] = {}
    for sub in sorted(root.iterdir()):
        if not (sub.is_dir() and sub.name.startswith("-")):
            continue
        try:
            sessions = [p for p in sub.iterdir() if p.is_file() and p.suffix == ".jsonl"]
        except OSError:
            continue
        if not sessions:
            continue

        name = _resolve_project_name(sub)
        session_count = len(sessions)

        proj = ProjectInfo(
            name=name, repo_root=sub, manifest=None,
            has_git=False, total_commits=session_count,
            user_commits=session_count, is_mine=True,
        )
        existing = projects.get(name)
        if existing is None or session_count > existing.user_commits:
            projects[name] = proj

    return sorted(projects.values(), key=lambda p: (-p.user_commits, p.name))
