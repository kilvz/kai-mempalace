"""Palace schema migration and FAISS rebuild.

Detects the current palace schema version, applies pending migrations,
and provides disaster recovery — rebuild the FAISS index from SQLite
when vectors are lost or corrupted.

Schema versions
---------------
v0 — unversioned palace (pre-migrate): wings, rooms, drawers (with FTS5), closets
v1 — adds _meta version tracking table and content_date column to drawers
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

from kai_mempalace.backends.embedder import get_embedder

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 1
META_TABLE = "_meta"


# ── Version tracking ───────────────────────────────────────────────────────


def _open_db(palace_path: str) -> sqlite3.Connection:
    base = Path(palace_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "palace.db"))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_meta_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {META_TABLE} (key TEXT PRIMARY KEY, value TEXT)"
    )


def get_palace_version(palace_path: str) -> int:
    """Return the schema version of an existing palace, or 0 if unversioned."""
    conn = _open_db(palace_path)
    try:
        _ensure_meta_table(conn)
        row = conn.execute(
            f"SELECT value FROM {META_TABLE} WHERE key='palace_version'"
        ).fetchone()
        return int(row["value"]) if row else 0
    except (sqlite3.Error, ValueError):
        return 0
    finally:
        conn.close()


def set_palace_version(palace_path: str, version: int) -> None:
    conn = _open_db(palace_path)
    try:
        _ensure_meta_table(conn)
        conn.execute(
            f"INSERT OR REPLACE INTO {META_TABLE} (key, value) VALUES ('palace_version', ?)",
            (str(version),),
        )
        conn.commit()
    finally:
        conn.close()


# ── Schema migrations ──────────────────────────────────────────────────────


def _migrate_v0_to_v1(palace_path: str, conn: sqlite3.Connection) -> None:
    """v0 -> v1: add _meta version table and content_date column."""
    _ensure_meta_table(conn)
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(drawers)").fetchall()
    }
    if "content_date" not in existing:
        conn.execute("ALTER TABLE drawers ADD COLUMN content_date TEXT DEFAULT ''")
        logger.info("migrate: added content_date column to drawers")


_MIGRATIONS = {
    1: _migrate_v0_to_v1,
}


def migrate_schema(palace_path: str, dry_run: bool = False) -> dict:
    """Run all pending schema migrations and return a report."""
    current = get_palace_version(palace_path)
    report = {
        "path": str(Path(palace_path).expanduser().resolve()),
        "version_before": current,
        "version_after": current,
        "migrations_applied": [],
        "dry_run": dry_run,
    }

    if current >= CURRENT_SCHEMA_VERSION:
        report["version_after"] = current
        return report

    conn = _open_db(palace_path)
    try:
        for target_version in range(current + 1, CURRENT_SCHEMA_VERSION + 1):
            fn = _MIGRATIONS.get(target_version)
            if fn is None:
                logger.warning("No migration defined for v%d -> v%d", target_version - 1, target_version)
                continue
            label = f"v{target_version - 1}_to_v{target_version}"
            logger.info("Applying migration %s", label)
            if not dry_run:
                fn(palace_path, conn)
                set_palace_version(palace_path, target_version)
            report["migrations_applied"].append(label)
            report["version_after"] = target_version

        if not dry_run:
            conn.commit()
    finally:
        conn.close()

    return report


# ── FAISS rebuild ──────────────────────────────────────────────────────────


def rebuild_faiss(palace_path: str) -> dict:
    """Rebuild the FAISS index from palace.db SQLite content.

    Reads all drawer texts, re-embeds them, and writes a fresh
    ``data/index.faiss`` + ``data/metadata.db``. Existing data is
    replaced. Useful when the FAISS index is corrupted or missing.
    """
    from kai_mempalace.backends.faiss_store import FaissStore

    base = Path(palace_path).expanduser().resolve()
    data_dir = base / "data"

    # Read all drawers
    conn = _open_db(palace_path)
    try:
        rows = conn.execute(
            "SELECT id, content, metadata FROM drawers ORDER BY rowid"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return {"ok": True, "vectors_rebuilt": 0, "message": "No drawers to rebuild"}

    ids = []
    texts = []
    metadatas = []
    for r in rows:
        ids.append(r["id"])
        texts.append(r["content"])
        try:
            metadatas.append(json.loads(r["metadata"]) if r["metadata"] else {})
        except (json.JSONDecodeError, TypeError):
            metadatas.append({})

    embedder = get_embedder()
    vectors = embedder.embed(texts)

    data_dir.mkdir(parents=True, exist_ok=True)
    store = FaissStore(str(data_dir), dimension=vectors.shape[1])
    store.clear()
    store.add(ids, texts, metadatas, vectors)

    n = store.count()
    store.close()

    logger.info("Rebuilt FAISS index with %d vectors", n)
    return {"ok": True, "vectors_rebuilt": n}


# ── Orchestrator ────────────────────────────────────────────────────────────


def migrate(palace_path: str, dry_run: bool = False, confirm: bool = False) -> dict:
    """Run all pending schema migrations.

    Returns a dict with ``migrations_applied``, ``version_before``,
    ``version_after``, and ``rebuild_triggered``.
    """
    result = migrate_schema(palace_path, dry_run=dry_run)
    return result


# ── CLI integration ────────────────────────────────────────────────────────


def status(palace_path: str) -> dict:
    """Detailed schema-version status of the palace."""
    base = Path(palace_path).expanduser().resolve()
    palace_db = base / "palace.db"
    data_dir = base / "data"
    index_file = data_dir / "index.faiss"
    meta_file = data_dir / "metadata.db"

    info: dict = {
        "path": str(base),
        "palace_db_exists": palace_db.exists(),
        "index_exists": index_file.exists(),
        "metadata_db_exists": meta_file.exists(),
    }

    if not palace_db.exists():
        info["version"] = 0
        info["message"] = "Palace not initialized"
        return info

    version = get_palace_version(str(palace_path))
    info["version"] = version
    info["latest_version"] = CURRENT_SCHEMA_VERSION
    info["up_to_date"] = version >= CURRENT_SCHEMA_VERSION

    # Count items
    conn = _open_db(palace_path)
    try:
        info["drawers"] = conn.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]
        info["wings"] = conn.execute("SELECT COUNT(*) FROM wings").fetchone()[0]
        info["rooms"] = conn.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
    except sqlite3.Error:
        pass
    finally:
        conn.close()

    # FAISS count
    if index_file.exists() and meta_file.exists():
        try:
            store_conn = sqlite3.connect(str(meta_file))
            row = store_conn.execute("SELECT COUNT(*) FROM pos_map").fetchone()
            info["vectors"] = row[0] if row else 0
            store_conn.close()
        except sqlite3.Error:
            info["vectors"] = None
    else:
        info["vectors"] = 0

    return info
