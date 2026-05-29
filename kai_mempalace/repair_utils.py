"""Portable repair utilities — SQLite integrity, VACUUM, FTS5 rebuild, confirmation."""

from __future__ import annotations

import logging
import os
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def sqlite_drawer_count(db_path: str, table: str = "drawers") -> Optional[int]:
    """Count rows in the drawers table as ground truth.

    Returns ``None`` when the DB is unreadable (missing file, locked,
    missing table) — callers treat ``None`` as "unknown".
    """
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def sqlite_integrity_errors(db_path: str) -> list[str]:
    """Run ``PRAGMA quick_check`` and return non-``"ok"`` messages.

    Returns an empty list when the database passes all checks or when
    the file is missing (the latter is not an error — just unrepairable).
    """
    if not os.path.isfile(db_path):
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("PRAGMA quick_check").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        return [f"PRAGMA quick_check failed: {e}"]

    errors: list[str] = []
    for row in rows:
        if not row:
            continue
        msg = str(row[0])
        if msg.lower() != "ok":
            errors.append(msg)
    return errors


def print_integrity_abort(db_path: str, errors: list[str]) -> None:
    """Print a clear repair-abort message for SQLite-layer corruption."""
    preview = errors[:5]
    print(f"\n  ABORT: SQLite-layer corruption detected in {db_path}")
    print("  `PRAGMA quick_check` returned non-OK messages:")
    for msg in preview:
        print(f"    - {msg}")
    if len(errors) > len(preview):
        print(f"    ... and {len(errors) - len(preview)} more issue(s)")
    print()
    print("  Recovery options:")
    print("    1. Stop all writers / MCP clients.")
    print("    2. Back up the palace directory.")
    print("    3. Recover the SQLite database with `.recover` or `.dump`.")
    print("    4. Run `PRAGMA integrity_check` and verify it returns `ok`.")
    print("    5. Re-run the repair operation.")
    print()


def run_vacuum(db_path: str) -> None:
    """VACUUM a SQLite database to reclaim free pages.

    Failures are non-fatal (logged as warnings).
    """
    if not os.path.isfile(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("VACUUM")
            logger.info("SQLite VACUUM complete on %s", db_path)
        finally:
            conn.close()
    except Exception:
        logger.warning("VACUUM failed on %s", db_path, exc_info=True)


def rebuild_fts5(db_path: str, fts_table: str = "drawers_fts") -> None:
    """Rebuild an FTS5 virtual table index.

    After mass deletes or collection rebuilds the FTS5 shadow tables can
    become internally inconsistent; this rebuilds them atomically without
    touching any row data.

    Failures are non-fatal (logged as warnings).
    """
    if not os.path.isfile(db_path):
        return
    try:
        conn = sqlite3.connect(db_path, isolation_level=None)
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if fts_table in tables:
                conn.execute(
                    f"INSERT INTO {fts_table}({fts_table}) VALUES('rebuild')"
                )
                conn.commit()
                logger.info("FTS5 index %s rebuilt", fts_table)
        finally:
            conn.close()
    except Exception:
        logger.warning("FTS5 rebuild failed on %s", db_path, exc_info=True)


def confirm_destructive_action(
    operation: str, path: str, assume_yes: bool = False
) -> bool:
    """Require confirmation before destructive operations.

    Returns ``True`` when the user confirms (or ``assume_yes`` is set).
    """
    if assume_yes:
        return True
    print(f"\n  {operation} will modify data at: {path}")
    try:
        answer = input("  Continue? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted.")
        return False
    if answer not in {"y", "yes"}:
        print("  Aborted.")
        return False
    return True


def close_sqlite_handles(db_path: str) -> None:
    """Force-close any open SQLite connections to a file.

    On Windows, lingering SQLite connections hold file locks that prevent
    directory renames / deletes. This is a best-effort hint; it works
    within the current process but cannot touch connections held by
    other processes.

    The approach: open and close a fresh connection with ``PRAGMA
    shrink_memory``. SQLite's internal connection tracking is opaque,
    so this is advisory — rely on proper connection management in
    production code.
    """
    if not os.path.isfile(db_path):
        return
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA shrink_memory")
        finally:
            conn.close()
        logger.debug("SQLite handle sweep complete on %s", db_path)
    except Exception:
        logger.debug("SQLite handle sweep failed on %s", db_path, exc_info=True)
