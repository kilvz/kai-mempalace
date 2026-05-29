"""FAISS-specific repair — scan, prune, rebuild, health status.

Adapted from mempalace/repair.py (ChromaDB HNSW → FAISS FlatIP + SQLite).
No onnxruntime dependency — works with NumpyEmbedder only.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

from kai_mempalace.repair_utils import (
    confirm_destructive_action,
    rebuild_fts5,
    run_vacuum,
    sqlite_drawer_count,
    sqlite_integrity_errors,
)

logger = logging.getLogger(__name__)


def _get_palace_path(palace_path: Optional[str] = None) -> str:
    if palace_path:
        return palace_path
    default = os.path.join(os.path.expanduser("~"), ".kai-palace")
    return os.environ.get("KAI_PALACE_PATH", default)


def _open_palace_db(palace_path: str) -> sqlite3.Connection:
    base = Path(palace_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "palace.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _get_data_dir(palace_path: str) -> str:
    return str(Path(palace_path).expanduser().resolve() / "data")


def _open_meta_db(data_dir: str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(Path(data_dir) / "metadata.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


class TruncationDetected(Exception):
    """Raised by check_extraction_safety when extraction looks truncated."""

    def __init__(self, message: str, sqlite_count: Optional[int], extracted: int):
        super().__init__(message)
        self.message = message
        self.sqlite_count = sqlite_count
        self.extracted = extracted


# ── Status ────────────────────────────────────────────────────────────────


def status(palace_path: Optional[str] = None) -> dict:
    """Health check: compare SQLite vs FAISS counts, run integrity checks.

    Returns a dict with ``ok``, ``drawers_sqlite``, ``vectors_faiss``,
    ``integrity_errors``, and a human-readable ``message``.
    """
    palace_path = _get_palace_path(palace_path)
    base = Path(palace_path).expanduser().resolve()

    sqlite_db = base / "palace.db"
    data_dir = base / "data"

    result: dict = {
        "path": str(base),
        "ok": False,
        "drawers_sqlite": 0,
        "vectors_faiss": 0,
        "integrity_errors": [],
        "message": "",
    }

    integrity = sqlite_integrity_errors(str(sqlite_db))
    if integrity:
        result["integrity_errors"] = integrity
        result["message"] = f"SQLite integrity errors: {integrity[0][:80]}"
        return result
    result["drawers_sqlite"] = sqlite_drawer_count(str(sqlite_db)) or 0

    index_path = data_dir / "index.faiss"
    if not index_path.exists():
        result["message"] = "FAISS index not found — palace not initialized"
        return result

    try:
        from kai_mempalace.backends.faiss_store import FaissStore

        store = FaissStore(str(data_dir))
        result["vectors_faiss"] = store.count()
        store.close()
    except Exception as e:
        result["message"] = f"FAISS error: {e}"
        return result

    if result["drawers_sqlite"] != result["vectors_faiss"]:
        result["message"] = (
            f"Mismatch: SQLite {result['drawers_sqlite']} drawers "
            f"vs FAISS {result['vectors_faiss']} vectors"
        )
    else:
        result["ok"] = True
        result["message"] = "Healthy"
    return result


# ── Scan ──────────────────────────────────────────────────────────────────


def scan_palace(palace_path: Optional[str] = None) -> tuple[set[str], set[str]]:
    """Cross-reference SQLite drawers vs FAISS pos_map.

    Returns ``(good_ids, bad_ids)`` where ``bad_ids`` are IDs present in
    one store but not the other. Writes ``corrupt_ids.txt`` to the palace
    directory when bad IDs are found.
    """
    palace_path = _get_palace_path(palace_path)
    base = Path(palace_path).expanduser().resolve()
    data_dir = _get_data_dir(palace_path)
    print(f"\n  Palace: {base}")

    sqlite_ids: set[str] = set()
    db = _open_palace_db(str(base))
    try:
        for row in db.execute("SELECT id FROM drawers"):
            sqlite_ids.add(row[0])
    finally:
        db.close()
    print(f"  SQLite drawers: {len(sqlite_ids):,}")

    if not sqlite_ids:
        print("  Nothing to scan.")
        return set(), set()

    meta_path = Path(data_dir) / "metadata.db"
    if not meta_path.exists():
        print("  metadata.db not found — repair needs re-init")
        return set(), sqlite_ids

    faiss_ids: set[str] = set()
    meta = _open_meta_db(data_dir)
    try:
        for row in meta.execute("SELECT doc_id FROM pos_map"):
            faiss_ids.add(row[0])
    finally:
        meta.close()
    print(f"  FAISS vectors: {len(faiss_ids):,}")

    good = sqlite_ids & faiss_ids
    bad = (sqlite_ids - faiss_ids) | (faiss_ids - sqlite_ids)

    print(f"\n  Scan complete.")
    print(f"  GOOD: {len(good):,}")
    print(f"  BAD:  {len(bad):,}")

    if bad:
        bad_file = base / "corrupt_ids.txt"
        with open(str(bad_file), "w") as f:
            for bid in sorted(bad):
                f.write(bid + "\n")
        print(f"  Bad IDs written to: {bad_file}")

    return good, bad


# ── Prune ─────────────────────────────────────────────────────────────────


def prune_corrupt(palace_path: Optional[str] = None, confirm: bool = False) -> None:
    """Delete corrupt IDs listed in ``corrupt_ids.txt``.

    Removes from both SQLite (``drawers`` + ``drawers_fts``) and FAISS.
    """
    palace_path = _get_palace_path(palace_path)
    base = Path(palace_path).expanduser().resolve()
    data_dir = _get_data_dir(palace_path)
    bad_file = base / "corrupt_ids.txt"

    if not bad_file.exists():
        print("  No corrupt_ids.txt found — run scan first.")
        return

    with open(str(bad_file)) as f:
        bad_ids = [line.strip() for line in f if line.strip()]
    print(f"  {len(bad_ids):,} corrupt IDs queued for deletion")

    if not confirm:
        print("\n  DRY RUN — no deletions performed.")
        print("  Re-run with --confirm to actually delete.")
        return

    db = _open_palace_db(str(base))
    from kai_mempalace.backends.faiss_store import FaissStore

    store = FaissStore(data_dir)
    try:
        before = store.count()
        print(f"  Collection size before: {before:,}")

        batch = 100
        for i in range(0, len(bad_ids), batch):
            chunk = bad_ids[i : i + batch]
            ph = ",".join("?" * len(chunk))
            db.execute(f"DELETE FROM drawers WHERE id IN ({ph})", chunk)
            db.execute(f"DELETE FROM drawers_fts WHERE id IN ({ph})", chunk)
            db.commit()
            store.delete(ids=chunk)

        after = store.count()
        print(f"\n  Deleted: {len(bad_ids):,}")
        print(f"  Collection size: {before:,} \u2192 {after:,}")
        bad_file.unlink()
        print("  corrupt_ids.txt removed.")
    finally:
        store.close()
        db.close()


# ── Extraction safety ─────────────────────────────────────────────────────


def check_extraction_safety(palace_path: str, extracted: int) -> None:
    """Verify extraction count matches SQLite ground truth.

    Raises :class:`TruncationDetected` when extracted < SQLite count.
    """
    sqlite_count = sqlite_drawer_count(
        str(Path(palace_path).expanduser().resolve() / "palace.db")
    )
    if sqlite_count is None:
        print("  WARNING: cannot read SQLite count — skipping safety check")
        return
    if extracted < sqlite_count:
        msg = (
            f"Extraction returned {extracted} drawers but SQLite has "
            f"{sqlite_count}. This may indicate truncation."
        )
        raise TruncationDetected(msg, sqlite_count, extracted)
    print(f"  Extraction verified: {extracted} == {sqlite_count} (SQLite)")


# ── Rebuild index (from stored vectors) ────────────────────────────────────


def rebuild_index(
    palace_path: Optional[str] = None,
    rebuild_fts: bool = True,
    vacuum: bool = True,
) -> int:
    """Rebuild FAISS index from stored vector blobs.

    Uses ``FaissStore._rebuild_index()`` which reads vectors from
    ``metadata.db`` and rebuilds ``index.faiss``. Also optionally
    rebuilds FTS5 and vacuums ``palace.db``.

    Returns the number of vectors rebuilt.
    """
    palace_path = _get_palace_path(palace_path)
    base = Path(palace_path).expanduser().resolve()
    data_dir = _get_data_dir(palace_path)
    palace_db = base / "palace.db"

    from kai_mempalace.backends.faiss_store import FaissStore

    print(f"\n  Rebuilding FAISS index at: {data_dir}")

    store = FaissStore(data_dir)
    try:
        before = store.count()
        print(f"  Vectors before rebuild: {before:,}")

        store._rebuild_index()
        after = store.count()

        print(f"  Vectors after rebuild: {after:,}")

        if rebuild_fts:
            print("  Rebuilding FTS5 index...")
            rebuild_fts5(str(palace_db))

        if vacuum:
            print("  Running VACUUM...")
            run_vacuum(str(palace_db))

        return after
    finally:
        store.close()


# ── Full rebuild from SQLite (re-embed) ────────────────────────────────────


def rebuild_from_sqlite(
    palace_path: Optional[str] = None,
    rebuild_fts: bool = True,
) -> int:
    """Full rebuild: re-embed all drawers from SQLite ground truth.

    Reads every drawer from ``palace.db``, re-embeds with
    :class:`NumpyEmbedder`, and rebuilds the FAISS index from scratch.
    Slower than :func:`rebuild_index` but recovers from corrupted vector
    blobs or dimensional mismatches.

    Returns the number of vectors rebuilt.
    """
    palace_path = _get_palace_path(palace_path)
    base = Path(palace_path).expanduser().resolve()
    data_dir = _get_data_dir(palace_path)
    palace_db = base / "palace.db"

    from kai_mempalace.backends.embedder import NumpyEmbedder
    from kai_mempalace.backends.faiss_store import FaissStore

    embedder = NumpyEmbedder()

    print(f"\n  Full rebuild from SQLite: {base}")

    db = _open_palace_db(str(base))
    try:
        rows = db.execute(
            "SELECT id, content, metadata FROM drawers ORDER BY rowid"
        ).fetchall()
    finally:
        db.close()

    if not rows:
        print("  No drawers found — nothing to rebuild.")
        return 0

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    metadatas = [json.loads(r[2] or "{}") for r in rows]

    print(f"  Loaded {len(ids):,} drawers from SQLite")
    print(f"  Embedding {len(texts):,} texts...")
    t0 = time.time()
    all_embeddings = embedder.embed(texts)
    elapsed = time.time() - t0
    rate = len(texts) / max(elapsed, 0.01)
    print(f"  Embedded in {elapsed:.1f}s ({rate:.0f}/s)")

    store = FaissStore(data_dir)
    try:
        for tbl in ("pos_map", "docs"):
            store._db.execute(f"DELETE FROM {tbl}")
        store._db.commit()

        import faiss

        store.index = faiss.IndexFlatIP(store.dimension)
        store.add(ids=ids, texts=texts, metadatas=metadatas, embeddings=all_embeddings)

        after = store.count()
        print(f"  Rebuilt: {after:,} vectors in FAISS")

        if rebuild_fts:
            print("  Rebuilding FTS5 index...")
            rebuild_fts5(str(palace_db))

        return after
    finally:
        store.close()
