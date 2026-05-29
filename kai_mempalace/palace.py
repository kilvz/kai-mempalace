"""Kai MemPalace v3 — FAISS-powered memory palace with wings, rooms, drawers, hybrid search, entity graph, closets."""

import contextlib
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kai_mempalace.backends.embedder import get_embedder
from kai_mempalace.backends.faiss_store import FaissStore
from kai_mempalace.backends.knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}[a-z0-9]$|^[a-z0-9]$")

# Closet infrastructure constants
NORMALIZE_VERSION = 2
CLOSET_CHAR_LIMIT = 2000
CLOSET_EXTRACT_WINDOW = 5000

# Files/dirs to skip during directory walks
SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv", ".opencode", ".claude"})

# Common capitalized words that look like proper nouns but are usually
# sentence-starters or filler. Filtered out of entity extraction.
_ENTITY_STOPLIST = frozenset(
    {
        "The", "This", "That", "These", "Those",
        "When", "Where", "What", "Why", "Who", "Which", "How",
        "After", "Before", "Then", "Now", "Here", "There",
        "And", "But", "Or", "Yet", "So", "If", "Else",
        "Yes", "No", "Maybe", "Okay",
        "User", "Assistant", "System", "Tool",
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    }
)


class MineAlreadyRunning(RuntimeError):
    """Raised when another mine already holds the per-palace lock."""


class MineValidationError(RuntimeError):
    """Raised at end of mine when PRAGMA quick_check reports errors."""

    def __init__(self, palace_path: str, errors: list[str]) -> None:
        if not errors:
            raise ValueError("MineValidationError requires at least one error string")
        if not palace_path:
            raise ValueError("MineValidationError requires a non-empty palace_path")
        super().__init__(f"FTS5/SQLite quick_check failed: {len(errors)} issue(s)")
        self.palace_path = palace_path
        self.errors: tuple[str, ...] = tuple(errors)


@contextlib.contextmanager
def mine_lock(source_file: str):
    """Cross-platform file lock for mine operations."""
    import hashlib
    lock_dir = os.path.join(os.path.expanduser("~"), ".kai-palace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(
        lock_dir, hashlib.sha256(source_file.encode()).hexdigest()[:16] + ".lock"
    )
    lf = open(lock_path, "w")
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lf, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lf, fcntl.LOCK_UN)
        except Exception:
            logger.debug("Mine-lock release failed", exc_info=True)
        lf.close()


# ── Per-palace mine lock (non-blocking, re-entrant) ─────────────────────


import threading as _threading

_palace_lock_holders = _threading.local()


def _holder_state():
    keys = getattr(_palace_lock_holders, "keys", None)
    pid = getattr(_palace_lock_holders, "pid", None)
    current_pid = os.getpid()
    if keys is None or pid != current_pid:
        keys = set()
        _palace_lock_holders.keys = keys
        _palace_lock_holders.pid = current_pid
    return keys


def _held_by_this_thread(lock_key: str) -> bool:
    return lock_key in _holder_state()


def _mark_held(lock_key: str) -> None:
    _holder_state().add(lock_key)


def _mark_released(lock_key: str) -> None:
    _holder_state().discard(lock_key)


_LOCK_SENTINEL_BYTES = 1


def _read_lock_holder(lock_file) -> str:
    try:
        lock_file.seek(_LOCK_SENTINEL_BYTES)
        content = lock_file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        content = content.strip()
    except OSError:
        return "another writer (identity not recorded)"
    if not content:
        return "another writer (identity not recorded)"
    parts = content.split(maxsplit=1)
    pid = parts[0] if parts else "?"
    cmd = parts[1].strip() if len(parts) > 1 else ""
    return f"PID {pid} ({cmd})" if cmd else f"PID {pid}"


def _write_lock_holder(lock_file) -> None:
    try:
        import sys as _sys
        ident = f"{os.getpid()} {' '.join(_sys.argv[:3])}".strip()
        ident_bytes = ident.encode("utf-8")
        lock_file.seek(_LOCK_SENTINEL_BYTES)
        lock_file.truncate(_LOCK_SENTINEL_BYTES + len(ident_bytes))
        lock_file.write(ident_bytes)
        lock_file.flush()
    except (OSError, UnicodeError):
        pass


@contextlib.contextmanager
def mine_palace_lock(palace_path: str):
    """Per-palace non-blocking lock around the full mine pipeline.

    Non-blocking: raises MineAlreadyRunning if another mine is active on
    this palace. Re-entrant: same thread passes through without re-acquiring.
    """
    lock_dir = os.path.join(os.path.expanduser("~"), ".kai-palace", "locks")
    os.makedirs(lock_dir, exist_ok=True)
    resolved = os.path.realpath(os.path.expanduser(palace_path))
    lock_key_source = os.path.normcase(resolved)
    import hashlib as _hashlib
    palace_key = _hashlib.sha256(lock_key_source.encode()).hexdigest()[:16]
    lock_path = os.path.join(lock_dir, f"mine_palace_{palace_key}.lock")

    if _held_by_this_thread(palace_key):
        yield
        return

    if not os.path.exists(lock_path):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
    lf = open(lock_path, "r+b")
    acquired = False
    try:
        lf.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError as exc:
                holder = _read_lock_holder(lf)
                raise MineAlreadyRunning(
                    f"palace {resolved} is held by {holder}; "
                    "wait for it to finish or stop the holder before retrying"
                ) from exc
        else:
            import fcntl
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError as exc:
                holder = _read_lock_holder(lf)
                raise MineAlreadyRunning(
                    f"palace {resolved} is held by {holder}; "
                    "wait for it to finish or stop the holder before retrying"
                ) from exc
        _write_lock_holder(lf)
        _mark_held(palace_key)
        try:
            yield
        finally:
            _mark_released(palace_key)
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt
                    lf.seek(0)
                    msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lf, fcntl.LOCK_UN)
            except Exception:
                pass
        lf.close()


def _validate_palace_fts5_after_mine(palace_path: str) -> None:
    """Raise MineValidationError if SQLite quick_check reports errors after a mine."""
    from kai_mempalace.repair_utils import sqlite_integrity_errors
    errors = sqlite_integrity_errors(str(Path(palace_path).expanduser() / "palace.db"))
    if errors:
        raise MineValidationError(palace_path, errors)


def bulk_check_mined(palace_path: str) -> dict[str, float]:
    """Return dict mapping source_file -> source_mtime for all drawers."""
    base = Path(palace_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "palace.db"))
    try:
        rows = conn.execute(
            "SELECT DISTINCT source_file, json_extract(metadata, '$.source_mtime') as mtime "
            "FROM drawers WHERE source_file != '' AND source_file IS NOT NULL"
        ).fetchall()
        return {row[0]: float(row[1]) for row in rows if row[1] is not None}
    except (sqlite3.Error, ValueError, TypeError):
        return {}
    finally:
        conn.close()


def prefetch_mined_set(palace_path: str) -> set[str]:
    """Return set of source_file paths already mined at current normalize_version."""
    base = Path(palace_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "palace.db"))
    try:
        rows = conn.execute(
            "SELECT source_file FROM drawers WHERE source_file != '' AND source_file IS NOT NULL"
        ).fetchall()
        return {row[0] for row in rows}
    except sqlite3.Error:
        return set()
    finally:
        conn.close()


def _sanitize(name: str, kind: str = "name") -> str:
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    if not _SAFE_NAME.match(name):
        raise ValueError(f"Invalid {kind}: {name!r}")
    return name


@dataclass
class SearchResult:
    id: str
    text: str
    distance: float
    metadata: dict
    wing: str = ""
    room: str = ""


class Palace:
    """Main memory palace - wings, rooms, drawers, search."""

    def __init__(self, path: str = "~/.kai-palace"):
        self._base = Path(path).expanduser().resolve()
        self._data_dir = self._base / "data"
        self._lock = threading.Lock()
        self._initialized = False

        self._db: Optional[sqlite3.Connection] = None
        self._store: Optional[FaissStore] = None
        self._embedder: Optional[Any] = None
        self._kg: Optional[KnowledgeGraph] = None

        # Hybrid search state (FTS5)
        self._fts_enabled = False

    def init(self) -> bool:
        """Initialize or open the palace. Returns True if newly created."""
        self._base.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        is_new = not (self._base / "palace.json").exists()

        self._db = sqlite3.connect(str(self._base / "palace.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS wings (
                name TEXT PRIMARY KEY, description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS rooms (
                name TEXT, wing TEXT, description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (name, wing),
                FOREIGN KEY (wing) REFERENCES wings(name))""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS drawers (
                id TEXT PRIMARY KEY, wing TEXT NOT NULL, room TEXT NOT NULL,
                content TEXT NOT NULL, metadata TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (wing, room) REFERENCES rooms(wing, name))""")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_drawers_wing_room ON drawers(wing, room)")
        self._db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS drawers_fts USING fts5(content, id)")
        self._db.execute("""CREATE TABLE IF NOT EXISTS closets (
                id TEXT PRIMARY KEY, content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '', wing TEXT DEFAULT '', room TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_closets_source ON closets(source_file)")
        self._db.commit()

        self._store = FaissStore(str(self._data_dir))
        self._embedder = self._load_embedder(is_new=is_new)
        self._kg = KnowledgeGraph(str(self._data_dir))
        self._fts_enabled = True

        if is_new:
            with open(self._base / "palace.json", "w") as f:
                json.dump({
                    "version": 3, "created_at": datetime.utcnow().isoformat(),
                    "backend": "faiss",
                    "embedding": self._embedder_config_name(self._embedder),
                    "dimension": 384,
                }, f)

        self._initialized = True
        logger.info("Palace %s at %s", "created" if is_new else "opened", self._base)
        return is_new

    def close(self) -> None:
        self._save_embedder()
        if self._store:
            self._store.close()
        if self._kg:
            self._kg.close()
        if self._db:
            self._db.close()
        self._initialized = False

    def _save_embedder(self):
        if self._embedder and self._embedder.is_fitted:
            save = getattr(self._embedder, "save", None)
            if save:
                try:
                    save(str(self._data_dir))
                except Exception:
                    pass

    def _load_embedder(self, is_new: bool = False) -> Any:
        config_path = self._base / "palace.json"
        model = "sentence"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            model = config.get("embedding", "numpy")
            mapping = {
                "numpy_tfidf_svd": "numpy",
                "sentence_transformers": "sentence",
                "spacy_glove": "spacy",
            }
            model = mapping.get(model, model)
        else:
            from kai_mempalace.config import KaiPalaceConfig
            model = KaiPalaceConfig().default_embedder
        model = self._resolve_embedder(model, config_path)
        return get_embedder(model=model, model_dir=str(self._data_dir))

    @staticmethod
    def _resolve_embedder(model: str, config_path: Path) -> str:
        """Check if the configured embedder's dependency is available; fall back to numpy if not."""
        if model == "sentence":
            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer
            except ImportError:
                logger.warning("sentence_transformers not installed — falling back to numpy. Install with: pip install sentence-transformers")
                model = "numpy"
        elif model == "spacy":
            try:
                import spacy
                spacy
            except ImportError:
                logger.warning("spacy not installed — falling back to numpy. Install with: pip install spacy")
                model = "numpy"
        elif model == "bert":
            try:
                import onnxruntime
                onnxruntime
            except ImportError:
                logger.info("onnxruntime not available for bert — using numpy backend")
        elif model == "minilm" or model == "embeddinggemma":
            try:
                import onnxruntime
                onnxruntime
            except ImportError:
                logger.warning("onnxruntime not installed for %s — falling back to numpy", model)
                model = "numpy"
        if model not in ("sentence", "bert") and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            if config.get("embedding") != "numpy_tfidf_svd":
                config["embedding"] = "numpy_tfidf_svd"
                with open(config_path, "w") as f:
                    json.dump(config, f)
        return model

    @staticmethod
    def _embedder_config_name(embedder: Any) -> str:
        """Map embedder instance back to its palace.json config name."""
        cls_name = type(embedder).__name__
        mapping = {
            "NumpyEmbedder": "numpy_tfidf_svd",
            "SentenceTransformerEmbedder": "sentence_transformers",
            "SpacyGloveEmbedder": "spacy_glove",
            "OnnxEmbedder": "minilm",
            "EmbeddinggemmaONNX": "embeddinggemma",
            "NumpyBertEmbedder": "bert",
        }
        return mapping.get(cls_name, "numpy_tfidf_svd")

    @property
    def kg(self) -> KnowledgeGraph:
        if not self._kg:
            raise RuntimeError("Palace not initialized")
        return self._kg

    # -- Wing management --

    def list_wings(self) -> list[dict]:
        rows = self._db.execute("""SELECT w.name, w.description, w.created_at,
                COUNT(d.id) as drawer_count FROM wings w
                LEFT JOIN drawers d ON d.wing = w.name
                GROUP BY w.name ORDER BY w.name""").fetchall()
        return [{"name": r[0], "description": r[1], "created_at": r[2], "drawer_count": r[3]}
                for r in rows]

    def get_or_create_wing(self, name: str, description: str = "") -> str:
        name = _sanitize(name, "wing")
        self._db.execute("INSERT OR IGNORE INTO wings (name, description) VALUES (?, ?)",
                         (name, description))
        self._db.commit()
        return name

    def delete_wing(self, name: str) -> bool:
        name = _sanitize(name, "wing")
        # Delete all drawers and rooms in this wing
        self._db.execute("DELETE FROM drawers WHERE wing = ?", (name,))
        self._db.execute("DELETE FROM rooms WHERE wing = ?", (name,))
        self._db.execute("DELETE FROM wings WHERE name = ?", (name,))
        self._db.commit()
        return True

    # -- Room management --

    def list_rooms(self, wing: Optional[str] = None) -> list[dict]:
        if wing:
            rows = self._db.execute("""SELECT r.name, r.wing, r.description, r.created_at,
                    COUNT(d.id) as drawer_count FROM rooms r
                    LEFT JOIN drawers d ON d.room = r.name AND d.wing = r.wing
                    WHERE r.wing = ? GROUP BY r.name, r.wing ORDER BY r.name""", (wing,))
        else:
            rows = self._db.execute("""SELECT r.name, r.wing, r.description, r.created_at,
                    COUNT(d.id) as drawer_count FROM rooms r
                    LEFT JOIN drawers d ON d.room = r.name AND d.wing = r.wing
                    GROUP BY r.name, r.wing ORDER BY r.wing, r.name""")
        return [{"name": r[0], "wing": r[1], "description": r[2], "created_at": r[3],
                 "drawer_count": r[4]} for r in rows.fetchall()]

    def get_or_create_room(self, wing: str, name: str, description: str = "") -> str:
        wing = _sanitize(wing, "wing")
        name = _sanitize(name, "room")
        self.get_or_create_wing(wing)
        self._db.execute("INSERT OR IGNORE INTO rooms (name, wing, description) VALUES (?, ?, ?)",
                         (name, wing, description))
        self._db.commit()
        return name

    def delete_room(self, wing: str, name: str) -> bool:
        wing = _sanitize(wing, "wing")
        name = _sanitize(name, "room")
        self._db.execute("DELETE FROM drawers WHERE wing = ? AND room = ?", (wing, name))
        self._db.execute("DELETE FROM rooms WHERE wing = ? AND name = ?", (wing, name))
        self._db.commit()
        return True

    # -- Drawer operations --

    def add_drawer(self, wing: str, room: str, content: str,
                   metadata: Optional[dict] = None,
                   source_file: str = "",
                   drawer_id: Optional[str] = None) -> str:
        if not content.strip():
            raise ValueError("Content cannot be empty")
        wing = _sanitize(wing, "wing")
        room = _sanitize(room, "room")
        self.get_or_create_room(wing, room)
        if drawer_id is None:
            drawer_id = self._store.next_id()
        meta = dict(metadata or {})
        meta["wing"] = wing
        meta["room"] = room
        embedding = self._embedder.embed([content])[0]
        embedding_2d = embedding.reshape(1, -1)
        self._store.add(ids=[drawer_id], texts=[content], metadatas=[meta],
                        embeddings=embedding_2d)
        self._db.execute(
            "INSERT OR REPLACE INTO drawers (id, wing, room, content, metadata, source_file) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (drawer_id, wing, room, content, json.dumps(meta), source_file))
        self._db.execute(
            "INSERT OR REPLACE INTO drawers_fts (id, content) VALUES (?, ?)",
            (drawer_id, content))
        self._db.commit()
        self._save_embedder()
        logger.debug("Added drawer %s in %s/%s", drawer_id, wing, room)
        return drawer_id

    def get_drawer(self, drawer_id: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT id, wing, room, content, metadata, source_file, created_at "
            "FROM drawers WHERE id = ?", (drawer_id,)).fetchone()
        if not row:
            return None
        return {"id": row[0], "wing": row[1], "room": row[2], "content": row[3],
                "metadata": json.loads(row[4] or "{}"), "source_file": row[5],
                "created_at": row[6]}

    def list_drawers(self, wing: Optional[str] = None, room: Optional[str] = None,
                     limit: int = 20, offset: int = 0) -> list[dict]:
        conditions = []
        params = []
        if wing:
            conditions.append("wing = ?")
            params.append(wing)
        if room:
            conditions.append("room = ?")
            params.append(room)
        sql = "SELECT id, wing, room, content, metadata, source_file, created_at FROM drawers"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._db.execute(sql, params).fetchall()
        return [{"id": r[0], "wing": r[1], "room": r[2],
                 "content": r[3][:500] + ("..." if len(r[3]) > 500 else ""),
                 "metadata": json.loads(r[4] or "{}"), "source_file": r[5],
                 "created_at": r[6]} for r in rows]

    def delete_drawer(self, drawer_id: str) -> bool:
        row = self._db.execute("SELECT id FROM drawers WHERE id = ?", (drawer_id,)).fetchone()
        if not row:
            return False
        self._db.execute("DELETE FROM drawers WHERE id = ?", (drawer_id,))
        self._db.execute("DELETE FROM drawers_fts WHERE id = ?", (drawer_id,))
        self._db.commit()
        self._store.delete(ids=[drawer_id])
        return True

    def update_drawer(self, drawer_id: str, content: Optional[str] = None,
                      metadata: Optional[dict] = None,
                      wing: Optional[str] = None,
                      room: Optional[str] = None) -> bool:
        existing = self.get_drawer(drawer_id)
        if not existing:
            return False
        new_content = content if content is not None else existing["content"]
        new_meta = dict(existing["metadata"])
        if metadata:
            new_meta.update(metadata)
        if wing:
            new_meta["wing"] = _sanitize(wing, "wing")
        if room:
            new_meta["room"] = _sanitize(room, "room")
        if content is not None:
            embedding = self._embedder.embed([new_content])[0]
            embedding_2d = embedding.reshape(1, -1)
            self._store.upsert(ids=[drawer_id], texts=[new_content],
                               metadatas=[new_meta], embeddings=embedding_2d)
            self._db.execute("DELETE FROM drawers_fts WHERE id = ?", (drawer_id,))
            self._db.execute("INSERT OR REPLACE INTO drawers_fts (id, content) VALUES (?, ?)",
                             (drawer_id, new_content))
        new_wing = new_meta.get("wing", existing["wing"])
        new_room = new_meta.get("room", existing["room"])
        self._db.execute(
            "UPDATE drawers SET content=?, metadata=?, wing=?, room=? WHERE id=?",
            (new_content, json.dumps(new_meta), new_wing, new_room, drawer_id))
        self._db.commit()
        self._save_embedder()
        return True

    # -- Search (hybrid: FTS5 BM25 + FAISS vector) --

    def search(self, query: str, n_results: int = 10,
               wing: Optional[str] = None, room: Optional[str] = None,
               mode: str = "hybrid") -> list[SearchResult]:
        """Search across all drawers. mode: 'vector', 'keyword', or 'hybrid'."""
        if not self._store or self._store.count() == 0:
            return []

        if mode == "keyword":
            return self._keyword_search(query, n_results, wing, room)
        elif mode == "vector":
            return self._vector_search(query, n_results, wing, room)
        else:
            return self._hybrid_search(query, n_results, wing, room)

    def _vector_search(self, query: str, n_results: int = 10,
                       wing: Optional[str] = None, room: Optional[str] = None) -> list[SearchResult]:
        query_emb = self._embedder.embed([query])[0]
        ids, texts, distances, metadatas = self._store.search(query_emb, n_results=n_results * 2)
        results = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            dw = meta.get("wing", "")
            dr = meta.get("room", "")
            if wing and dw != wing:
                continue
            if room and dr != room:
                continue
            results.append(SearchResult(id=ids[i], text=texts[i],
                            distance=float(distances[i]) if i < len(distances) else 0.0,
                            metadata=meta, wing=dw, room=dr))
            if len(results) >= n_results:
                break
        return results

    def _keyword_search(self, query: str, n_results: int = 10,
                        wing: Optional[str] = None, room: Optional[str] = None) -> list[SearchResult]:
        if not self._fts_enabled:
            return self._vector_search(query, n_results, wing, room)
        fts_query = self._build_fts_query(query)
        sql = """SELECT d.id, d.wing, d.room, d.content, d.metadata,
                 rank FROM drawers_fts f
                 JOIN drawers d ON d.id = f.id
                 WHERE drawers_fts MATCH ?"""
        params = [fts_query]
        if wing:
            sql += " AND d.wing = ?"
            params.append(wing)
        if room:
            sql += " AND d.room = ?"
            params.append(room)
        sql += " ORDER BY rank LIMIT ?"
        params.append(n_results)
        try:
            rows = self._db.execute(sql, params).fetchall()
        except Exception:
            return self._vector_search(query, n_results, wing, room)
        results = []
        for r in rows:
            meta = json.loads(r[4] or "{}")
            results.append(SearchResult(id=r[0], text=r[3], distance=1.0 - float(r[5]),
                            metadata=meta, wing=r[1], room=r[2]))
        return results

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Build an FTS5 query string with prefix matching for bare terms.

        FTS5 natively supports AND, OR, NOT, NEAR/N, ``"phrase"``, and
        ``term*`` prefix syntax. If the query already contains any of these
        operators it is passed through verbatim. Bare space-separated terms
        gain ``*`` prefix wildcards so partial input matches.
        """
        tokens = query.strip().split()
        if not tokens:
            return query
        for t in tokens:
            upper = t.upper()
            if upper in ("AND", "OR", "NOT") or upper.startswith("NEAR"):
                return query
        if any(t.startswith('"') for t in tokens):
            return query
        return " AND ".join(t + "*" for t in tokens)

    def _hybrid_search(self, query: str, n_results: int = 10,
                        wing: Optional[str] = None, room: Optional[str] = None) -> list[SearchResult]:
        vec_results = self._vector_search(query, n_results * 3, wing, room)
        kw_results = self._keyword_search(query, n_results * 3, wing, room)

        seen = set()
        combined: list[SearchResult] = []
        for r in vec_results:
            seen.add(r.id)
            combined.append(r)
        for r in kw_results:
            if r.id not in seen:
                combined.append(r)

        combined.sort(key=lambda x: -x.distance)

        # Closet boost: if any drawer's source_file is referenced in the
        # closets table, boost its rank by 0.15.
        source_files = set()
        for r in combined:
            src = r.metadata.get("source_file") if r.metadata else None
            if src:
                source_files.add(src)
        if source_files:
            boosted_ids = self._get_closet_source_ids(source_files)
            if boosted_ids:
                for i, r in enumerate(combined):
                    if r.id in boosted_ids:
                        combined[i] = SearchResult(
                            id=r.id, text=r.text,
                            distance=min(1.0, r.distance + 0.15),
                            metadata=r.metadata, wing=r.wing, room=r.room,
                        )

        combined.sort(key=lambda x: -x.distance)
        return combined[:n_results]

    def _get_closet_source_ids(self, source_files: set[str]) -> set[str]:
        """Return drawer IDs whose source_file is referenced in closets."""
        try:
            placeholders = ",".join("?" * len(source_files))
            rows = self._db.execute(
                f"SELECT DISTINCT drawer_id FROM closets "
                f"WHERE source_file IN ({placeholders})",
                list(source_files),
            ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    # -- FTS maintenance --

    def rebuild_fts(self) -> None:
        """Rebuild FTS index from all drawer contents."""
        self._db.execute("DELETE FROM drawers_fts")
        rows = self._db.execute("SELECT id, content FROM drawers").fetchall()
        for rid, content in rows:
            self._db.execute("INSERT INTO drawers_fts (id, content) VALUES (?, ?)",
                             (rid, content))
        self._db.commit()
        logger.info("Rebuilt FTS index with %d drawers", len(rows))

    # -- Status --

    def status(self) -> dict:
        if not self._initialized:
            return {"initialized": False}
        wings = self.list_wings()
        total_drawers = self._db.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]
        total_rooms = self._db.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
        return {"initialized": True, "path": str(self._base), "wings": len(wings),
                "rooms": total_rooms, "drawers": total_drawers,
                "wings_detail": wings, "embedding": "numpy_tfidf_svd_384",
                "fts_enabled": self._fts_enabled}

    # -- Agent diary --

    def diary_write(self, agent_name: str, entry: str, topic: str = "general",
                    wing: str = "") -> str:
        target_wing = wing or f"agent_{_sanitize(agent_name)}"
        self.add_drawer(wing=target_wing, room="diary", content=entry,
                        metadata={"agent": agent_name, "topic": topic, "type": "diary"})
        return target_wing

    def diary_read(self, agent_name: str, last_n: int = 10, wing: str = "") -> list[dict]:
        target_wing = wing or f"agent_{_sanitize(agent_name)}"
        return self.list_drawers(wing=target_wing, room="diary", limit=last_n)

    # -- Duplicate check --

    def check_duplicate(self, content: str, threshold: float = 0.9) -> Optional[dict]:
        if self._store.count() == 0:
            return None
        query_emb = self._embedder.embed([content])[0]
        ids, texts, distances, metadatas = self._store.search(query_emb, n_results=1)
        if ids and distances[0] >= threshold:
            return {"id": ids[0], "text": texts[0], "similarity": distances[0],
                    "metadata": metadatas[0] if metadatas else {}}
        return None

    # -- Taxonomy --

    def get_taxonomy(self) -> dict:
        """Full taxonomy tree: wing → room → drawer_count."""
        rows = self._db.execute(
            "SELECT wing, room, COUNT(*) as cnt FROM drawers GROUP BY wing, room ORDER BY wing, room"
        ).fetchall()
        tree = {}
        for wing, room, cnt in rows:
            tree.setdefault(wing, {})[room] = cnt
        return tree

    # -- Reconnect --

    def reconnect(self) -> bool:
        """Re-initialize database, FAISS, and KG connections in-place."""
        self._save_embedder()
        if self._store:
            self._store.close()
        if self._kg:
            self._kg.close()
        if self._db:
            self._db.close()
        self._db = sqlite3.connect(str(self._base / "palace.db"))
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._fts_enabled = bool(self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='drawers_fts'"
        ).fetchone())
        from kai_mempalace.backends.faiss_store import FaissStore
        from kai_mempalace.backends.knowledge_graph import KnowledgeGraph
        self._store = FaissStore(str(self._data_dir))
        self._embedder = self._load_embedder()
        self._kg = KnowledgeGraph(str(self._data_dir))
        self._initialized = True
        return True

    # -- Embedder management --

    _EMBEDDER_CONFIG_NAMES: dict[str, str] = {
        "numpy": "numpy_tfidf_svd",
        "numpy_tfidf_svd": "numpy_tfidf_svd",
        "sentence": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
        "spacy": "spacy_glove",
        "spacy_glove": "spacy_glove",
        "minilm": "minilm",
        "onnx": "minilm",
        "embeddinggemma": "embeddinggemma",
        "gemma": "embeddinggemma",
        "bert": "bert",
        "numpy_bert": "bert",
    }

    def set_embedder(self, model: str, reindex: bool = True) -> dict:
        """Switch to a different embedding model.

        ``model`` accepts short names (``"sentence"``, ``"spacy"``,
        ``"numpy"``) or config names (``"sentence_transformers"``,
        ``"spacy_glove"``, ``"numpy_tfidf_svd"``).

        When ``reindex=True``, all existing drawers are re-embedded with
        the new model and the FAISS index is rebuilt.
        """
        config_name = self._EMBEDDER_CONFIG_NAMES.get(model)
        if config_name is None:
            valid = sorted(set(self._EMBEDDER_CONFIG_NAMES.values()))
            raise ValueError(
                f"Unknown embedder model: {model!r}. "
                f"Valid options: {valid}"
            )

        # Write new config
        config_path = self._base / "palace.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        config["embedding"] = config_name
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Load new embedder
        self._save_embedder()
        self._embedder = self._load_embedder()

        info = {
            "model": config_name,
            "dimension": self._embedder.dimension,
        }

        if reindex:
            count = self._reindex_embeddings()
            info["reindexed"] = count

        return info

    def _reindex_embeddings(self) -> int:
        """Re-embed all drawers with the current embedder and rebuild FAISS."""
        rows = self._db.execute(
            "SELECT id, content, metadata, source_file FROM drawers"
        ).fetchall()
        if not rows:
            return 0

        ids = [r[0] for r in rows]
        contents = [r[1] for r in rows]
        vectors = self._embedder.embed(contents)
        metadatas = []
        for r in rows:
            md = json.loads(r[2]) if isinstance(r[2], str) and r[2] else {}
            md["source_file"] = r[3] or ""
            metadatas.append(md)

        from kai_mempalace.backends.faiss_store import FaissStore
        if self._store:
            self._store.close()
        for f in ["index.faiss", "metadata.db", "seq.txt"]:
            (self._data_dir / f).unlink(missing_ok=True)
        self._store = FaissStore(str(self._data_dir))
        self._store.add(ids, contents, metadatas, vectors)

        return len(ids)

    # -- Memories filed away --

    def memories_filed_away(self) -> dict:
        """Check when the last memory was saved and total counts."""
        last = self._db.execute(
            "SELECT created_at, content FROM drawers ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        total = self._db.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]
        preview = None
        last_time = None
        if last:
            try:
                last_time = last["created_at"]
                content = last["content"]
            except (TypeError, IndexError):
                last_time = last[0]
                content = last[1] if len(last) > 1 else ""
            if content:
                preview = (content[:100] + "...") if len(content) > 100 else content
        return {
            "total_drawers": total,
            "last_saved_at": last_time,
            "last_content_preview": preview,
        }


# ==================== Closet-line entity extraction (for diary_ingest) ====================


def _candidate_entity_words(text: str) -> list[str]:
    from kai_mempalace.entity_detector import _get_coca_filter
    from kai_mempalace.i18n import get_entity_patterns
    from kai_mempalace.entity_detector import _apply_known_systems_prepass

    coca = _get_coca_filter()
    ep = get_entity_patterns()
    stopwords = frozenset(w.lower() for w in ep.get("stopwords", []))

    # Apply known-systems prepass — replaces recognized system names with spaces
    # so they won't be picked up again as generic entities.
    cleaned, _ = _apply_known_systems_prepass(text)

    result: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b[A-Z][a-zA-Z'\-]{1,50}\b", cleaned):
        word = m.group()
        low = word.lower()
        if word in _ENTITY_STOPLIST:
            continue
        if low in stopwords:
            continue
        if low in coca:
            continue
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


_DATE_LINE = re.compile(
    r"^(\s*(?:[-*]\s+)?)"                          # optional list prefix
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}"                # YYYY-MM-DD or YYYY/MM/DD
    r"|\d{1,2}[-/]\d{1,2}[-/]\d{4})"               # MM/DD/YYYY or DD/MM/YYYY
    r"(\s*[:\-]\s*.+)?$",                           # optional text after colon/dash
    re.MULTILINE,
)


def _build_date_line_segment(text: str) -> str:
    """Collapse date-prefixed lines into a compact entry."""
    lines = text.strip().splitlines()
    clean: list[str] = []
    for line in lines:
        m = _DATE_LINE.match(line)
        if m:
            prefix = m.group(1) or ""
            date = m.group(2) or ""
            rest = (m.group(3) or "").strip()
            if rest:
                rest = rest.lstrip(":-\t ")
            clean.append(f"{prefix}{date} | {rest}".strip())
        else:
            clean.append(line.strip())
    return "\n".join(clean)


def build_closet_lines(
    text: str,
    existing: dict[str, str],
    source_line: Optional[str] = None,
) -> list[dict]:
    """Build closet-line entries from text, grouping named-entity snippets.

    Parameters
    ----------
    text : str
        Source text to mine.
    existing : dict[str, str]
        Existing closet lines keyed by entity name — maps to their current
        accumulated content.
    source_line : str, optional
        Optional source-file line for provenance.

    Returns
    -------
    list[dict]
        List of ``{entity: str, content: str}`` dicts. These should be
        upserted by the caller.
    """
    if not text.strip():
        return [{"entity": "_meta", "content": "\n"}]

    entities = _candidate_entity_words(text)

    # If no entities found, fall back to "_meta" with collapsed dates.
    if not entities:
        collapsed = _build_date_line_segment(text)
        if len(collapsed) > CLOSET_CHAR_LIMIT:
            collapsed = collapsed[:CLOSET_CHAR_LIMIT]
        existing_meta = existing.get("_meta", "")
        merged = existing_meta + "\n" + collapsed if existing_meta else collapsed
        return [{"entity": "_meta", "content": merged.strip()}]

    result: list[dict] = []
    seen_entities: set[str] = set()
    for e in entities:
        if e in seen_entities:
            continue
        seen_entities.add(e)
        window_start = max(0, text.lower().find(e.lower()) - CLOSET_EXTRACT_WINDOW)
        window_end = min(len(text), window_start + 2 * CLOSET_EXTRACT_WINDOW + len(e))
        snippet = text[window_start:window_end].strip()
        if len(snippet) > CLOSET_CHAR_LIMIT:
            snippet = snippet[:CLOSET_CHAR_LIMIT]
        existing_content = existing.get(e, "")
        merged = existing_content + "\n" + snippet if existing_content else snippet
        result.append({"entity": e, "content": merged.strip()})
    return result


# ==================== Standalone collection helpers (for closet_llm etc.) ====================


def _open_palace_db(palace_path: str) -> sqlite3.Connection:
    base = Path(palace_path).expanduser().resolve()
    conn = sqlite3.connect(str(base / "palace.db"), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS closets (
            id TEXT PRIMARY KEY, content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            source_file TEXT DEFAULT '', wing TEXT DEFAULT '', room TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_closets_source ON closets(source_file)")
    conn.commit()
    return conn


class _DrawerCollection:
    """Minimal ChromaDB-compatible wrapper around the drawers SQLite table."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]

    def get(self, limit: int = 5000, offset: int = 0, include: Optional[list[str]] = None):
        rows = self._conn.execute(
            "SELECT id, content, metadata FROM drawers ORDER BY created_at LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows] if (not include or "documents" in include) else [],
            "metadatas": [json.loads(r[2] or "{}") for r in rows]
            if (not include or "metadatas" in include)
            else [],
        }


class _ClosetCollection:
    """Minimal ChromaDB-compatible wrapper around the closets SQLite table."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def delete(self, where: Optional[dict] = None) -> None:
        if where and "source_file" in where:
            self._conn.execute("DELETE FROM closets WHERE source_file = ?", (where["source_file"],))
            self._conn.commit()

    def upsert(self, documents: list[str], ids: list[str], metadatas: Optional[list[dict]] = None) -> None:
        if metadatas is None:
            metadatas = [{}] * len(documents)
        for doc, cid, meta in zip(documents, ids, metadatas):
            self._conn.execute(
                "INSERT OR REPLACE INTO closets (id, content, metadata, source_file, wing, room) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    doc,
                    json.dumps(meta),
                    meta.get("source_file", ""),
                    meta.get("wing", ""),
                    meta.get("room", ""),
                ),
            )
        self._conn.commit()


def get_collection(
    palace_path: str,
    collection_name: Optional[str] = None,
    create: bool = True,
) -> _DrawerCollection:
    _ = collection_name  # kept for API compat; we only have one collection
    conn = _open_palace_db(palace_path)
    return _DrawerCollection(conn)


def get_closets_collection(palace_path: str, create: bool = True) -> _ClosetCollection:
    conn = _open_palace_db(palace_path)
    return _ClosetCollection(conn)


def file_already_mined(db_or_conn, source_file: str, check_mtime: bool = False, extract_mode: Optional[str] = None) -> bool:
    """Check if a source file has already been mined (has sentinel or closets rows).

    Parameters
    ----------
    db_or_conn :
        Either a Palace instance (with ``_db`` attribute) or a raw sqlite3 Connection.
    source_file :
        File path to check.
    check_mtime :
        If True, also re-mine if the source file's mtime has changed (i.e. the file
        was updated after the last mine).
    extract_mode :
        If set to "format", scope the check to format-mode sentinels.
    """
    try:
        if hasattr(db_or_conn, "_db"):
            conn = db_or_conn._db
        else:
            conn = db_or_conn
        rows = conn.execute(
            "SELECT metadata FROM closets WHERE source_file = ? LIMIT 1",
            (source_file,),
        ).fetchall()
        if not rows:
            return False
        if check_mtime:
            try:
                current_mtime = os.path.getmtime(source_file)
            except OSError:
                return True
            meta = json.loads(rows[0][0] or "{}")
            stored_mtime = meta.get("source_mtime")
            if stored_mtime is not None and current_mtime > stored_mtime:
                return False
        return True
    except Exception:
        return False


def purge_file_closets(closets_col, source_file: str) -> None:
    try:
        closets_col.delete(where={"source_file": source_file})
    except Exception:
        logger.debug("Closet purge failed for %s", source_file, exc_info=True)


def upsert_closet_lines(closets_col, closet_id_base, lines, metadata):
    closet_num = 1
    current_lines: list = []
    current_chars = 0
    closets_written = 0

    def _flush():
        nonlocal closets_written
        if not current_lines:
            return
        closet_id = f"{closet_id_base}_{closet_num:02d}"
        text = "\n".join(current_lines)
        closets_col.upsert(documents=[text], ids=[closet_id], metadatas=[metadata])
        closets_written += 1

    for line in lines:
        line_len = len(line)
        if current_chars > 0 and current_chars + line_len + 1 > CLOSET_CHAR_LIMIT:
            _flush()
            closet_num += 1
            current_lines = []
            current_chars = 0
        current_lines.append(line)
        current_chars += line_len + 1
    _flush()
    return closets_written
