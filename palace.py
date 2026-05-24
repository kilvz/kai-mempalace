"""
Kai MemPalace — lightweight FAISS-powered memory palace.

Architecture (same as MemPalace):
  Wings  → people or projects (e.g. "zeth", "kai-9000")
  Rooms  → specific topics (e.g. "database-setup", "preferences")
  Drawers → individual memory chunks (verbatim text with metadata)

Storage:
  - FAISS index for vector similarity search
  - SQLite for metadata, wings/rooms/drawers mapping
  - numpy/scipy embeddings (TF-IDF + SVD)

Usage:
    palace = Palace("~/.kai-palace")
    palace.init()
    palace.add_drawer("wing_zeth", "room_preferences", "Prefers dark mode", {"source": "convo"})
    results = palace.search("dark mode preferences")
"""

import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from embedder import NumpyEmbedder, get_embedder
from faiss_store import FaissStore
from knowledge_graph import KnowledgeGraph

logger = logging.getLogger(__name__)

# Regex: wing_xxx, room_yyy
_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,62}[a-z0-9]$|^[a-z0-9]$")


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
    """Main memory palace — wings, rooms, drawers, search."""

    def __init__(self, path: str = "~/.kai-palace"):
        self._base = Path(path).expanduser().resolve()
        self._data_dir = self._base / "data"
        self._lock = threading.Lock()
        self._initialized = False

        # Sub-components (lazily initialized)
        self._db: Optional[sqlite3.Connection] = None
        self._store: Optional[FaissStore] = None
        self._embedder: Optional[NumpyEmbedder] = None
        self._kg: Optional[KnowledgeGraph] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def init(self) -> bool:
        """Initialize or open the palace. Returns True if newly created."""
        self._base.mkdir(parents=True, exist_ok=True)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        is_new = not (self._base / "palace.json").exists()

        # SQLite metadata DB
        self._db = sqlite3.connect(str(self._base / "palace.db"), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS wings (
                name TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                name TEXT,
                wing TEXT,
                description TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (name, wing),
                FOREIGN KEY (wing) REFERENCES wings(name)
            )
        """)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS drawers (
                id TEXT PRIMARY KEY,
                wing TEXT NOT NULL,
                room TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                source_file TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (wing, room) REFERENCES rooms(wing, name)
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_drawers_wing_room
            ON drawers(wing, room)
        """)
        self._db.commit()

        # FAISS store
        self._store = FaissStore(str(self._data_dir))

        # Embedder
        self._embedder = NumpyEmbedder(model_dir=str(self._data_dir))

        # Knowledge graph
        self._kg = KnowledgeGraph(str(self._data_dir))

        # Save palace metadata
        if is_new:
            with open(self._base / "palace.json", "w") as f:
                json.dump({
                    "version": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "backend": "faiss",
                    "embedding": "numpy_tfidf_svd",
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
            try:
                self._embedder.save(str(self._data_dir))
            except Exception:
                pass

    @property
    def kg(self) -> KnowledgeGraph:
        if not self._kg:
            raise RuntimeError("Palace not initialized")
        return self._kg

    # ── Wing management ────────────────────────────────────────────────

    def list_wings(self) -> list[dict]:
        """List all wings."""
        rows = self._db.execute("""
            SELECT w.name, w.description, w.created_at,
                   COUNT(d.id) as drawer_count
            FROM wings w
            LEFT JOIN drawers d ON d.wing = w.name
            GROUP BY w.name
            ORDER BY w.name
        """).fetchall()
        return [
            {"name": r[0], "description": r[1], "created_at": r[2], "drawer_count": r[3]}
            for r in rows
        ]

    def get_or_create_wing(self, name: str, description: str = "") -> str:
        name = _sanitize(name, "wing")
        self._db.execute(
            "INSERT OR IGNORE INTO wings (name, description) VALUES (?, ?)",
            (name, description)
        )
        self._db.commit()
        return name

    # ── Room management ────────────────────────────────────────────────

    def list_rooms(self, wing: Optional[str] = None) -> list[dict]:
        if wing:
            rows = self._db.execute("""
                SELECT r.name, r.wing, r.description, r.created_at,
                       COUNT(d.id) as drawer_count
                FROM rooms r
                LEFT JOIN drawers d ON d.room = r.name AND d.wing = r.wing
                WHERE r.wing = ?
                GROUP BY r.name, r.wing
                ORDER BY r.name
            """, (wing,))
        else:
            rows = self._db.execute("""
                SELECT r.name, r.wing, r.description, r.created_at,
                       COUNT(d.id) as drawer_count
                FROM rooms r
                LEFT JOIN drawers d ON d.room = r.name AND d.wing = r.wing
                GROUP BY r.name, r.wing
                ORDER BY r.wing, r.name
            """)
        return [
            {"name": r[0], "wing": r[1], "description": r[2], "created_at": r[3], "drawer_count": r[4]}
            for r in rows.fetchall()
        ]

    def get_or_create_room(self, wing: str, name: str, description: str = "") -> str:
        wing = _sanitize(wing, "wing")
        name = _sanitize(name, "room")
        self.get_or_create_wing(wing)
        self._db.execute(
            "INSERT OR IGNORE INTO rooms (name, wing, description) VALUES (?, ?, ?)",
            (name, wing, description)
        )
        self._db.commit()
        return name

    # ── Drawer operations ──────────────────────────────────────────────

    def add_drawer(self, wing: str, room: str, content: str,
                   metadata: Optional[dict] = None,
                   source_file: str = "",
                   drawer_id: Optional[str] = None) -> str:
        """Add a memory drawer with vector embedding."""
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

        # Embed the content
        embedding = self._embedder.embed([content])[0]
        embedding_2d = embedding.reshape(1, -1)

        # Store in FAISS + SQLite
        self._store.add(
            ids=[drawer_id],
            texts=[content],
            metadatas=[meta],
            embeddings=embedding_2d
        )

        # Also store in drawers table
        self._db.execute(
            "INSERT OR REPLACE INTO drawers (id, wing, room, content, metadata, source_file) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (drawer_id, wing, room, content, json.dumps(meta), source_file)
        )
        self._db.commit()

        self._save_embedder()

        logger.debug("Added drawer %s in %s/%s", drawer_id, wing, room)
        return drawer_id

    def get_drawer(self, drawer_id: str) -> Optional[dict]:
        row = self._db.execute(
            "SELECT id, wing, room, content, metadata, source_file, created_at "
            "FROM drawers WHERE id = ?", (drawer_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "wing": row[1], "room": row[2],
            "content": row[3], "metadata": json.loads(row[4] or "{}"),
            "source_file": row[5], "created_at": row[6]
        }

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
        return [
            {
                "id": r[0], "wing": r[1], "room": r[2],
                "content": r[3][:500] + ("..." if len(r[3]) > 500 else ""),
                "metadata": json.loads(r[4] or "{}"),
                "source_file": r[5], "created_at": r[6]
            }
            for r in rows
        ]

    def delete_drawer(self, drawer_id: str) -> bool:
        row = self._db.execute("SELECT id FROM drawers WHERE id = ?", (drawer_id,)).fetchone()
        if not row:
            return False
        self._db.execute("DELETE FROM drawers WHERE id = ?", (drawer_id,))
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

        # Re-embed if content changed
        if content is not None:
            embedding = self._embedder.embed([new_content])[0]
            embedding_2d = embedding.reshape(1, -1)
            self._store.upsert(
                ids=[drawer_id],
                texts=[new_content],
                metadatas=[new_meta],
                embeddings=embedding_2d
            )

        new_wing = new_meta.get("wing", existing["wing"])
        new_room = new_meta.get("room", existing["room"])
        self._db.execute(
            "UPDATE drawers SET content=?, metadata=?, wing=?, room=? WHERE id=?",
            (new_content, json.dumps(new_meta), new_wing, new_room, drawer_id)
        )
        self._db.commit()

        self._save_embedder()

        return True

    # ── Search ─────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 10,
               wing: Optional[str] = None, room: Optional[str] = None) -> list[SearchResult]:
        """Semantic search across all drawers."""
        if not self._store or self._store.count() == 0:
            return []

        query_emb = self._embedder.embed([query])[0]
        ids, texts, distances, metadatas = self._store.search(
            query_emb, n_results=n_results * 2
        )

        results = []
        for i in range(len(ids)):
            meta = metadatas[i] if i < len(metadatas) else {}
            dw = meta.get("wing", "")
            dr = meta.get("room", "")
            if wing and dw != wing:
                continue
            if room and dr != room:
                continue
            results.append(SearchResult(
                id=ids[i],
                text=texts[i],
                distance=float(distances[i]) if i < len(distances) else 0.0,
                metadata=meta,
                wing=dw,
                room=dr,
            ))
            if len(results) >= n_results:
                break

        return results

    def status(self) -> dict:
        """Return palace status overview."""
        if not self._initialized:
            return {"initialized": False}
        wings = self.list_wings()
        total_drawers = self._db.execute("SELECT COUNT(*) FROM drawers").fetchone()[0]
        total_rooms = self._db.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
        return {
            "initialized": True,
            "path": str(self._base),
            "wings": len(wings),
            "rooms": total_rooms,
            "drawers": total_drawers,
            "wings_detail": wings,
            "embedding": "numpy_tfidf_svd_384",
        }

    def diary_write(self, agent_name: str, entry: str, topic: str = "general", wing: str = "") -> str:
        """Write an agent diary entry (stored as a drawer)."""
        target_wing = wing or f"agent_{_sanitize(agent_name)}"
        self.add_drawer(
            wing=target_wing,
            room="diary",
            content=entry,
            metadata={"agent": agent_name, "topic": topic, "type": "diary"}
        )
        return target_wing

    def diary_read(self, agent_name: str, last_n: int = 10, wing: str = "") -> list[dict]:
        """Read recent diary entries."""
        target_wing = wing or f"agent_{_sanitize(agent_name)}"
        return self.list_drawers(wing=target_wing, room="diary", limit=last_n)

    def check_duplicate(self, content: str, threshold: float = 0.9) -> Optional[dict]:
        """Check if similar content already exists."""
        if self._store.count() == 0:
            return None
        query_emb = self._embedder.embed([content])[0]
        ids, texts, distances, metadatas = self._store.search(query_emb, n_results=1)
        if ids and distances[0] >= threshold:
            return {
                "id": ids[0],
                "text": texts[0],
                "similarity": distances[0],
                "metadata": metadatas[0] if metadatas else {}
            }
        return None
