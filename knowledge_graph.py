"""
Temporal knowledge graph — SQLite-backed entity-relationship store.

Mirrors MemPalace's knowledge graph schema but uses bare SQLite
instead of the full mempalace pipeline.
"""

import json
import logging
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Lightweight temporal knowledge graph."""

    def __init__(self, path: str):
        self._db_path = Path(path) / "knowledge.db"
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_from TEXT,
                valid_to TEXT,
                source TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_subject
            ON facts(subject)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_predicate
            ON facts(predicate)
        """)
        self._db.execute("""
            CREATE INDEX IF NOT EXISTS idx_facts_object
            ON facts(object)
        """)
        self._db.commit()

    def add(self, subject: str, predicate: str, object: str,
            valid_from: Optional[str] = None, valid_to: Optional[str] = None,
            source: str = "") -> int:
        """Add a fact. Returns the fact ID."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject, predicate, object, valid_from, valid_to, source)
            )
            self._db.commit()
            return cur.lastrowid

    def invalidate(self, subject: str, predicate: str, object: str,
                   ended: Optional[str] = None) -> int:
        """Mark a fact as no longer true by setting its valid_to."""
        if ended is None:
            ended = date.today().isoformat()
        with self._lock:
            cur = self._db.execute(
                "UPDATE facts SET valid_to = ? "
                "WHERE subject = ? AND predicate = ? AND object = ? AND valid_to IS NULL",
                (ended, subject, predicate, object)
            )
            self._db.commit()
            return cur.rowcount

    def query(self, entity: Optional[str] = None, predicate: Optional[str] = None,
              as_of: Optional[str] = None, direction: str = "both") -> list[dict]:
        """Query facts. Returns list of dicts."""
        conditions = []
        params = []

        if entity:
            if direction in ("outgoing", "both"):
                conditions.append("(subject = ?)")
                params.append(entity)
            if direction in ("incoming", "both"):
                conditions.append("(object = ?)")
                params.append(entity)
            if len(conditions) == 2:
                conditions = [f"({' OR '.join(conditions)})"]

        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)

        if as_of:
            conditions.append("(valid_from IS NULL OR valid_from <= ?)")
            params.append(as_of)
            conditions.append("(valid_to IS NULL OR valid_to > ?)")
            params.append(as_of)

        sql = "SELECT * FROM facts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"

        with self._lock:
            rows = self._db.execute(sql, params).fetchall()

        return [
            {
                "id": r[0], "subject": r[1], "predicate": r[2],
                "object": r[3], "valid_from": r[4], "valid_to": r[5],
                "source": r[6], "created_at": r[7]
            }
            for r in rows
        ]

    def timeline(self, entity: Optional[str] = None) -> list[dict]:
        """Get chronological story of an entity."""
        return self.query(entity=entity, as_of=None)

    def stats(self) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            subjects = self._db.execute(
                "SELECT COUNT(DISTINCT subject) FROM facts"
            ).fetchone()[0]
            predicates = self._db.execute(
                "SELECT COUNT(DISTINCT predicate) FROM facts"
            ).fetchone()[0]
        return {"total_facts": total, "unique_subjects": subjects, "unique_predicates": predicates}

    def close(self) -> None:
        if hasattr(self, '_db'):
            self._db.close()
