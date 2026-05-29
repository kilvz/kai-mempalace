"""Temporal knowledge graph - SQLite-backed entity-relationship store."""

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from kai_mempalace.config import sanitize_iso_temporal

logger = logging.getLogger(__name__)

DEFAULT_KG_PATH = None


def _is_date_only_temporal(value: str) -> bool:
    return isinstance(value, str) and len(value) == 10 and value[4] == "-" and value[7] == "-"


def _temporal_start_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if _is_date_only_temporal(value):
        return f"{value}T00:00:00Z"
    return value


def _temporal_end_key(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if _is_date_only_temporal(value):
        return f"{value}T23:59:59Z"
    return value


def _sql_temporal_start_expr(column: str) -> str:
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T00:00:00Z' ELSE {column} END"
    )


def _sql_temporal_end_expr(column: str) -> str:
    return (
        f"CASE WHEN length({column}) = 10 "
        f"AND substr({column}, 5, 1) = '-' "
        f"AND substr({column}, 8, 1) = '-' "
        f"THEN {column} || 'T23:59:59Z' ELSE {column} END"
    )


def _temporal_filter_sql(as_of: str) -> tuple[str, list[str]]:
    as_of_key = _temporal_start_key(as_of)
    valid_from_expr = _sql_temporal_start_expr("t.valid_from")
    valid_to_expr = _sql_temporal_end_expr("t.valid_to")
    return (
        f" AND (t.valid_from IS NULL OR {valid_from_expr} <= ?) "
        f"AND (t.valid_to IS NULL OR {valid_to_expr} >= ?)",
        [as_of_key, as_of_key],
    )


class KnowledgeGraph:
    """Lightweight temporal knowledge graph."""

    def __init__(self, path: str):
        self._db_path = Path(path)
        if self._db_path.suffix == ".sqlite3" or self._db_path.suffix == ".db":
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._db_path = self._db_path / "knowledge.db"
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = None
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("""CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
                valid_from TEXT, valid_to TEXT, source TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT DEFAULT 'unknown',
                properties TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        self._migrate_schema(conn)
        conn.commit()
        self._connection = conn

    def _migrate_schema(self, conn) -> None:
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}
        for col, dtype in [
            ("confidence", "REAL DEFAULT 1.0"),
            ("source_closet", "TEXT"),
            ("source_file", "TEXT"),
            ("source_drawer_id", "TEXT"),
            ("adapter_name", "TEXT"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {dtype}")

    def _conn(self):
        if self._connection is None:
            self._connection = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    @staticmethod
    def _entity_id(name: str) -> str:
        return name.lower().replace(" ", "_").replace("'", "")

    # ── Write operations ──────────────────────────────────────────────────

    def add(self, subject: str, predicate: str, object: str,
            valid_from: Optional[str] = None, valid_to: Optional[str] = None,
            source: str = "") -> int:
        """Add a fact. Backward-compatible — keeps existing signature."""
        with self._lock:
            cur = self._conn().execute(
                "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (subject, predicate, object, valid_from, valid_to, source))
            self._conn().commit()
            return cur.lastrowid

    def add_entity(self, name: str, entity_type: str = "unknown",
                   properties: dict = None) -> str:
        """Add or update an entity node."""
        eid = self._entity_id(name)
        props = json.dumps(properties or {})
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
                (eid, name, entity_type, props))
            conn.commit()
        return eid

    def add_triple(self, subject: str, predicate: str, object: str,
                   valid_from: Optional[str] = None,
                   valid_to: Optional[str] = None,
                   confidence: float = 1.0,
                   source_closet: Optional[str] = None,
                   source_file: Optional[str] = None,
                   source_drawer_id: Optional[str] = None,
                   adapter_name: Optional[str] = None) -> int:
        """Add a relationship triple with full provenance and temporal validation.

        Sanitizes temporal values, rejects inverted intervals, auto-creates
        entity entries, and skips duplicate active triples.

        Returns the integer row ID of the facts table entry.
        """
        valid_from = sanitize_iso_temporal(valid_from, "valid_from")
        valid_to = sanitize_iso_temporal(valid_to, "valid_to")
        if (valid_from is not None and valid_to is not None
                and _temporal_end_key(valid_to) < _temporal_start_key(valid_from)):
            raise ValueError(
                f"valid_to={valid_to!r} is before valid_from={valid_from!r}; "
                "an inverted interval would be invisible to every KG query")
        sub_id = self._entity_id(subject)
        obj_id = self._entity_id(object)
        pred = predicate.lower().replace(" ", "_")
        with self._lock:
            conn = self._conn()
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                (sub_id, subject))
            conn.execute(
                "INSERT OR IGNORE INTO entities (id, name) VALUES (?, ?)",
                (obj_id, object))
            existing = conn.execute(
                "SELECT id FROM facts WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (subject, pred, object)).fetchone()
            if existing:
                return existing["id"]
            conn.execute(
                "INSERT INTO facts (subject, predicate, object, valid_from, valid_to, "
                "confidence, source_closet, source_file, source_drawer_id, adapter_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (subject, pred, object, valid_from, valid_to,
                 confidence, source_closet, source_file, source_drawer_id, adapter_name))
            conn.commit()
            return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def invalidate(self, subject: str, predicate: str, object: str,
                   ended: Optional[str] = None) -> int:
        """Mark a fact as no longer valid.

        Returns number of rows updated.
        """
        if ended is None:
            ended = date.today().isoformat()
        ended = sanitize_iso_temporal(ended, "ended")
        with self._lock:
            conn = self._conn()
            rows = conn.execute(
                "SELECT id, valid_from FROM facts "
                "WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (subject, predicate, object)).fetchall()
            for row in rows:
                vf = row["valid_from"]
                if vf is not None and _temporal_end_key(ended) < _temporal_start_key(vf):
                    raise ValueError(
                        f"valid_to={ended!r} is before valid_from={vf!r}; "
                        "an inverted interval would be invisible to every KG query")
            cur = conn.execute(
                "UPDATE facts SET valid_to=? "
                "WHERE subject=? AND predicate=? AND object=? AND valid_to IS NULL",
                (ended, subject, predicate, object))
            conn.commit()
            return cur.rowcount

    # ── Query operations ──────────────────────────────────────────────────

    def query(self, entity: Optional[str] = None, predicate: Optional[str] = None,
              as_of: Optional[str] = None, direction: str = "both") -> list[dict]:
        """Query facts. Backward-compatible signature and return format."""
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
            rows = self._conn().execute(sql, params).fetchall()
        return [{"id": r[0], "subject": r[1], "predicate": r[2], "object": r[3],
                 "valid_from": r[4], "valid_to": r[5], "source": r[6], "created_at": r[7]}
                for r in rows]

    def query_entity(self, name: str, as_of: Optional[str] = None,
                     direction: str = "outgoing") -> list[dict]:
        """Get all relationships for an entity with temporal filtering.

        direction: 'outgoing' (entity -> ?), 'incoming' (? -> entity), 'both'
        """
        as_of = sanitize_iso_temporal(as_of, "as_of")
        results = []
        temporal_sql = ""
        temporal_params = []
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(as_of)
        with self._lock:
            conn = self._conn()
            if direction in ("outgoing", "both"):
                rows = conn.execute(
                    "SELECT * FROM facts AS t WHERE t.subject = ?" + temporal_sql,
                    [name] + temporal_params).fetchall()
                for r in rows:
                    results.append({
                        "direction": "outgoing",
                        "subject": name,
                        "predicate": r["predicate"],
                        "object": r["object"],
                        "valid_from": r["valid_from"],
                        "valid_to": r["valid_to"],
                        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                        "source_closet": r["source_closet"],
                        "current": r["valid_to"] is None,
                    })
            if direction in ("incoming", "both"):
                rows = conn.execute(
                    "SELECT * FROM facts AS t WHERE t.object = ?" + temporal_sql,
                    [name] + temporal_params).fetchall()
                for r in rows:
                    results.append({
                        "direction": "incoming",
                        "subject": r["subject"],
                        "predicate": r["predicate"],
                        "object": name,
                        "valid_from": r["valid_from"],
                        "valid_to": r["valid_to"],
                        "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                        "source_closet": r["source_closet"],
                        "current": r["valid_to"] is None,
                    })
        return results

    def query_relationship(self, predicate: str,
                           as_of: Optional[str] = None) -> list[dict]:
        """Get all triples with a given relationship type."""
        as_of = sanitize_iso_temporal(as_of, "as_of")
        pred = predicate.lower().replace(" ", "_")
        query = "SELECT * FROM facts AS t WHERE t.predicate = ?"
        params = [pred]
        if as_of:
            temporal_sql, temporal_params = _temporal_filter_sql(as_of)
            query += temporal_sql
            params.extend(temporal_params)
        results = []
        with self._lock:
            for r in self._conn().execute(query, params).fetchall():
                results.append({
                    "subject": r["subject"],
                    "predicate": pred,
                    "object": r["object"],
                    "valid_from": r["valid_from"],
                    "valid_to": r["valid_to"],
                    "current": r["valid_to"] is None,
                })
        return results

    def timeline(self, entity: Optional[str] = None) -> list[dict]:
        """Get all facts in chronological order (valid_from ASC NULLS LAST)."""
        with self._lock:
            conn = self._conn()
            if entity:
                rows = conn.execute(
                    "SELECT * FROM facts AS t "
                    "WHERE (t.subject = ? OR t.object = ?) "
                    "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100",
                    (entity, entity)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM facts AS t "
                    "ORDER BY t.valid_from ASC NULLS LAST LIMIT 100").fetchall()
        return [
            {
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "valid_from": r["valid_from"],
                "valid_to": r["valid_to"],
                "current": r["valid_to"] is None,
            }
            for r in rows
        ]

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            conn = self._conn()
            total = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
            subjects = conn.execute(
                "SELECT COUNT(DISTINCT subject) FROM facts").fetchone()[0]
            predicates_list = [
                r[0] for r in conn.execute(
                    "SELECT DISTINCT predicate FROM facts ORDER BY predicate").fetchall()
            ]
            entities = conn.execute(
                "SELECT COUNT(*) FROM entities").fetchone()[0]
            current = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE valid_to IS NULL").fetchone()[0]
            expired = total - current
        return {
            "total_facts": total,
            "unique_subjects": subjects,
            "unique_predicates": len(predicates_list),
            "entities": entities,
            "current_facts": current,
            "expired_facts": expired,
            "relationship_types": predicates_list,
        }

    # ── Seed from known facts ─────────────────────────────────────────────

    def seed_from_entity_facts(self, entity_facts: dict) -> None:
        """Seed the knowledge graph from fact_checker.py ENTITY_FACTS."""
        for key, facts in entity_facts.items():
            name = facts.get("full_name", key.capitalize())
            etype = facts.get("type", "person")
            self.add_entity(name, etype, {
                "gender": facts.get("gender", ""),
                "birthday": facts.get("birthday", ""),
            })
            parent = facts.get("parent")
            if parent:
                self.add_triple(
                    name, "child_of", parent.capitalize(),
                    valid_from=facts.get("birthday"))
            partner = facts.get("partner")
            if partner:
                self.add_triple(name, "married_to", partner.capitalize())
            relationship = facts.get("relationship", "")
            if relationship == "daughter":
                self.add_triple(
                    name, "is_child_of",
                    facts.get("parent", "").capitalize() or name,
                    valid_from=facts.get("birthday"))
            elif relationship == "husband":
                self.add_triple(
                    name, "is_partner_of",
                    facts.get("partner", name).capitalize())
            elif relationship == "brother":
                self.add_triple(
                    name, "is_sibling_of",
                    facts.get("sibling", name).capitalize())
            elif relationship == "dog":
                self.add_triple(
                    name, "is_pet_of",
                    facts.get("owner", name).capitalize())
                self.add_entity(name, "animal")
            for interest in facts.get("interests", []):
                self.add_triple(name, "loves", interest.capitalize(),
                                valid_from="2025-01-01")

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
