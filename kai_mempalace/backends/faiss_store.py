"""FAISS vector store with SQLite metadata + vector persistence."""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 384


class FaissStore:
    """Persistent vector store backed by FAISS + SQLite."""

    def __init__(self, path: str, dimension: int = _DEFAULT_DIM):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self._lock = threading.Lock()

        self._db_path = self.path / "metadata.db"
        self._init_db()

        index_path = self.path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        else:
            self.index = faiss.IndexFlatIP(dimension)
            logger.info("Created new FAISS flat index (dim=%d)", dimension)

        self._seq_path = self.path / "seq.txt"
        self._seq = self._load_seq()

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict],
            embeddings: np.ndarray) -> None:
        if len(ids) == 0:
            return
        n = len(ids)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)
        with self._lock:
            start = self.index.ntotal
            self.index.add(embeddings)
            self._write_docs(ids, texts, metadatas, embeddings, start)
            self._save()

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict],
               embeddings: np.ndarray) -> None:
        existing = set(self._get_existing_ids(ids))
        if existing:
            self.delete(ids=list(existing))
        self.add(ids, texts, metadatas, embeddings)

    def search(self, query_emb: np.ndarray, n_results: int = 10,
               where: Optional[dict] = None) -> tuple[list[str], list[str], list[float], list[dict]]:
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_emb)
        k = min(n_results, self.index.ntotal)
        if k == 0:
            return [], [], [], []
        distances, indices = self.index.search(query_emb, k)
        idxs = indices[0].tolist()
        dists = distances[0].tolist()
        with self._lock:
            rows = self._get_rows(idxs)
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        metadatas = [json.loads(r[2]) if r[2] else {} for r in rows]
        if where:
            filtered = [(ids[i], texts[i], dists[i], metadatas[i])
                        for i in range(len(ids))
                        if self._matches_where(metadatas[i], where)]
            if filtered:
                ids, texts, dists, metadatas = zip(*filtered)
            else:
                return [], [], [], []
        return list(ids), list(texts), list(dists), list(metadatas)

    def get(self, ids: Optional[list[str]] = None, where: Optional[dict] = None,
            where_document: Optional[dict] = None, limit: Optional[int] = None,
            offset: Optional[int] = 0) -> tuple[list[str], list[str], list[dict]]:
        with self._lock:
            if ids:
                placeholders = ",".join("?" * len(ids))
                cur = self._db.execute(
                    f"SELECT id, text, metadata FROM docs WHERE id IN ({placeholders})",
                    ids
                )
            else:
                conditions = []
                params = []
                if where:
                    cond, p = self._where_to_sql(where)
                    if cond != "1=1":
                        conditions.append(cond)
                        params.extend(p)
                if where_document:
                    cond, p = self._where_doc_to_sql(where_document)
                    if cond != "1=1":
                        conditions.append(cond)
                        params.extend(p)
                sql = "SELECT id, text, metadata FROM docs"
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY rowid"
                if limit is not None:
                    sql += f" LIMIT {limit}"
                if offset:
                    sql += f" OFFSET {offset}"
                cur = self._db.execute(sql, params)
            rows = cur.fetchall()
        return [r[0] for r in rows], [r[1] for r in rows], \
               [json.loads(r[2]) if r[2] else {} for r in rows]

    def delete(self, ids: Optional[list[str]] = None, where: Optional[dict] = None) -> None:
        with self._lock:
            if ids:
                placeholders = ",".join("?" * len(ids))
                self._db.execute(f"DELETE FROM docs WHERE id IN ({placeholders})", ids)
                self._db.execute(f"DELETE FROM pos_map WHERE doc_id IN ({placeholders})", ids)
            elif where:
                cond, params = self._where_to_sql(where)
                self._db.execute(f"DELETE FROM docs WHERE {cond}", params)
            else:
                return
            self._db.commit()
            self._rebuild_index()

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]

    def close(self) -> None:
        if hasattr(self, '_db'):
            self._db.close()

    def _init_db(self) -> None:
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("""CREATE TABLE IF NOT EXISTS docs (
                id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT DEFAULT '{}',
                vector BLOB, rowid INTEGER)""")
        self._db.execute("""CREATE TABLE IF NOT EXISTS pos_map (
                faiss_pos INTEGER PRIMARY KEY, doc_id TEXT NOT NULL)""")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_pos_doc ON pos_map(doc_id)")
        self._db.commit()

    def _write_docs(self, ids: list[str], texts: list[str],
                    metadatas: list[dict], embeddings: np.ndarray, start_pos: int) -> None:
        for i, (doc_id, text, meta) in enumerate(zip(ids, texts, metadatas)):
            faiss_pos = start_pos + i
            vec_blob = embeddings[i].tobytes()
            self._db.execute(
                "INSERT OR REPLACE INTO docs (id, text, metadata, vector, rowid) VALUES (?, ?, ?, ?, ?)",
                (doc_id, text, json.dumps(meta), vec_blob, faiss_pos))
            self._db.execute(
                "INSERT OR REPLACE INTO pos_map (faiss_pos, doc_id) VALUES (?, ?)",
                (faiss_pos, doc_id))
        self._db.commit()

    def _get_rows(self, faiss_indices: list[int]) -> list[tuple]:
        if not faiss_indices:
            return []
        placeholders = ",".join("?" * len(faiss_indices))
        order_clause = " ".join(f"WHEN {pos} THEN {i}" for i, pos in enumerate(faiss_indices))
        cur = self._db.execute(
            f"SELECT d.id, d.text, d.metadata FROM docs d "
            f"JOIN pos_map p ON d.id = p.doc_id "
            f"WHERE p.faiss_pos IN ({placeholders}) "
            f"ORDER BY CASE p.faiss_pos {order_clause} END",
            faiss_indices)
        return cur.fetchall()

    def _get_existing_ids(self, ids: list[str]) -> list[str]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        cur = self._db.execute(f"SELECT id FROM docs WHERE id IN ({placeholders})", ids)
        return [r[0] for r in cur.fetchall()]

    def _rebuild_index(self) -> None:
        cur = self._db.execute(
            "SELECT id, vector, rowid FROM docs WHERE vector IS NOT NULL ORDER BY rowid")
        rows = cur.fetchall()
        self.index = faiss.IndexFlatIP(self.dimension)
        self._db.execute("DELETE FROM pos_map")
        if not rows:
            self._db.commit()
            logger.info("FAISS index rebuilt (empty)")
            return
        vectors = []
        doc_ids = []
        for doc_id, vec_blob, _ in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            if len(vec) != self.dimension:
                logger.warning("Skipping doc %s: wrong vector dim %d", doc_id, len(vec))
                continue
            vectors.append(vec)
            doc_ids.append(doc_id)
        if not vectors:
            self._db.commit()
            self._save()
            return
        all_vecs = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(all_vecs)
        self.index.add(all_vecs)
        for i, doc_id in enumerate(doc_ids):
            self._db.execute(
                "INSERT OR REPLACE INTO pos_map (faiss_pos, doc_id) VALUES (?, ?)", (i, doc_id))
            self._db.execute("UPDATE docs SET rowid = ? WHERE id = ?", (i, doc_id))
        self._db.commit()
        logger.info("FAISS index rebuilt with %d vectors", len(doc_ids))

    def _matches_where(self, metadata: dict, where: dict) -> bool:
        for key, condition in where.items():
            if key == "$and":
                if not all(self._matches_where(metadata, c) for c in condition):
                    return False
            elif key == "$or":
                if not any(self._matches_where(metadata, c) for c in condition):
                    return False
            else:
                val = metadata.get(key)
                if isinstance(condition, dict):
                    for op, op_val in condition.items():
                        if op == "$eq" and val != op_val:
                            return False
                        elif op == "$ne" and val == op_val:
                            return False
                        elif op == "$in" and val not in op_val:
                            return False
                        elif op == "$nin" and val in op_val:
                            return False
                else:
                    if val != condition:
                        return False
        return True

    def _where_to_sql(self, where: dict, prefix: str = "") -> tuple[str, list]:
        conditions = []
        params = []
        meta_col = f"{prefix}metadata" if prefix else "metadata"
        for key, condition in where.items():
            if key == "$and":
                sub_conds = []
                for c in condition:
                    sub, p = self._where_to_sql(c, prefix)
                    sub_conds.append(f"({sub})")
                    params.extend(p)
                conditions.append("(" + " AND ".join(sub_conds) + ")")
            elif key == "$or":
                sub_conds = []
                for c in condition:
                    sub, p = self._where_to_sql(c, prefix)
                    sub_conds.append(f"({sub})")
                    params.extend(p)
                conditions.append("(" + " OR ".join(sub_conds) + ")")
            else:
                if isinstance(condition, dict):
                    for op, op_val in condition.items():
                        if op in ("$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte"):
                            json_path = f"$.{key}"
                            if op == "$eq":
                                conditions.append(f"json_extract({meta_col}, ?) = ?")
                                params.extend([json_path, str(op_val)])
                            elif op == "$ne":
                                conditions.append(f"json_extract({meta_col}, ?) != ?")
                                params.extend([json_path, str(op_val)])
                            elif op == "$in":
                                ph = ",".join("?" * len(op_val))
                                conditions.append(f"json_extract({meta_col}, ?) IN ({ph})")
                                params.extend([json_path] + list(op_val))
                            elif op == "$nin":
                                ph = ",".join("?" * len(op_val))
                                conditions.append(f"json_extract({meta_col}, ?) NOT IN ({ph})")
                                params.extend([json_path] + list(op_val))
                else:
                    conditions.append(f"json_extract({meta_col}, ?) = ?")
                    params.extend([f"$.{key}", str(condition)])
        return " AND ".join(conditions) if conditions else "1=1", params

    def _where_doc_to_sql(self, where_doc: dict) -> tuple[str, list]:
        conditions = []
        params = []
        for key, condition in where_doc.items():
            if key == "$contains":
                conditions.append("text LIKE ?")
                params.append(f"%{condition}%")
        return " AND ".join(conditions) if conditions else "1=1", params

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.path / "index.faiss"))
        self._db.commit()

    def _load_seq(self) -> int:
        if self._seq_path.exists():
            return int(self._seq_path.read_text().strip())
        try:
            cur = self._db.execute(
                "SELECT id FROM docs WHERE id GLOB 'doc_*' ORDER BY CAST(SUBSTR(id, 5) AS INTEGER) DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                parts = row[0].split('_')
                if len(parts) >= 2 and parts[-1].isdigit():
                    return int(parts[-1])
        except Exception:
            pass
        return 0

    def _save_seq(self) -> None:
        self._seq_path.write_text(str(self._seq))

    def next_id(self) -> str:
        self._seq += 1
        self._save_seq()
        return f"doc_{self._seq}"
