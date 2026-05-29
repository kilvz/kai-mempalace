"""FaissBackend + FaissCollection — RFC 001 backend contract for FAISS.

Wraps :class:`FaissStore` in the :class:`BaseCollection` / :class:`BaseBackend`
ABCs so the registry, Palace, and repair code can consume the backend
through a uniform interface.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from kai_mempalace.backends.base import (
    BaseBackend,
    BaseCollection,
    GetResult,
    HealthStatus,
    PalaceRef,
    QueryResult,
    _IncludeSpec,
)
from kai_mempalace.backends.faiss_store import FaissStore

logger = logging.getLogger(__name__)


class FaissCollection(BaseCollection):
    """Adapter wrapping :class:`FaissStore` in the :class:`BaseCollection` ABC."""

    def __init__(self, store: FaissStore):
        self._store = store

    def add(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("FaissCollection.add requires embeddings")
        emb_array = np.asarray(embeddings, dtype=np.float32)
        self._store.add(
            ids=ids,
            texts=documents,
            metadatas=metadatas or [{}] * len(ids),
            embeddings=emb_array,
        )

    def upsert(
        self,
        *,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("FaissCollection.upsert requires embeddings")
        emb_array = np.asarray(embeddings, dtype=np.float32)
        self._store.upsert(
            ids=ids,
            texts=documents,
            metadatas=metadatas or [{}] * len(ids),
            embeddings=emb_array,
        )

    def query(
        self,
        *,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: Optional[list[str]] = None,
    ) -> QueryResult:
        spec = _IncludeSpec.resolve(include, default_distances=True)

        if query_embeddings is not None:
            qe = np.asarray(query_embeddings, dtype=np.float32)
        elif query_texts is not None:
            raise ValueError(
                "FaissCollection.query requires query_embeddings; "
                "text-to-embed conversion is the caller's responsibility"
            )
        else:
            raise ValueError("query requires query_texts or query_embeddings")

        all_ids: list[list[str]] = []
        all_docs: list[list[str]] = []
        all_dists: list[list[float]] = []
        all_metas: list[list[dict]] = []

        for q in range(qe.shape[0]):
            qv = qe[q : q + 1]
            ids, texts, dists, metadatas = self._store.search(
                qv, n_results=n_results, where=where
            )
            all_ids.append(ids)
            all_docs.append(texts if spec.documents else [])
            all_dists.append(dists if spec.distances else [])
            all_metas.append(metadatas if spec.metadatas else [])

        return QueryResult(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            distances=all_dists,
            embeddings=None,
        )

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        spec = _IncludeSpec.resolve(include, default_distances=False)
        g_ids, g_docs, g_metas = self._store.get(
            ids=ids, where=where, where_document=where_document,
            limit=limit, offset=offset or 0,
        )
        return GetResult(
            ids=g_ids,
            documents=g_docs if spec.documents else [],
            metadatas=g_metas if spec.metadatas else [],
            embeddings=None,
        )

    def delete(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
    ) -> None:
        self._store.delete(ids=ids, where=where)

    def count(self) -> int:
        return self._store.count()

    def close(self) -> None:
        self._store.close()

    def health(self) -> HealthStatus:
        try:
            n = self._store.count()
            return HealthStatus.healthy(detail=f"FAISS index healthy, {n} vectors")
        except Exception as e:
            return HealthStatus.unhealthy(detail=str(e))


class FaissBackend(BaseBackend):
    """RFC 001 backend wrapping :class:`FaissStore`.

    One ``FaissStore`` per palace; cached by ``data_dir``.
    """

    name: ClassVar[str] = "faiss"
    spec_version: ClassVar[str] = "1.0"
    capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self):
        self._stores: dict[str, FaissStore] = {}
        self._closed = False

    def get_collection(
        self,
        *,
        palace: PalaceRef,
        collection_name: str,
        create: bool = False,
        options: Optional[dict] = None,
    ) -> FaissCollection:
        if self._closed:
            raise RuntimeError("FaissBackend is closed")

        data_dir = palace.local_path
        if data_dir is None:
            data_dir = str(Path(palace.id).expanduser().resolve() / "data")

        store = self._stores.get(data_dir)
        if store is None:
            if not os.path.isdir(data_dir):
                if not create:
                    from kai_mempalace.backends.types import PalaceNotFoundError

                    raise PalaceNotFoundError(f"Palace not found: {data_dir}")
                os.makedirs(data_dir, exist_ok=True)
            store = FaissStore(data_dir)
            self._stores[data_dir] = store

        return FaissCollection(store)

    def close_palace(self, palace: PalaceRef) -> None:
        data_dir = palace.local_path
        if data_dir is None:
            data_dir = str(Path(palace.id).expanduser().resolve() / "data")
        store = self._stores.pop(data_dir, None)
        if store:
            store.close()

    def close(self) -> None:
        for store in self._stores.values():
            try:
                store.close()
            except Exception:
                logger.exception("error closing FaissStore")
        self._stores.clear()
        self._closed = True

    def health(self, palace: Optional[PalaceRef] = None) -> HealthStatus:
        try:
            if palace:
                data_dir = palace.local_path or str(
                    Path(palace.id).expanduser().resolve() / "data"
                )
                store = FaissStore(data_dir)
                n = store.count()
                store.close()
                return HealthStatus.healthy(detail=f"FAISS {n} vectors")
            return HealthStatus.healthy(detail="FAISS backend ready")
        except Exception as e:
            return HealthStatus.unhealthy(detail=str(e))

    @classmethod
    def detect(cls, path: str) -> bool:
        """Detect a FAISS-palace by the presence of ``data/index.faiss``."""
        return (Path(path) / "data" / "index.faiss").exists()
