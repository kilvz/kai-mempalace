"""PalaceContext facade passed to source adapters (RFC 002 §9)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from kai_mempalace.sources.base import DrawerRecord


class _CollectionLike(Protocol):
    def add(self, **kwargs: Any) -> None: ...
    def upsert(self, **kwargs: Any) -> None: ...
    def query(self, **kwargs: Any) -> Any: ...
    def get(self, **kwargs: Any) -> Any: ...
    def delete(self, **kwargs: Any) -> None: ...
    def count(self) -> int: ...


class _KnowledgeGraphLike(Protocol):
    def add_triple(self, subject: str, predicate: str, obj: str, **kwargs: Any) -> Any: ...


ProgressHook = Callable[..., None]


@dataclass
class PalaceContext:
    drawer_collection: _CollectionLike
    knowledge_graph: _KnowledgeGraphLike
    palace_path: str
    closet_collection: Optional[_CollectionLike] = None
    config: Optional[Any] = None
    adapter_name: str = ""
    adapter_version: str = ""
    progress_hooks: list[ProgressHook] = field(default_factory=list)
    _skip_requested: bool = False

    def upsert_drawer(self, record: DrawerRecord) -> None:
        meta = dict(record.metadata)
        meta.setdefault("source_file", record.source_file)
        meta.setdefault("chunk_index", record.chunk_index)
        if self.adapter_name:
            meta.setdefault("adapter_name", self.adapter_name)
        if self.adapter_version:
            meta.setdefault("adapter_version", self.adapter_version)
        drawer_id = _build_drawer_id(record)
        self.drawer_collection.upsert(
            documents=[record.content],
            ids=[drawer_id],
            metadatas=[meta],
        )

    def skip_current_item(self) -> None:
        self._skip_requested = True

    def emit(self, event: str, **details: Any) -> None:
        for hook in self.progress_hooks:
            try:
                hook(event, **details)
            except Exception:
                import logging

                logging.getLogger(__name__).exception("progress hook failed on %r", event)


def _build_drawer_id(record: DrawerRecord) -> str:
    import hashlib

    digest = hashlib.sha256(record.source_file.encode("utf-8")).hexdigest()[:24]
    return f"{digest}_{record.chunk_index}"
