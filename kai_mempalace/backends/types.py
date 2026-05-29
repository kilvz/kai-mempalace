"""Portable typed result dataclasses and error hierarchy for storage backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


# ── Error hierarchy ───────────────────────────────────────────────────────


class BackendError(Exception):
    """Base class for every storage-backend error."""


class PalaceNotFoundError(BackendError, FileNotFoundError):
    """Raised when a palace directory or database is missing."""


class CollectionNotInitializedError(PalaceNotFoundError):
    """Raised when the palace exists but the collection has never been created."""


class DimensionMismatchError(BackendError):
    """Raised when embedding dimension on write does not match the collection."""


class BackendClosedError(BackendError):
    """Raised when a backend method is called after close()."""


class EmbedderIdentityMismatchError(BackendError):
    """Raised when the stored embedder model name differs from the current one."""


class UnsupportedFilterError(BackendError):
    """Raised when a where-clause operator is not implemented."""


# ── Value objects ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HealthStatus:
    ok: bool
    detail: str = ""

    @classmethod
    def healthy(cls, detail: str = "") -> "HealthStatus":
        return cls(ok=True, detail=detail)

    @classmethod
    def unhealthy(cls, detail: str) -> "HealthStatus":
        return cls(ok=False, detail=detail)


# ── Typed result classes ──────────────────────────────────────────────────


_TYPED_RESULT_FIELDS = ("ids", "documents", "metadatas", "distances", "embeddings")


class _DictCompatMixin:
    """Transitional dict-protocol access for typed results.

    Primary access is attribute access (``result.ids``). The
    ``result["ids"]`` and ``result.get("ids")`` shims support legacy
    callers that haven't been migrated yet.
    """

    def __getitem__(self, key: str) -> Any:
        if key in _TYPED_RESULT_FIELDS:
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key in _TYPED_RESULT_FIELDS:
            val = getattr(self, key, default)
            return default if val is None else val
        return default

    def __contains__(self, key: object) -> bool:
        return key in _TYPED_RESULT_FIELDS and getattr(self, key, None) is not None


@dataclass(frozen=True)
class QueryResult(_DictCompatMixin):
    """Typed return from a vector store ``query`` call.

    Outer list dimension = number of query vectors / texts.
    Inner list dimension = hits per query (may be zero).

    Fields not requested via ``include=`` are populated with empty lists
    of the correct outer shape (never ``None``), except ``embeddings``
    which is ``None`` when not requested.
    """

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict]]
    distances: list[list[float]]
    embeddings: Optional[list[list[list[float]]]] = None

    @classmethod
    def empty(
        cls, num_queries: int = 1, embeddings_requested: bool = False
    ) -> QueryResult:
        """Construct an all-empty result preserving outer dimension."""
        empty_outer = [[] for _ in range(num_queries)]
        return cls(
            ids=[[] for _ in range(num_queries)],
            documents=[[] for _ in range(num_queries)],
            metadatas=[[] for _ in range(num_queries)],
            distances=[[] for _ in range(num_queries)],
            embeddings=empty_outer if embeddings_requested else None,
        )


@dataclass(frozen=True)
class GetResult(_DictCompatMixin):
    """Typed return from a vector store ``get`` call."""

    ids: list[str]
    documents: list[str]
    metadatas: list[dict]
    embeddings: Optional[list[list[float]]] = None

    @classmethod
    def empty(cls) -> GetResult:
        return cls(ids=[], documents=[], metadatas=[], embeddings=None)
