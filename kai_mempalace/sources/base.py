"""Source adapter contract for MemPalace (RFC 002)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Iterator, Literal, Optional

if TYPE_CHECKING:
    from kai_mempalace.sources.context import PalaceContext


class SourceAdapterError(Exception):
    """Base class for every source-adapter error raised by core."""


class SourceNotFoundError(SourceAdapterError):
    """Raised when a SourceRef does not resolve to a readable source."""


class AuthRequiredError(SourceAdapterError):
    """Raised when an adapter needs credentials that were not provided."""


class AdapterClosedError(SourceAdapterError):
    """Raised when an adapter method is called after close()."""


class TransformationViolationError(SourceAdapterError):
    """Raised by the conformance suite when round-tripping a drawer requires
    an undeclared transformation (RFC 002 §7.2–7.3)."""


class SchemaConformanceError(SourceAdapterError):
    """Raised when a DrawerRecord.metadata violates the adapter schema
    returned by BaseSourceAdapter.describe_schema()."""


@dataclass(frozen=True)
class SourceRef:
    local_path: Optional[str] = None
    uri: Optional[str] = None
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RouteHint:
    wing: Optional[str] = None
    room: Optional[str] = None
    hall: Optional[str] = None


@dataclass(frozen=True)
class SourceItemMetadata:
    source_file: str
    version: str
    size_hint: Optional[int] = None
    route_hint: Optional[RouteHint] = None


@dataclass(frozen=True)
class DrawerRecord:
    content: str
    source_file: str
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)
    route_hint: Optional[RouteHint] = None


@dataclass(frozen=True)
class SourceSummary:
    description: str
    item_count: Optional[int] = None


IngestMode = Literal["chunked_content", "whole_record", "metadata_only"]


@dataclass(frozen=True)
class FieldSpec:
    type: Literal["string", "int", "float", "bool", "delimiter_joined_string", "json_string"]
    required: bool
    description: str
    indexed: bool = False
    delimiter: str = ";"
    json_schema: Optional[dict] = None


@dataclass(frozen=True)
class AdapterSchema:
    fields: dict[str, FieldSpec]
    version: str


IngestResult = object


class BaseSourceAdapter(ABC):
    name: ClassVar[str]
    spec_version: ClassVar[str] = "1.0"
    adapter_version: ClassVar[str] = "0.0.0"
    capabilities: ClassVar[frozenset[str]] = frozenset()
    supported_modes: ClassVar[frozenset[str]] = frozenset({"chunked_content"})
    declared_transformations: ClassVar[frozenset[str]] = frozenset()
    default_privacy_class: ClassVar[str] = "pii_potential"

    @abstractmethod
    def ingest(self, *, source: SourceRef, palace: PalaceContext) -> Iterator[IngestResult]:
        ...

    @abstractmethod
    def describe_schema(self) -> AdapterSchema:
        ...

    def is_current(self, *, item: SourceItemMetadata, existing_metadata: Optional[dict]) -> bool:
        return False

    def source_summary(self, *, source: SourceRef) -> SourceSummary:
        return SourceSummary(description=self.name)

    def close(self) -> None:
        return None
