"""Backend registry + entry-point discovery (RFC 001 §3).

Third-party backends ship as installable packages that declare a
``kai_mempalace.backends`` entry point. Explicit registration wins on
name conflict. Built-in ``faiss`` backend is registered by default.
"""

from __future__ import annotations

import logging
from importlib import metadata
from threading import Lock
from typing import Optional, Type

from kai_mempalace.backends.base import BaseBackend

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "kai_mempalace.backends"

_registry: dict[str, Type[BaseBackend]] = {}
_instances: dict[str, BaseBackend] = {}
_explicit: set[str] = set()
_discovered = False
_lock = Lock()


def register(name: str, backend_cls: Type[BaseBackend]) -> None:
    """Register ``backend_cls`` under ``name``.

    Explicit registration wins over entry-point discovery on conflict.
    """
    with _lock:
        _registry[name] = backend_cls
        _explicit.add(name)
        _instances.pop(name, None)


def unregister(name: str) -> None:
    """Remove a backend registration (primarily for tests)."""
    with _lock:
        _registry.pop(name, None)
        _explicit.discard(name)
        _instances.pop(name, None)


def _discover_entry_points() -> None:
    """Load entry-point-declared backends once per process."""
    global _discovered
    if _discovered:
        return
    with _lock:
        if _discovered:
            return
        try:
            eps = metadata.entry_points()
            group = (
                eps.select(group=_ENTRY_POINT_GROUP)
                if hasattr(eps, "select")
                else eps.get(_ENTRY_POINT_GROUP, [])
            )
        except Exception:
            logger.exception("entry-point discovery for %s failed", _ENTRY_POINT_GROUP)
            group = []
        for ep in group:
            if ep.name in _explicit:
                continue
            try:
                cls = ep.load()
            except Exception:
                logger.exception("failed to load backend entry point %r", ep.name)
                continue
            if not isinstance(cls, type) or not issubclass(cls, BaseBackend):
                logger.warning(
                    "entry point %r did not resolve to a BaseBackend subclass (got %r)",
                    ep.name,
                    cls,
                )
                continue
            _registry.setdefault(ep.name, cls)
        _discovered = True


def available_backends() -> list[str]:
    """Return sorted list of all registered backend names."""
    _discover_entry_points()
    return sorted(_registry.keys())


def get_backend_class(name: str) -> Type[BaseBackend]:
    """Return the registered backend class for ``name``."""
    _discover_entry_points()
    try:
        return _registry[name]
    except KeyError as e:
        raise KeyError(f"unknown backend {name!r}; available: {available_backends()}") from e


def get_backend(name: str) -> BaseBackend:
    """Return a long-lived cached instance of the named backend."""
    _discover_entry_points()
    with _lock:
        inst = _instances.get(name)
        if inst is not None:
            return inst
        cls = _registry.get(name)
        if cls is None:
            raise KeyError(f"unknown backend {name!r}; available: {sorted(_registry.keys())}")
        inst = cls()
        _instances[name] = inst
        return inst


def reset_backends() -> None:
    """Close and drop all cached backend instances (primarily for tests)."""
    with _lock:
        for inst in _instances.values():
            try:
                inst.close()
            except Exception:
                logger.exception("error closing backend during reset")
        _instances.clear()


def resolve_backend_for_palace(
    *,
    explicit: Optional[str] = None,
    config_value: Optional[str] = None,
    env_value: Optional[str] = None,
    palace_path: Optional[str] = None,
    default: str = "faiss",
) -> str:
    """Resolve the backend name per RFC 001 §3.3 priority order.

    1. Explicit kwarg
    2. Config value
    3. ``KAI_MEMPALACE_BACKEND`` env var
    4. Auto-detect from on-disk artifacts
    5. Default (``faiss``)
    """
    for candidate in (explicit, config_value, env_value):
        if candidate:
            return candidate
    _discover_entry_points()
    if palace_path:
        for name, cls in _registry.items():
            try:
                if cls.detect(palace_path):
                    return name
            except Exception:
                logger.exception("detect() raised on backend %r", name)
                continue
    return default


def _register_builtins() -> None:
    """Register faiss as the in-tree default backend."""
    from kai_mempalace.backends.faiss_backend import FaissBackend

    if "faiss" not in _registry:
        _registry["faiss"] = FaissBackend


_register_builtins()
