"""dynamics.py — Living-connection math for halls + tunnels.

Hebbian potentiation (strength grows on co-access) and Ebbinghaus exponential
decay (strength fades with time since last activation), with the Cepeda
spacing effect: stability grows when reinforcement is spaced rather than
massed.

This module is pure. No I/O, no DB, no chromadb. It operates on plain
dicts (hall records, tunnel records) and mutates them in place. Callers
in ``hallways.py`` and ``palace_graph.py`` invoke these functions; the
math lives here in one place so both connection kinds share identical
semantics.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

STRENGTH_FLOOR = 0.05
MAX_STRENGTH = 5.0
DEFAULT_STABILITY = 1.0
DEFAULT_STRENGTH = 1.0
POTENTIATION_INCREMENT = 0.05
SPACED_INTERVAL_HOURS = 1.0
STABILITY_INCREMENT = 0.1


def _ensure_tz(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return datetime.now(timezone.utc)
    if isinstance(dt, datetime) and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def initialize_dynamics_fields(connection: dict, *, now: Optional[datetime] = None) -> dict:
    now = _ensure_tz(now)
    now_iso = now.isoformat() if isinstance(now, datetime) else now
    created_at = connection.get("created_at", now_iso)
    connection.setdefault("strength", DEFAULT_STRENGTH)
    connection.setdefault("stability", DEFAULT_STABILITY)
    connection.setdefault("last_activated", created_at)
    connection.setdefault("access_count", 0)
    return connection


def potentiate(
    connection: dict,
    *,
    increment: float = POTENTIATION_INCREMENT,
    now: Optional[datetime] = None,
) -> dict:
    now = _ensure_tz(now)
    initialize_dynamics_fields(connection, now=now)
    last_activated_str = connection.get("last_activated") or connection.get("created_at")
    last_dt = _parse_iso(last_activated_str)
    if last_dt is not None:
        hours_since = (now - last_dt).total_seconds() / 3600.0
    else:
        hours_since = 0.0
    current_strength = float(connection.get("strength", DEFAULT_STRENGTH))
    connection["strength"] = min(MAX_STRENGTH, current_strength + float(increment))
    if hours_since >= SPACED_INTERVAL_HOURS:
        current_stability = float(connection.get("stability", DEFAULT_STABILITY))
        connection["stability"] = current_stability + STABILITY_INCREMENT
    connection["last_activated"] = now.isoformat()
    connection["access_count"] = int(connection.get("access_count", 0)) + 1
    return connection


def apply_decay(connection: dict, *, now: Optional[datetime] = None) -> dict:
    now = _ensure_tz(now)
    initialize_dynamics_fields(connection, now=now)
    last_activated_str = connection.get("last_activated") or connection.get("created_at")
    last_dt = _parse_iso(last_activated_str)
    if last_dt is None:
        return connection
    days_since = (now - last_dt).total_seconds() / 86400.0
    if days_since <= 0:
        return connection
    stability = float(connection.get("stability", DEFAULT_STABILITY))
    if stability <= 0:
        stability = DEFAULT_STABILITY
    current_strength = float(connection.get("strength", DEFAULT_STRENGTH))
    decay_factor = math.exp(-days_since / stability)
    new_strength = current_strength * decay_factor
    connection["strength"] = max(STRENGTH_FLOOR, new_strength)
    return connection


def _parse_iso(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        v = value.strip()
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        dt = datetime.fromisoformat(v)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


__all__ = [
    "STRENGTH_FLOOR",
    "MAX_STRENGTH",
    "DEFAULT_STABILITY",
    "DEFAULT_STRENGTH",
    "POTENTIATION_INCREMENT",
    "SPACED_INTERVAL_HOURS",
    "STABILITY_INCREMENT",
    "initialize_dynamics_fields",
    "potentiate",
    "apply_decay",
]
