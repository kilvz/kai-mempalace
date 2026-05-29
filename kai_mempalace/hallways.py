"""Hallways — within-wing entity-to-entity connectors.

A **hallway** is a connection between two entities (people, projects,
concepts, interests) inside one wing, materialized from their
co-occurrence across that wing's drawers. Conceptually:

    WING -> has DRAWERS (each tagged with entities)
            entities -> connected to other entities by HALLWAYS
                       (within-wing, built from drawer co-occurrence)
                       hallways -> are the primitive
                                   tunnels -> use hallways to spawn
                                             cross-wing connections

If Aya and Lumi are both mentioned in 47 drawers across the diary,
letters, and ideas rooms, there's a hallway between them. If Aya
and "consciousness" co-occur in 19 drawers, there's a hallway between
them too. The hallway *is* the structural fact of "these two entities
travel together inside this wing."

Adapted from ChromaDB (mempalace) to FAISS/SQLite (kai-mempalace).

Persistence mirrors the original: a JSON file under ``~/.kai-palace/``
so the records survive across mines and are inspectable / editable by
hand if needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

from kai_mempalace.palace import Palace

logger = logging.getLogger("kai_mempalace_hallways")

# Persistence target. Mirrors the original ``_HALLWAY_FILE`` at
# ``~/.mempalace/hallways.json``, adapted to ``~/.kai-palace/``.
_HALLWAY_FILE = os.path.join(os.path.expanduser("~"), ".kai-palace", "hallways.json")

_SCHEMA_VERSION = 1


__all__ = [
    "compute_hallways_for_wing",
    "list_hallways",
    "delete_hallway",
]


# --------------------------------------------------------------------------
# Persistence — JSON file at _HALLWAY_FILE, restricted perms (0600) on POSIX
# --------------------------------------------------------------------------


def _load_hallways() -> list[dict]:
    """Read all hallway records. Returns ``[]`` if the file is missing or corrupt."""
    if not os.path.exists(_HALLWAY_FILE):
        return []
    try:
        with open(_HALLWAY_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.debug("hallways: load failed, treating as empty", exc_info=True)
        return []
    if isinstance(raw, dict) and "hallways" in raw:
        return raw.get("hallways") or []
    if isinstance(raw, list):
        return raw
    return []


def _save_hallways(hallways: list[dict]) -> None:
    """Atomically persist hallway records to _HALLWAY_FILE.

    Uses an os.replace temp-file dance so a crash mid-write doesn't
    corrupt the file. POSIX permission is restricted to 0600 because
    hallways reveal within-wing entity connections that the user may
    not want world-readable.
    """
    directory = os.path.dirname(_HALLWAY_FILE)
    os.makedirs(directory, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "hallways": list(hallways),
    }
    fd, tmp_path = tempfile.mkstemp(prefix=".hallways-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            # Non-POSIX systems may not support chmod; not fatal.
            pass
        os.replace(tmp_path, _HALLWAY_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Core algorithm — compute entity-pair hallways for one wing
# --------------------------------------------------------------------------


def _parse_entities(value) -> list[str]:
    """Drawer ``entities`` metadata is a semicolon-separated string. Parse it.

    Returns a deterministic *list* (not a set) because order matters for
    the deduplication semantics below: a drawer that mentions ``Aya;Aya``
    should only contribute one Aya to the entity set for that drawer.
    """
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip() for v in value.split(";") if v.strip()]
    else:
        return []
    # Dedupe while preserving first-seen order so id derivation is stable.
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _hallway_id(wing: str, entity_a: str, entity_b: str) -> str:
    """Deterministic id derived from wing + sorted entity pair.

    Sorting before hashing makes the id symmetric — (Aya, Lumi) and
    (Lumi, Aya) produce the same record. So an idempotent re-mine
    upserts the same hallway instead of creating two parallel records.
    """
    a, b = sorted([entity_a, entity_b])
    key = f"{wing}::{a}::{b}".encode("utf-8")
    suffix = hashlib.sha256(key).hexdigest()[:8]
    return f"hallway_{wing}_{a}_{b}_{suffix}"


def compute_hallways_for_wing(
    wing: str,
    palace_path: str = "~/.kai-palace",
    min_count: int = 2,
) -> list[dict]:
    """Compute entity-pair hallways for one wing.

    Algorithm:
      1. Open the palace at ``palace_path`` and query drawers for ``wing``
         from the SQLite backend.
      2. For each drawer with entities, every pair of distinct entities in
         that drawer is one co-occurrence. Increment a counter for each
         pair; also record the room the drawer lives in.
      3. For each (entity_a, entity_b) pair whose co-occurrence count is
         ``>= min_count``, materialize a hallway record. The record
         carries the pair, the count, and the set of rooms where they
         co-occurred (useful context for navigation).
      4. Persist the full hallway list (records for other wings preserved,
         this wing's records replaced) and return the just-computed list.

    Args:
        wing: wing name to scan.
        palace_path: path to the kai-mempalace data directory. Defaults to
            ``~/.kai-palace``.
        min_count: minimum co-occurrence count required to materialize a
            hallway between two entities. Default 2 — single co-occurrences
            are noise (entities mentioned together once in one drawer);
            two or more is a real signal. Clamped to ``>=1``.

    Returns:
        List of hallway dicts created for this wing. Records for other
        wings already on disk are preserved.
    """
    min_count = max(1, int(min_count))

    # 1. Open palace and query drawers for this wing.
    try:
        palace = Palace(palace_path)
        palace.init()
        conn = sqlite3.connect(str(palace._base / "palace.db"))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, content, metadata FROM drawers WHERE wing = ?", (wing,)
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "compute_hallways_for_wing: query failed for %s", wing, exc_info=True
        )
        return []

    if not rows:
        return []

    # 2. Walk drawers, counting entity-pair co-occurrence + tracking rooms.
    # pair_counts: {(entity_a, entity_b): count} — keys always sorted to
    # canonicalize the (a, b) vs (b, a) symmetry.
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    pair_rooms: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        meta = json.loads(row["metadata"])
        # Sentinel drawers carry no real content — skip them.
        if meta.get("is_sentinel"):
            continue
        entities = _parse_entities(meta.get("entities"))
        if len(entities) < 2:
            # Need at least 2 entities for a pair to exist.
            continue
        room = meta.get("room")
        room_str = room if isinstance(room, str) and room.strip() else None

        # Each unordered pair of distinct entities in this drawer is one
        # co-occurrence. itertools.combinations already gives unordered
        # pairs without repetition.
        for a, b in combinations(entities, 2):
            # Canonicalize order so (Aya, Lumi) and (Lumi, Aya) are the
            # same key. Skip self-pairs defensively.
            if a == b:
                continue
            key = tuple(sorted([a, b]))
            pair_counts[key] += 1
            if room_str:
                pair_rooms[key].add(room_str)

    if not pair_counts:
        return []

    # 3. Materialize hallway records for pairs above the threshold.
    #    Before building, load existing records so we can PRESERVE
    #    dynamics fields (strength, stability, last_activated, access_count)
    #    across recomputes. Without this preservation, every mine wipes
    #    the connection weights accumulated through use — defeating the
    #    living-connection layer entirely.
    existing = _load_hallways()
    existing_dynamics_lookup: dict = {}
    for h in existing:
        if h.get("wing") != wing:
            continue
        # Canonicalize the lookup key by sorting the entity pair — must
        # match the symmetric ID generation in _hallway_id (which also
        # sorts). Without this, a persisted record with reversed entity
        # order would silently miss the lookup and lose its accumulated
        # dynamics on every recompute.
        key = tuple(sorted([h.get("entity_a"), h.get("entity_b")]))
        # Only copy the fields the dynamics layer cares about; everything
        # else is recomputed deterministically from the drawer set.
        existing_dynamics_lookup[key] = {
            k: h[k] for k in ("strength", "stability", "last_activated", "access_count") if k in h
        }

    created: list[dict] = []
    created_at = datetime.now(timezone.utc).isoformat()
    for key in sorted(pair_counts.keys()):
        count = pair_counts[key]
        if count < min_count:
            continue
        entity_a, entity_b = key
        rooms = sorted(pair_rooms.get(key, set()))
        room_summary = ", ".join(rooms[:3]) if rooms else "(no room tags)"
        if len(rooms) > 3:
            room_summary += f", +{len(rooms) - 3} more"
        record = {
            "id": _hallway_id(wing, entity_a, entity_b),
            "wing": wing,
            "entity_a": entity_a,
            "entity_b": entity_b,
            "co_occurrence_count": count,
            "rooms": rooms,
            "label": f"{entity_a} \u2194 {entity_b} (co-occur in {count} drawers across {len(rooms) or 'no'} room{'s' if len(rooms) != 1 else ''}: {room_summary})",
            "created_at": created_at,
            "created_by": "auto",
        }
        # Apply preserved dynamics if this entity pair existed in the
        # prior wing snapshot.
        preserved = existing_dynamics_lookup.get(key, {})
        record.update(preserved)
        # NOTE: initialize_dynamics_fields intentionally omitted —
        #       kai-mempalace does not include that LLM-dependent module.
        created.append(record)

    # 4. Persist — preserve other-wing records, replace this wing's records.
    preserved_other_wings = [h for h in existing if h.get("wing") != wing]
    _save_hallways(preserved_other_wings + created)

    return created


# --------------------------------------------------------------------------
# Query API — list_hallways, delete_hallway
# --------------------------------------------------------------------------


def list_hallways(wing: Optional[str] = None) -> list[dict]:
    """List hallway records. Filter by ``wing`` if specified."""
    all_hallways = _load_hallways()
    if wing is None:
        return list(all_hallways)
    return [h for h in all_hallways if h.get("wing") == wing]


def delete_hallway(hallway_id: str) -> bool:
    """Remove one hallway record by id. Returns True if a record was removed."""
    hallways = _load_hallways()
    filtered = [h for h in hallways if h.get("id") != hallway_id]
    if len(filtered) == len(hallways):
        return False
    _save_hallways(filtered)
    return True
