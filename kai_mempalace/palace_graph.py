"""
palace_graph.py — Graph traversal layer for Kai MemPalace
=========================================================

Builds a navigable graph from the palace structure:
  - Nodes = rooms (named ideas)
  - Edges = shared rooms across wings (tunnels)
  - Edge types = halls (the corridors)

Enables queries like:
  "Start at chromadb-setup in wing_code, walk to wing_myproject"
  "Find all rooms connected to riley-college-apps"
  "What topics bridge wing_hardware and wing_myproject?"

No external graph DB needed — built from SQLite metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from kai_mempalace.config import KaiPalaceConfig, normalize_wing_name
from kai_mempalace.palace import mine_lock
from kai_mempalace.palace import Palace

logger = logging.getLogger("kai_mempalace_graph")


def _normalize_wing(wing: str | None) -> str | None:
    if not isinstance(wing, str):
        return None
    wing = wing.strip()
    if not wing:
        return None
    return normalize_wing_name(wing)


_graph_cache_lock = threading.Lock()
_graph_cache_nodes = None
_graph_cache_edges = None
_graph_cache_time = 0.0
_GRAPH_CACHE_TTL = 60.0


def invalidate_graph_cache():
    global _graph_cache_nodes, _graph_cache_edges, _graph_cache_time
    with _graph_cache_lock:
        _graph_cache_nodes = None
        _graph_cache_edges = None
        _graph_cache_time = 0.0


def _get_palace(config=None):
    config = config or KaiPalaceConfig()
    try:
        palace = Palace(config.palace_path)
        palace.init()
        return palace
    except Exception:
        return None


def build_graph(palace=None, config=None):
    """
    Build the palace graph from SQLite metadata.

    Returns cached result if fresh (within TTL). Cache is invalidated
    on writes via invalidate_graph_cache(). Thread-safe via _graph_cache_lock.

    Note: warm cache ignores ``palace`` and ``config`` arguments — this is
    intentional for the MCP server's single-palace use case. Callers
    switching collections should call ``invalidate_graph_cache()`` first.

    Returns:
        nodes: dict of {room: {wings: set, halls: set, count: int}}
        edges: list of {room, wing_a, wing_b, hall} — one per tunnel crossing
    """
    global _graph_cache_nodes, _graph_cache_edges, _graph_cache_time
    now = time.time()
    with _graph_cache_lock:
        if _graph_cache_nodes is not None and (now - _graph_cache_time) < _GRAPH_CACHE_TTL:
            return _graph_cache_nodes, _graph_cache_edges

    if palace is None:
        palace = _get_palace(config)
    if not palace:
        return {}, []

    conn = sqlite3.connect(str(palace._base / "palace.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id, wing, room, metadata FROM drawers").fetchall()
    finally:
        conn.close()

    room_data = defaultdict(lambda: {"wings": set(), "halls": set(), "count": 0, "dates": set()})

    for row in rows:
        meta = json.loads(row["metadata"] or "{}")
        room = row["room"] or meta.get("room", "")
        wing = row["wing"] or meta.get("wing", "")
        hall = meta.get("hall", "")
        date = meta.get("date", "")
        if room and room != "general" and wing:
            room_data[room]["wings"].add(wing)
            if hall:
                room_data[room]["halls"].add(hall)
            if date:
                room_data[room]["dates"].add(date)
            room_data[room]["count"] += 1

    edges = []
    for room, data in room_data.items():
        wings = sorted(data["wings"])
        if len(wings) >= 2:
            for i, wa in enumerate(wings):
                for wb in wings[i + 1:]:
                    for hall in data["halls"]:
                        edges.append({
                            "room": room,
                            "wing_a": wa,
                            "wing_b": wb,
                            "hall": hall,
                            "count": data["count"],
                        })

    nodes = {}
    for room, data in room_data.items():
        nodes[room] = {
            "wings": sorted(data["wings"]),
            "halls": sorted(data["halls"]),
            "count": data["count"],
            "dates": sorted(data["dates"])[-5:] if data["dates"] else [],
        }

    if nodes:
        with _graph_cache_lock:
            _graph_cache_nodes = nodes
            _graph_cache_edges = edges
            _graph_cache_time = time.time()

    return nodes, edges


def traverse(start_room: str, palace=None, config=None, max_hops: int = 2):
    """
    Walk the graph from a starting room. Find connected rooms
    through shared wings.

    Returns list of paths: [{room, wing, hall, hop_distance}]
    """
    nodes, edges = build_graph(palace, config)

    if start_room not in nodes:
        return {
            "error": f"Room '{start_room}' not found",
            "suggestions": _fuzzy_match(start_room, nodes),
        }

    start = nodes[start_room]
    visited = {start_room}
    results = [{
        "room": start_room,
        "wings": start["wings"],
        "halls": start["halls"],
        "count": start["count"],
        "hop": 0,
    }]

    frontier = [(start_room, 0)]
    while frontier:
        current_room, depth = frontier.pop(0)
        if depth >= max_hops:
            continue

        current = nodes.get(current_room, {})
        current_wings = set(current.get("wings", []))

        for room, data in nodes.items():
            if room in visited:
                continue
            shared_wings = current_wings & set(data["wings"])
            if shared_wings:
                visited.add(room)
                results.append({
                    "room": room,
                    "wings": data["wings"],
                    "halls": data["halls"],
                    "count": data["count"],
                    "hop": depth + 1,
                    "connected_via": sorted(shared_wings),
                })
                if depth + 1 < max_hops:
                    frontier.append((room, depth + 1))

    results.sort(key=lambda x: (x["hop"], -x["count"]))
    return results[:50]


def find_tunnels(wing_a: str = None, wing_b: str = None, palace=None, config=None):
    """
    Find rooms that connect two wings (or all tunnel rooms if no wings specified).
    These are the "hallways" — same named idea appearing in multiple domains.
    """
    nodes, edges = build_graph(palace, config)

    norm_a = _normalize_wing(wing_a)
    norm_b = _normalize_wing(wing_b)

    tunnels = []
    for room, data in nodes.items():
        wings = data["wings"]
        if len(wings) < 2:
            continue

        if norm_a and norm_a not in wings:
            continue
        if norm_b and norm_b not in wings:
            continue

        tunnels.append({
            "room": room,
            "wings": wings,
            "halls": data["halls"],
            "count": data["count"],
            "recent": data["dates"][-1] if data["dates"] else "",
        })

    if not tunnels and (wing_a or wing_b):
        logger.warning(
            "No tunnels found for wing filter(s): wing_a=%r (normalized=%r), wing_b=%r (normalized=%r)",
            wing_a,
            norm_a,
            wing_b,
            norm_b,
        )

    tunnels.sort(key=lambda x: -x["count"])
    return tunnels[:50]


def graph_stats(palace=None, config=None):
    """Summary statistics about the palace graph."""
    nodes, edges = build_graph(palace, config)

    tunnel_rooms = sum(1 for n in nodes.values() if len(n["wings"]) >= 2)
    wing_counts = Counter()
    for data in nodes.values():
        for w in data["wings"]:
            wing_counts[w] += 1

    return {
        "total_rooms": len(nodes),
        "tunnel_rooms": tunnel_rooms,
        "total_edges": len(edges),
        "rooms_per_wing": dict(wing_counts.most_common()),
        "top_tunnels": [
            {"room": r, "wings": d["wings"], "count": d["count"]}
            for r, d in sorted(nodes.items(), key=lambda x: -len(x[1]["wings"]))[:10]
            if len(d["wings"]) >= 2
        ],
    }


def _fuzzy_match(query: str, nodes: dict, n: int = 5):
    """Find rooms that approximately match a query string."""
    query_lower = query.lower()
    scored = []
    for room in nodes:
        if query_lower in room:
            scored.append((room, 1.0))
        elif any(word in room for word in query_lower.split("-")):
            scored.append((room, 0.5))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:n]]


def _get_tunnel_file(config=None) -> str:
    """Return the path to the tunnels.json file, derived from KaiPalaceConfig.palace_path."""
    config = config or KaiPalaceConfig()
    return config.tunnel_file


def _legacy_tunnel_file() -> str:
    """The pre-3.3.6 hardcoded path. Kept only for one-time orphan detection."""
    return os.path.join(os.path.expanduser("~"), ".mempalace", "tunnels.json")


def _load_tunnels(config=None):
    """Load explicit tunnels from disk.

    Returns an empty list if the file is missing or corrupt (e.g. truncated
    by a crash mid-write on a system that lacks atomic-rename semantics).

    Backwards-compatibility: prior to 3.3.6 the tunnel file was hardcoded at
    ``~/.mempalace/tunnels.json`` regardless of the configured palace_path.
    If the configured tunnel file is missing but a legacy file exists at a
    different path, log a one-line warning naming both paths so users can
    move the file manually. We do NOT auto-migrate.

    ``config`` may be passed in by the caller to avoid re-instantiating
    ``KaiPalaceConfig`` (which re-reads config from disk) on every helper
    call within a single create_tunnel cycle.
    """
    current_tunnel_file = _get_tunnel_file(config)
    if os.path.exists(current_tunnel_file):
        try:
            with open(current_tunnel_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logger.warning(
                "Tunnels file '%s' is corrupt or unreadable; starting empty.",
                current_tunnel_file,
            )
            return []
        return data if isinstance(data, list) else []

    legacy = _legacy_tunnel_file()
    if legacy != current_tunnel_file and os.path.exists(legacy):
        logger.warning(
            "Legacy tunnels file at '%s' is being ignored; configured location is '%s'. "
            "Move or copy the legacy file to the configured path to recover its tunnels.",
            legacy,
            current_tunnel_file,
        )
    return []


def _save_tunnels(tunnels, config=None):
    """Persist explicit tunnels atomically.

    Writes to ``tunnels.json.tmp`` then ``os.replace``s it into place, so
    a crash mid-write can never leave a partial/empty tunnels.json that
    silently wipes every tunnel on next read.

    Also restricts the parent directory to 0o700 and the file to 0o600.

    ``config`` may be passed in by the caller to avoid re-instantiating
    ``KaiPalaceConfig`` on every save.
    """
    tunnel_file = _get_tunnel_file(config)
    parent = os.path.dirname(tunnel_file)
    os.makedirs(parent, exist_ok=True)
    try:
        os.chmod(parent, 0o700)
    except (OSError, NotImplementedError):
        pass
    tmp_path = tunnel_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(tunnels, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, tunnel_file)
    try:
        os.chmod(tunnel_file, 0o600)
    except (OSError, NotImplementedError):
        pass


def _endpoint_key(wing: str, room: str) -> str:
    return f"{wing}/{room}"


def _canonical_tunnel_id(
    source_wing: str, source_room: str, target_wing: str, target_room: str
) -> str:
    src = _endpoint_key(source_wing, source_room)
    tgt = _endpoint_key(target_wing, target_room)
    a, b = sorted((src, tgt))
    return hashlib.sha256(f"{a}↔{b}".encode()).hexdigest()[:16]


def _require_name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _check_room_exists(wing: str, room: str, palace) -> bool:
    """Check if at least one drawer exists for the given wing/room in SQLite."""
    if palace is None:
        return True
    try:
        conn = sqlite3.connect(str(palace._base / "palace.db"))
        row = conn.execute(
            "SELECT COUNT(*) FROM drawers WHERE wing=? AND room=?", (wing, room)
        ).fetchone()
        conn.close()
        return row[0] > 0
    except Exception:
        logger.warning(
            "Error checking room existence for %s/%s; allowing tunnel creation.",
            wing,
            room,
            exc_info=True,
        )
        return True


def create_tunnel(
    source_wing: str,
    source_room: str,
    target_wing: str,
    target_room: str,
    label: str = "",
    source_drawer_id: str = None,
    target_drawer_id: str = None,
    kind: str = "explicit",
):
    """Create an explicit (symmetric) tunnel between two locations in the palace.

    Tunnels are undirected: ``create_tunnel(A, B)`` and ``create_tunnel(B, A)``
    resolve to the same canonical ID. A second call with the same endpoints
    updates the stored label (and drawer IDs, if provided) rather than
    creating a duplicate. Endpoints are compared **verbatim** — ``"my-wing"``
    and ``"my_wing"`` are distinct (see Note below and #1504).

    The ``source`` / ``target`` fields on the returned dict preserve the
    argument order the caller used, so callers can display it directionally
    if they like. The ID and dedup are symmetric.

    Args:
        source_wing: Wing of the source (e.g., "project_api").
        source_room: Room in the source wing.
        target_wing: Wing of the target (e.g., "project_database").
        target_room: Room in the target wing.
        label: Description of the connection.
        source_drawer_id: Optional specific drawer ID.
        target_drawer_id: Optional specific drawer ID.
        kind: Tunnel category — ``"explicit"`` (default, user-created link
            between real rooms) or ``"topic"`` (auto-generated cross-wing
            topical link where rooms are synthetic ``topic:<name>``
            identifiers).

    Returns:
        The stored tunnel dict.

    Raises:
        ValueError: if any wing or room is empty or non-string, or if an explicit
                    tunnel points to a nonexistent room.

    Note:
        Wing slugs are stored verbatim — passing ``"my-wing"`` and ``"my_wing"``
        produces two distinct tunnels (canonical IDs differ). Read-path helpers
        (``list_tunnels`` / ``follow_tunnels``) normalize both sides at compare
        time so legacy underscore data and explicit-flag hyphen data both
        match queries in either form. See #1504.
    """
    source_wing = _require_name(source_wing, "source_wing")
    source_room = _require_name(source_room, "source_room")
    target_wing = _require_name(target_wing, "target_wing")
    target_room = _require_name(target_room, "target_room")

    config = KaiPalaceConfig()

    if kind == "explicit":
        palace = _get_palace(config)
        if not _check_room_exists(source_wing, source_room, palace):
            raise ValueError(f"Source room '{source_room}' does not exist in wing '{source_wing}'")
        if not _check_room_exists(target_wing, target_room, palace):
            raise ValueError(f"Target room '{target_room}' does not exist in wing '{target_wing}'")

    tunnel_id = _canonical_tunnel_id(source_wing, source_room, target_wing, target_room)

    tunnel = {
        "id": tunnel_id,
        "source": {"wing": source_wing, "room": source_room},
        "target": {"wing": target_wing, "room": target_room},
        "label": label,
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if source_drawer_id:
        tunnel["source"]["drawer_id"] = source_drawer_id
    if target_drawer_id:
        tunnel["target"]["drawer_id"] = target_drawer_id

    with mine_lock(_get_tunnel_file(config)):
        tunnels = _load_tunnels(config)
        for existing in tunnels:
            if existing.get("id") == tunnel_id:
                tunnel["created_at"] = existing.get("created_at", tunnel["created_at"])
                tunnel["updated_at"] = datetime.now(timezone.utc).isoformat()
                existing.clear()
                existing.update(tunnel)
                _save_tunnels(tunnels, config)
                return existing
        tunnels.append(tunnel)
        _save_tunnels(tunnels, config)
    return tunnel


def list_tunnels(wing: str = None):
    """List all explicit tunnels, optionally filtered by wing.

    Returns tunnels where ``wing`` appears as either source or target
    (tunnels are symmetric, so either endpoint is a valid filter match).
    """
    norm_wing = _normalize_wing(wing)
    tunnels = _load_tunnels()
    if norm_wing:
        tunnels = [
            t
            for t in tunnels
            if _normalize_wing((t.get("source") or {}).get("wing")) == norm_wing
            or _normalize_wing((t.get("target") or {}).get("wing")) == norm_wing
        ]
    return tunnels


def delete_tunnel(tunnel_id: str):
    """Delete an explicit tunnel by ID. Returns ``{"deleted": <id>}``."""
    with mine_lock(_get_tunnel_file()):
        tunnels = _load_tunnels()
        tunnels = [t for t in tunnels if t.get("id") != tunnel_id]
        _save_tunnels(tunnels)
    return {"deleted": tunnel_id}


def follow_tunnels(wing: str, room: str, palace=None, config=None):
    """Follow explicit tunnels from a room — returns connected drawers.

    Given a location (wing/room), finds all tunnels leading from or to it,
    and optionally fetches the connected drawer content.
    """
    norm_wing = _normalize_wing(wing) or wing
    tunnels = _load_tunnels()
    connections = []

    for t in tunnels:
        src = t.get("source") or {}
        tgt = t.get("target") or {}

        if _normalize_wing(src.get("wing")) == norm_wing and src.get("room") == room:
            connections.append({
                "direction": "outgoing",
                "connected_wing": tgt["wing"],
                "connected_room": tgt["room"],
                "label": t.get("label", ""),
                "drawer_id": tgt.get("drawer_id"),
                "tunnel_id": t["id"],
            })
        elif _normalize_wing(tgt.get("wing")) == norm_wing and tgt.get("room") == room:
            connections.append({
                "direction": "incoming",
                "connected_wing": src["wing"],
                "connected_room": src["room"],
                "label": t.get("label", ""),
                "drawer_id": src.get("drawer_id"),
                "tunnel_id": t["id"],
            })

    if not connections:
        logger.warning("No explicit tunnels found for %s/%s", wing, room)

    if palace and connections:
        try:
            for c in connections:
                did = c.get("drawer_id")
                if did:
                    drawer = palace.get_drawer(did)
                    if drawer:
                        c["drawer_preview"] = drawer["content"][:300]
        except Exception:
            logger.debug("Drawer preview hydration failed", exc_info=True)

    return connections


TOPIC_ROOM_PREFIX = "topic:"


def _normalize_topic(name: str) -> str:
    return str(name).strip().lower()


def topic_room(name: str) -> str:
    return f"{TOPIC_ROOM_PREFIX}{name}"


def compute_topic_tunnels(
    topics_by_wing: dict,
    min_count: int = 1,
    label_prefix: str = "shared topic",
) -> list[dict]:
    """Create tunnels for every pair of wings that share >= ``min_count`` topics.

    Args:
        topics_by_wing: ``{wing_name: [topic_name, ...]}`` mapping. Topic
            names are compared case-insensitively; the first observed
            casing is used for the tunnel room name.
        min_count: minimum number of overlapping topics required to drop
            any tunnel between a wing pair.
        label_prefix: human-readable string prefixed to the tunnel label.

    Returns:
        List of tunnel dicts as returned by ``create_tunnel`` — one per
        (wing_a, wing_b, topic) triple that crossed the threshold.

    No-op semantics:
      - empty/None ``topics_by_wing`` returns ``[]``.
      - wings whose topic list is empty are skipped.
      - ``min_count <= 0`` is clamped to 1.
    """
    if not topics_by_wing:
        return []

    min_count = max(1, int(min_count))

    wing_topics: dict[str, dict[str, str]] = {}
    for wing, names in topics_by_wing.items():
        if not isinstance(wing, str) or not wing.strip():
            continue
        if not isinstance(names, (list, tuple)):
            continue
        bucket: dict[str, str] = {}
        for n in names:
            if not isinstance(n, str):
                continue
            key = _normalize_topic(n)
            if not key:
                continue
            bucket.setdefault(key, n.strip())
        if bucket:
            wing_topics[normalize_wing_name(wing.strip())] = bucket

    wings = sorted(wing_topics.keys())
    created: list[dict] = []
    for i, wa in enumerate(wings):
        topics_a = wing_topics[wa]
        for wb in wings[i + 1:]:
            topics_b = wing_topics[wb]
            shared_keys = set(topics_a.keys()) & set(topics_b.keys())
            if len(shared_keys) < min_count:
                continue
            for key in sorted(shared_keys):
                topic_name = topics_a[key] if topics_a[key] else topics_b[key]
                room = topic_room(topic_name)
                tunnel = create_tunnel(
                    source_wing=wa,
                    source_room=room,
                    target_wing=wb,
                    target_room=room,
                    label=f"{label_prefix}: {topic_name}",
                    kind="topic",
                )
                created.append(tunnel)
    return created


def topic_tunnels_for_wing(
    wing: str,
    topics_by_wing: dict,
    min_count: int = 1,
    label_prefix: str = "shared topic",
) -> list[dict]:
    """Compute topic tunnels involving a single wing.

    Used by the miner to incrementally update tunnels for the wing that
    just finished mining without recomputing pairs that don't involve it.
    Returns the list of tunnels created or refreshed.
    """
    if not topics_by_wing or not isinstance(wing, str) or not wing.strip():
        return []

    wing = normalize_wing_name(wing.strip())
    own = topics_by_wing.get(wing)
    if own is None:
        for k, v in topics_by_wing.items():
            if isinstance(k, str) and normalize_wing_name(k.strip()) == wing:
                own = v
                break
    if not isinstance(own, (list, tuple)) or not own:
        return []

    created: list[dict] = []
    for other, other_topics in topics_by_wing.items():
        if not isinstance(other, str) or not other.strip():
            continue
        if normalize_wing_name(other.strip()) == wing:
            continue
        if not isinstance(other_topics, (list, tuple)) or not other_topics:
            continue
        slice_map = {wing: list(own), other: list(other_topics)}
        created.extend(
            compute_topic_tunnels(
                slice_map,
                min_count=min_count,
                label_prefix=label_prefix,
            )
        )
    return created


def entity_tunnels_for_wing(
    wing: str,
    hallways: list,
    label_prefix: str = "shared entity",
) -> list:
    """Compute entity tunnels involving a single wing.

    An entity tunnel bridges two wings when the same entity (person,
    project, concept, interest) appears in within-wing hallways of both.
    This is the architectural counterpart to ``topic_tunnels_for_wing`` —
    same storage path (``create_tunnel`` → tunnels.json),
    same dedup, same listing API.

    Endpoints use the synthetic room id ``entity:<name>`` (mirrors
    ``topic:<slug>``) so they can't collide with literal folder-derived
    rooms of the same name. Casing of the entity is preserved.

    Topic tunnels are NOT replaced — both systems coexist for one release
    cycle while entity tunnels prove out. Deprecation is a separate PR.
    """
    if not hallways or not isinstance(wing, str) or not wing.strip():
        return []

    wing_norm = normalize_wing_name(wing.strip())

    entity_wings: dict = {}
    for h in hallways:
        if not isinstance(h, dict):
            continue
        h_wing = h.get("wing")
        if not isinstance(h_wing, str) or not h_wing.strip():
            continue
        h_wing_norm = normalize_wing_name(h_wing.strip())
        for ent_key in ("entity_a", "entity_b"):
            ent = h.get(ent_key)
            if not isinstance(ent, str) or not ent.strip():
                continue
            entity_wings.setdefault(ent, {}).setdefault(h_wing_norm, h_wing)

    if not entity_wings:
        return []

    created: list = []
    for entity in sorted(entity_wings.keys()):
        wings_for_entity = entity_wings[entity]
        if wing_norm not in wings_for_entity:
            continue
        own_wing_display = wings_for_entity[wing_norm]
        other_wings_norm = sorted(w for w in wings_for_entity if w != wing_norm)
        for other_norm in other_wings_norm:
            other_display = wings_for_entity[other_norm]
            room = f"entity:{entity}"
            tunnel = create_tunnel(
                source_wing=own_wing_display,
                source_room=room,
                target_wing=other_display,
                target_room=room,
                label=f"{label_prefix}: {entity}",
                kind="entity",
            )
            created.append(tunnel)
    return created
