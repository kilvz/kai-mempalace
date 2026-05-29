"""MCP (Model Context Protocol) server for Kai MemPalace.

Exposes all Palace operations as JSON-RPC tools over stdio or SSE transport.
"""

import json
import logging
import sys
import traceback
from typing import Any, Optional

try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse

    HAS_HTTP = True
except ImportError:
    HAS_HTTP = False

from kai_mempalace import dialect
from kai_mempalace.dialect import aaak_compress, aaak_decompress, aaak_parse_entry
from kai_mempalace.entity_registry import EntityRegistry
from kai_mempalace import palace_graph
from kai_mempalace.sync import sync_palace
from kai_mempalace.config import KaiPalaceConfig

try:
    from kai_mempalace.layers import MemoryStack

    HAS_LAYERS = True
except ImportError:
    HAS_LAYERS = False

try:
    from kai_mempalace import miner

    HAS_MINER = True
except ImportError:
    HAS_MINER = False


logger = logging.getLogger(__name__)

VERSION = "4.2.0"
MCP_PROTOCOL_VERSION = "2024-11-05"

_TOOL_DEFINITIONS: list[dict] = []

AAAK_SPEC = (
    "AAAK is a compressed memory dialect that MemPalace uses for efficient storage. "
    "It is designed to be readable by both humans and LLMs without decoding.\n\n"
    "FORMAT:\n"
    "  ENTITIES: 3-letter uppercase codes. ALC=Alice, JOR=Jordan, RIL=Riley, MAX=Max, BEN=Ben.\n"
    "  EMOTIONS: *action markers* before/during text. *warm*=joy, *fierce*=determined, *raw*=vulnerable, *bloom*=tenderness.\n"
    "  STRUCTURE: Pipe-separated fields. FAM: family | PROJ: projects | ⚠: warnings/reminders.\n"
    "  DATES: ISO format (2026-03-31). COUNTS: Nx = N mentions (e.g., 570x).\n"
    "  IMPORTANCE: ★ to ★★★★★ (1-5 scale).\n"
    "  HALLS: hall_facts, hall_events, hall_discoveries, hall_preferences, hall_advice.\n"
    "  WINGS: kai_mempalace, documents, reference, benchmark, agent_*\n"
    "  ROOMS: Hyphenated slugs representing named ideas (e.g., chromadb-setup, gpu-pricing).\n\n"
    "EXAMPLE:\n"
    "  FAM: ALC→♡JOR | 2D(kids): RIL(18,sports) MAX(11,chess+swimming) | BEN(contributor)\n\n"
    "Read AAAK naturally — expand codes mentally, treat *markers* as emotional context.\n"
    "When WRITING AAAK: use entity codes, mark emotions, keep structure tight."
)


def _build_tool_definitions() -> list[dict]:
    """Build MCP tool definitions from the registered methods."""
    if _TOOL_DEFINITIONS:
        return _TOOL_DEFINITIONS
    tools = [
        {
            "name": "search",
            "description": "Search across all palace drawers using hybrid (vector+keyword), vector-only, or keyword-only mode",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query text"},
                    "n_results": {"type": "number", "description": "Max results to return (default 10)"},
                    "wing": {"type": "string", "description": "Restrict to a wing"},
                    "room": {"type": "string", "description": "Restrict to a room"},
                    "mode": {"type": "string", "description": "Search mode: hybrid, vector, or keyword"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_status",
            "description": "Get palace status: drawer count, wing/room breakdown, embedding type",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_wings",
            "description": "List all wings in the palace",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_rooms",
            "description": "List rooms in a wing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing name (optional; lists all rooms if omitted)"},
                },
            },
        },
        {
            "name": "add_drawer",
            "description": "Add a new drawer with content to a wing/room",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing name"},
                    "room": {"type": "string", "description": "Room name"},
                    "content": {"type": "string", "description": "Drawer content text"},
                    "metadata": {"type": "object", "description": "Optional metadata dict"},
                    "source_file": {"type": "string", "description": "Optional source file path"},
                },
                "required": ["wing", "room", "content"],
            },
        },
        {
            "name": "get_drawer",
            "description": "Get a drawer by its ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "drawer_id": {"type": "string", "description": "Drawer ID"},
                },
                "required": ["drawer_id"],
            },
        },
        {
            "name": "delete_drawer",
            "description": "Delete a drawer by its ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "drawer_id": {"type": "string", "description": "Drawer ID"},
                },
                "required": ["drawer_id"],
            },
        },
        {
            "name": "kg_add",
            "description": "Add a fact to the knowledge graph",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Subject entity"},
                    "predicate": {"type": "string", "description": "Relationship type"},
                    "object": {"type": "string", "description": "Object entity"},
                    "valid_from": {"type": "string", "description": "ISO date when fact becomes valid"},
                    "valid_to": {"type": "string", "description": "ISO date when fact expires"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        {
            "name": "kg_query",
            "description": "Query the knowledge graph for facts about an entity",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity name to query"},
                    "predicate": {"type": "string", "description": "Filter by predicate"},
                    "as_of": {"type": "string", "description": "ISO date to query temporally"},
                    "all": {"type": "boolean", "description": "Return all facts"},
                },
            },
        },
        {
            "name": "kg_invalidate",
            "description": "Mark a knowledge graph fact as no longer true",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Subject entity"},
                    "predicate": {"type": "string", "description": "Relationship type"},
                    "object": {"type": "string", "description": "Object entity"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
        {
            "name": "kg_stats",
            "description": "Get knowledge graph statistics",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "diary_write",
            "description": "Write a diary entry for an agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent name"},
                    "entry": {"type": "string", "description": "Diary entry text"},
                    "topic": {"type": "string", "description": "Topic tag"},
                },
                "required": ["agent", "entry"],
            },
        },
        {
            "name": "diary_read",
            "description": "Read recent diary entries for an agent",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "agent": {"type": "string", "description": "Agent name"},
                    "last_n": {"type": "number", "description": "Number of entries to read"},
                },
                "required": ["agent"],
            },
        },
        {
            "name": "list_drawers",
            "description": "List drawers with optional wing/room filter and pagination",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Filter by wing"},
                    "room": {"type": "string", "description": "Filter by room"},
                    "limit": {"type": "number", "description": "Max results (default 20)"},
                    "offset": {"type": "number", "description": "Offset for pagination"},
                },
            },
        },
        {
            "name": "mine_text",
            "description": "Mine text content into the palace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text content to mine"},
                    "wing": {"type": "string", "description": "Target wing"},
                    "room": {"type": "string", "description": "Target room"},
                    "source": {"type": "string", "description": "Optional source identifier"},
                },
                "required": ["text", "wing", "room"],
            },
        },
        {
            "name": "aaak_compress",
            "description": "Compress text using AAAK dialect",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to compress"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "aaak_decompress",
            "description": "Decompress AAAK-encoded text",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "AAAK text to decompress"},
                },
                "required": ["text"],
            },
        },
        {
            "name": "aaak_parse",
            "description": "Parse a single AAAK entry into its components",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "AAAK entry text"},
                },
                "required": ["text"],
            },
        },
    ]
    _TOOL_DEFINITIONS.extend(tools)

    # -- Tools missing from kai-palace (ported from mempalace upstream) --

    extra_tools = [
        {
            "name": "update_drawer",
            "description": "Update an existing drawer's content and/or metadata (wing, room)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "drawer_id": {"type": "string", "description": "Drawer ID"},
                    "content": {"type": "string", "description": "New content (optional)"},
                    "metadata": {"type": "object", "description": "New metadata (optional)"},
                    "wing": {"type": "string", "description": "New wing (optional)"},
                    "room": {"type": "string", "description": "New room (optional)"},
                },
                "required": ["drawer_id"],
            },
        },
        {
            "name": "check_duplicate",
            "description": "Check if content already exists in the palace before filing",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to check"},
                    "threshold": {"type": "number", "description": "Similarity threshold 0-1 (default 0.9)"},
                },
                "required": ["content"],
            },
        },
        {
            "name": "kg_timeline",
            "description": "Chronological timeline of facts. Shows the story of an entity (or everything) in order.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description": "Entity to get timeline for (optional — omit for full timeline)"},
                },
            },
        },
        {
            "name": "create_tunnel",
            "description": "Create a cross-wing tunnel linking two palace locations",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_wing": {"type": "string", "description": "Wing of the source"},
                    "source_room": {"type": "string", "description": "Room in the source wing"},
                    "target_wing": {"type": "string", "description": "Wing of the target"},
                    "target_room": {"type": "string", "description": "Room in the target wing"},
                    "label": {"type": "string", "description": "Description of the connection"},
                    "source_drawer_id": {"type": "string", "description": "Optional specific drawer ID"},
                    "target_drawer_id": {"type": "string", "description": "Optional specific drawer ID"},
                },
                "required": ["source_wing", "source_room", "target_wing", "target_room"],
            },
        },
        {
            "name": "delete_tunnel",
            "description": "Delete an explicit tunnel by its ID",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tunnel_id": {"type": "string", "description": "Tunnel ID to delete"},
                },
                "required": ["tunnel_id"],
            },
        },
        {
            "name": "find_tunnels",
            "description": "Find rooms that bridge two wings — the hallways connecting different domains",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing_a": {"type": "string", "description": "First wing (optional)"},
                    "wing_b": {"type": "string", "description": "Second wing (optional)"},
                },
            },
        },
        {
            "name": "follow_tunnels",
            "description": "Follow tunnels from a room to see what it connects to in other wings",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing to start from"},
                    "room": {"type": "string", "description": "Room to follow tunnels from"},
                },
                "required": ["wing", "room"],
            },
        },
        {
            "name": "list_tunnels",
            "description": "List all explicit cross-wing tunnels. Optionally filter by wing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Filter tunnels by wing"},
                },
            },
        },
        {
            "name": "traverse",
            "description": "Walk the palace graph from a room. Shows connected ideas across wings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_room": {"type": "string", "description": "Room to start from"},
                    "max_hops": {"type": "number", "description": "How many connections to follow (default: 2)"},
                },
                "required": ["start_room"],
            },
        },
        {
            "name": "graph_stats",
            "description": "Palace graph overview: total rooms, tunnel connections, edges between wings.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_taxonomy",
            "description": "Full taxonomy: wing → room → drawer count",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_aaak_spec",
            "description": "Get the AAAK dialect specification — the compressed memory format",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "memories_filed_away",
            "description": "Check if a recent palace checkpoint was saved. Returns drawer count and timestamp.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "sync",
            "description": "Prune drawers whose source files are gitignored, deleted, or moved",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Project root to scope the sync"},
                    "wing": {"type": "string", "description": "Limit to one wing"},
                    "apply": {"type": "boolean", "description": "Actually delete drawers; default is dry-run preview"},
                },
            },
        },
        {
            "name": "reconnect",
            "description": "Force reconnect to the palace database. Use after external changes.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "hook_settings",
            "description": "Get or set hook behavior. silent_save: True = save directly, desktop_toast: True = show notification.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "silent_save": {"type": "boolean", "description": "True = silent direct save"},
                    "desktop_toast": {"type": "boolean", "description": "True = show desktop toast via notify-send"},
                },
            },
        },
        {
            "name": "mine_file",
            "description": "Mine a single file into the palace (chunks, extracts metadata, stores drawers)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "Path to the file to mine"},
                    "wing": {"type": "string", "description": "Target wing"},
                    "room": {"type": "string", "description": "Target room"},
                },
                "required": ["filepath", "wing", "room"],
            },
        },
        {
            "name": "batch_mine",
            "description": "Mine all matching files in a directory into the palace",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to scan for files"},
                    "wing": {"type": "string", "description": "Target wing (optional; auto-detected from dir name)"},
                    "pattern": {"type": "string", "description": "Glob pattern to filter files (e.g. *.md, *.txt)"},
                },
                "required": ["directory"],
            },
        },
        {
            "name": "rebuild_fts",
            "description": "Rebuild the FTS5 full-text search index from all drawer contents",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_embedder",
            "description": "Switch to a different embedding model and optionally reindex all drawers",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Embedder name: sentence, spacy, numpy, minilm, embeddinggemma"},
                    "reindex": {"type": "boolean", "description": "Re-embed all existing drawers (default true)"},
                },
                "required": ["model"],
            },
        },
        {
            "name": "get_default_embedder",
            "description": "Get the default embedder model for new palaces (global config)",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "set_default_embedder",
            "description": "Set the default embedder model for new palaces (global config ~/.kai-palace/config.json)",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Embedder name: sentence, spacy, or numpy",
                        "enum": ["sentence", "spacy", "numpy"],
                    },
                },
                "required": ["model"],
            },
        },
    ]
    _TOOL_DEFINITIONS.extend(extra_tools)
    return _TOOL_DEFINITIONS


class MCPServer:
    """JSON-RPC MCP server dispatching to Palace operations."""

    def __init__(self, palace: "Palace"):
        self.palace = palace
        self.entity_registry = EntityRegistry(palace)
        self.memory_stack = MemoryStack(palace) if HAS_LAYERS else None

        self._methods: dict[str, tuple] = {}
        self._register_all()

    def _register(self, name: str, handler: callable, required: list[str] | None = None):
        self._methods[name] = (handler, required or [])

    def _register_all(self):
        # MCP protocol methods
        self._register("initialize", self._mcp_initialize, [])
        self._register("notifications/initialized", self._mcp_noop, [])
        self._register("notifications/notified", self._mcp_noop, [])
        self._register("tools/list", self._mcp_tools_list, [])
        self._register("tools/call", self._mcp_tools_call, ["name", "arguments"])

        # Legacy JSON-RPC methods
        self._register("ping", self._ping, [])
        self._register("get_status", self._get_status, [])
        self._register("init_palace", self._init_palace, [])
        self._register("close_palace", self._close_palace, [])

        self._register("list_wings", self._list_wings, [])
        self._register("create_wing", self._create_wing, ["name"])
        self._register("delete_wing", self._delete_wing, ["name"])

        self._register("list_rooms", self._list_rooms, [])
        self._register("create_room", self._create_room, ["wing", "name"])
        self._register("delete_room", self._delete_room, ["wing", "name"])

        self._register("add_drawer", self._add_drawer, ["wing", "room", "content"])
        self._register("get_drawer", self._get_drawer, ["drawer_id"])
        self._register("list_drawers", self._list_drawers, [])
        self._register("update_drawer", self._update_drawer, ["drawer_id"])
        self._register("delete_drawer", self._delete_drawer, ["drawer_id"])

        self._register("search", self._search, ["query"])
        self._register("check_duplicate", self._check_duplicate, ["content"])

        self._register("kg_add", self._kg_add, ["subject", "predicate", "object"])
        self._register("kg_query", self._kg_query, [])
        self._register("kg_invalidate", self._kg_invalidate, ["subject", "predicate", "object"])
        self._register("kg_stats", self._kg_stats, [])
        self._register("kg_timeline", self._kg_timeline, [])

        self._register("diary_write", self._diary_write, ["agent", "entry"])
        self._register("diary_read", self._diary_read, ["agent"])
        self._register("memory_write", self._memory_write, ["agent", "layer", "content"])
        self._register("memory_read", self._memory_read, ["agent"])
        self._register("memory_summarize", self._memory_summarize, ["agent"])

        self._register("mine_file", self._mine_file, ["filepath", "wing", "room"])
        self._register("mine_text", self._mine_text, ["text", "wing", "room"])
        self._register("batch_mine", self._batch_mine, ["directory"])

        self._register("aaak_compress", self._aaak_compress, ["text"])
        self._register("aaak_decompress", self._aaak_decompress, ["text"])
        self._register("aaak_parse", self._aaak_parse, ["text"])

        self._register("rebuild_fts", self._rebuild_fts, [])

        # New tools ported from mempalace upstream
        self._register("create_tunnel", self._create_tunnel, ["source_wing", "source_room", "target_wing", "target_room"])
        self._register("delete_tunnel", self._delete_tunnel, ["tunnel_id"])
        self._register("find_tunnels", self._find_tunnels, [])
        self._register("follow_tunnels", self._follow_tunnels, ["wing", "room"])
        self._register("list_tunnels", self._list_tunnels, [])
        self._register("traverse", self._traverse, ["start_room"])
        self._register("graph_stats", self._graph_stats, [])
        self._register("get_taxonomy", self._get_taxonomy, [])
        self._register("get_aaak_spec", self._get_aaak_spec, [])
        self._register("memories_filed_away", self._memories_filed_away, [])
        self._register("sync", self._sync, [])
        self._register("reconnect", self._reconnect, [])
        self._register("hook_settings", self._hook_settings, [])
        self._register("set_embedder", self._set_embedder, ["model"])
        self._register("get_default_embedder", self._get_default_embedder, [])
        self._register("set_default_embedder", self._set_default_embedder, ["model"])

    def handle_request(self, raw: str) -> str | None:
        """Process one JSON-RPC request line, return JSON response string.
        Returns None for notifications (no ``id`` field).
        """
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error: invalid JSON"}})

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        if not isinstance(params, dict):
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": "Params must be a JSON object"}}
            return json.dumps(err)

        handler_info = self._methods.get(method)
        if handler_info is None:
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
            return json.dumps(err)

        handler, required = handler_info
        missing = [p for p in required if p not in params or params[p] is None or (isinstance(params[p], str) and not params[p].strip())]
        if missing:
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Missing required parameter(s): {', '.join(missing)}"}}
            return json.dumps(err)

        try:
            result = handler(params)
            if req_id is None:
                return None
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
        except Exception as e:
            logger.exception("Error handling %s: %s", method, e)
            err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}
            if logger.isEnabledFor(logging.DEBUG):
                err["error"]["data"] = traceback.format_exc()
            return json.dumps(err)

    # -- MCP protocol handlers ---------------------------------------------------

    def _mcp_initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "kai-palace", "version": VERSION},
        }

    def _mcp_noop(self, params: dict) -> dict:
        return {}

    def _mcp_tools_list(self, params: dict) -> dict:
        return {"tools": _build_tool_definitions()}

    def _mcp_tools_call(self, params: dict) -> dict:
        name = params["name"]
        args = params.get("arguments", {})
        handler_info = self._methods.get(name)
        if handler_info is None:
            raise ValueError(f"Unknown tool: {name}")
        handler, required = handler_info
        result = handler(args)
        text = json.dumps(result, indent=2, ensure_ascii=False) if not isinstance(result, str) else str(result)
        return {"content": [{"type": "text", "text": text}]}

    # -- Handler implementations ------------------------------------------------

    def _ping(self, params: dict) -> dict:
        return {"pong": True, "version": VERSION}

    def _get_status(self, params: dict) -> dict:
        return self.palace.status()

    def _init_palace(self, params: dict) -> dict:
        return {"created": self.palace.init()}

    def _close_palace(self, params: dict) -> dict:
        self.palace.close()
        return {"closed": True}

    def _list_wings(self, params: dict) -> list:
        return self.palace.list_wings()

    def _create_wing(self, params: dict) -> dict:
        name = self.palace.get_or_create_wing(params["name"], params.get("description", ""))
        return {"name": name}

    def _delete_wing(self, params: dict) -> dict:
        return {"deleted": self.palace.delete_wing(params["name"])}

    def _list_rooms(self, params: dict) -> list:
        return self.palace.list_rooms(wing=params.get("wing"))

    def _create_room(self, params: dict) -> dict:
        name = self.palace.get_or_create_room(params["wing"], params["name"], params.get("description", ""))
        return {"name": name}

    def _delete_room(self, params: dict) -> dict:
        return {"deleted": self.palace.delete_room(params["wing"], params["name"])}

    def _add_drawer(self, params: dict) -> dict:
        did = self.palace.add_drawer(
            params["wing"],
            params["room"],
            params["content"],
            metadata=params.get("metadata"),
            source_file=params.get("source_file", ""),
            drawer_id=params.get("drawer_id"),
        )
        return {"drawer_id": did}

    def _get_drawer(self, params: dict) -> dict | None:
        drawer = self.palace.get_drawer(params["drawer_id"])
        if drawer is None:
            return {"found": False}
        return drawer

    def _list_drawers(self, params: dict) -> list:
        return self.palace.list_drawers(
            wing=params.get("wing"),
            room=params.get("room"),
            limit=params.get("limit", 20),
            offset=params.get("offset", 0),
        )

    def _update_drawer(self, params: dict) -> dict:
        ok = self.palace.update_drawer(
            params["drawer_id"],
            content=params.get("content"),
            metadata=params.get("metadata"),
            wing=params.get("wing"),
            room=params.get("room"),
        )
        return {"updated": ok}

    def _delete_drawer(self, params: dict) -> dict:
        return {"deleted": self.palace.delete_drawer(params["drawer_id"])}

    def _search(self, params: dict) -> list:
        results = self.palace.search(
            params["query"],
            n_results=params.get("n_results", 10),
            wing=params.get("wing"),
            room=params.get("room"),
            mode=params.get("mode", "hybrid"),
        )
        return [
            {
                "id": r.id,
                "text": r.text,
                "distance": r.distance,
                "metadata": r.metadata,
                "wing": r.wing,
                "room": r.room,
            }
            for r in results
        ]

    def _check_duplicate(self, params: dict) -> dict:
        dup = self.palace.check_duplicate(params["content"], threshold=params.get("threshold", 0.9))
        if dup:
            return dup
        return {"duplicate": False}

    def _kg_add(self, params: dict) -> dict:
        fid = self.palace.kg.add(
            params["subject"],
            params["predicate"],
            params["object"],
            valid_from=params.get("valid_from"),
            valid_to=params.get("valid_to"),
            source=params.get("source", ""),
        )
        return {"fact_id": fid}

    def _kg_query(self, params: dict) -> list:
        if params.get("all"):
            return self.palace.kg.query(as_of=params.get("as_of"))
        return self.palace.kg.query(
            entity=params.get("entity"),
            predicate=params.get("predicate"),
            as_of=params.get("as_of"),
            direction=params.get("direction", "both"),
        )

    def _kg_invalidate(self, params: dict) -> dict:
        n = self.palace.kg.invalidate(
            params["subject"],
            params["predicate"],
            params["object"],
            ended=params.get("ended"),
        )
        return {"invalidated": n}

    def _kg_stats(self, params: dict) -> dict:
        return self.palace.kg.stats()

    def _diary_write(self, params: dict) -> dict:
        wing = self.palace.diary_write(
            params["agent"],
            params["entry"],
            topic=params.get("topic", "general"),
            wing=params.get("wing", ""),
        )
        return {"wing": wing}

    def _diary_read(self, params: dict) -> list:
        return self.palace.diary_read(
            params["agent"],
            last_n=params.get("last_n", 10),
            wing=params.get("wing", ""),
        )

    def _memory_write(self, params: dict) -> Any:
        self._require_layers()
        return self.memory_stack.write(
            params["agent"],
            params["layer"],
            params["content"],
            topic=params.get("topic"),
            metadata=params.get("metadata"),
        )

    def _memory_read(self, params: dict) -> list:
        self._require_layers()
        return self.memory_stack.read(
            params["agent"],
            layer=params.get("layer"),
            last_n=params.get("last_n", 10),
        )

    def _memory_summarize(self, params: dict) -> dict:
        self._require_layers()
        return self.memory_stack.summarize(params["agent"])

    def _mine_file(self, params: dict) -> dict:
        self._require_miner()
        count = miner.mine_file_into_palace(
            self.palace, params["filepath"], params["wing"], params["room"],
        )
        return {"items_mined": count}

    def _mine_text(self, params: dict) -> Any:
        self._require_miner()
        return miner.mine_text_into_palace(
            self.palace,
            params["text"],
            params["wing"],
            params["room"],
            source=params.get("source"),
            chunk=params.get("chunk", True),
        )

    def _batch_mine(self, params: dict) -> dict:
        self._require_miner()
        count = miner.batch_mine(
            self.palace,
            params["directory"],
            wing=params.get("wing"),
            pattern=params.get("pattern"),
        )
        return {"items_mined": count}

    def _aaak_compress(self, params: dict) -> str:
        return aaak_compress(params["text"], max_len=params.get("max_len", 500))

    def _aaak_decompress(self, params: dict) -> str:
        return aaak_decompress(params["text"])

    def _aaak_parse(self, params: dict) -> dict:
        return aaak_parse_entry(params["text"])

    def _rebuild_fts(self, params: dict) -> dict:
        self.palace.rebuild_fts()
        return {"rebuilt": True}

    def _kg_timeline(self, params: dict) -> list:
        return self.palace.kg.timeline(entity=params.get("entity"))

    def _create_tunnel(self, params: dict) -> dict:
        tunnel = palace_graph.create_tunnel(
            source_wing=params["source_wing"],
            source_room=params["source_room"],
            target_wing=params["target_wing"],
            target_room=params["target_room"],
            label=params.get("label", ""),
            source_drawer_id=params.get("source_drawer_id"),
            target_drawer_id=params.get("target_drawer_id"),
        )
        return tunnel

    def _delete_tunnel(self, params: dict) -> dict:
        return palace_graph.delete_tunnel(params["tunnel_id"])

    def _find_tunnels(self, params: dict) -> list:
        return palace_graph.find_tunnels(
            wing_a=params.get("wing_a"),
            wing_b=params.get("wing_b"),
        )

    def _follow_tunnels(self, params: dict) -> list:
        return palace_graph.follow_tunnels(
            params["wing"],
            params["room"],
            palace=self.palace,
        )

    def _list_tunnels(self, params: dict) -> list:
        return palace_graph.list_tunnels(wing=params.get("wing"))

    def _traverse(self, params: dict) -> list:
        result = palace_graph.traverse(
            start_room=params["start_room"],
            palace=self.palace,
            max_hops=params.get("max_hops", 2),
        )
        return result

    def _graph_stats(self, params: dict) -> dict:
        return palace_graph.graph_stats(palace=self.palace)

    def _get_taxonomy(self, params: dict) -> dict:
        return self.palace.get_taxonomy()

    def _get_aaak_spec(self, params: dict) -> str:
        return AAAK_SPEC

    def _memories_filed_away(self, params: dict) -> dict:
        return self.palace.memories_filed_away()

    def _sync(self, params: dict) -> dict:
        project_dirs = [params["project_dir"]] if params.get("project_dir") else None
        report = sync_palace(
            palace_path=self.palace._base,
            project_dirs=project_dirs,
            wing=params.get("wing"),
            dry_run=not params.get("apply", False),
        )
        return dict(report)

    def _reconnect(self, params: dict) -> dict:
        ok = self.palace.reconnect()
        return {"reconnected": ok}

    def _hook_settings(self, params: dict) -> dict:
        config = KaiPalaceConfig()
        if params:
            return config.set_hook_settings(
                silent_save=params.get("silent_save"),
                desktop_toast=params.get("desktop_toast"),
            )
        return config.get_hook_settings()

    def _set_embedder(self, params: dict) -> dict:
        return self.palace.set_embedder(
            model=params["model"],
            reindex=params.get("reindex", True),
        )

    def _get_default_embedder(self, params: dict) -> dict:
        from kai_mempalace.config import KaiPalaceConfig
        return {"default_embedder": KaiPalaceConfig().default_embedder}

    def _set_default_embedder(self, params: dict) -> dict:
        from kai_mempalace.config import KaiPalaceConfig
        model = KaiPalaceConfig().set_default_embedder(params["model"])
        return {"default_embedder": model}

    # -- Helpers ----------------------------------------------------------------

    def _require_layers(self):
        if not HAS_LAYERS or self.memory_stack is None:
            raise RuntimeError("MemoryStack not available (kai_mempalace.layers not installed)")

    def _require_miner(self):
        if not HAS_MINER:
            raise RuntimeError("Miner not available (kai_mempalace.miner not installed)")


# -- Transports ----------------------------------------------------------------


def _stdio_server(server: MCPServer) -> None:
    """Read JSON-RPC lines from stdin, write response lines to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        response = server.handle_request(line)
        if response is not None:
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if HAS_HTTP:

    class _MCPHTTPHandler(BaseHTTPRequestHandler):
        server_instance: MCPServer = None  # type: ignore[assignment]

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send_json(200, {"status": "ok", "version": VERSION})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/mcp":
                self.send_response(404)
                self.end_headers()
                return

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"

            response = self.server_instance.handle_request(body)
            if response is None:
                response = json.dumps({"jsonrpc": "2.0", "id": None, "result": None})
            self._send_json(200, response)

        def _send_json(self, status: int, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if isinstance(data, str):
                self.wfile.write(data.encode("utf-8"))
            else:
                self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self, fmt, *args):
            logger.debug("HTTP: " + fmt, *args)

    def _sse_server(server: MCPServer, host: str, port: int) -> None:
        _MCPHTTPHandler.server_instance = server
        httpd = HTTPServer((host, port), _MCPHTTPHandler)
        logger.info("MCP SSE server listening on http://%s:%d", host, port)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.shutdown()


def run_server(palace: "Palace", host: str = "127.0.0.1", port: int = 8316,
               transport: str = "stdio") -> None:
    """Run the MCP server.

    Parameters
    ----------
    palace : Palace
        Initialized Palace instance.
    host : str
        Bind address for SSE transport (default ``127.0.0.1``).
    port : int
        Port for SSE transport (default ``8316``).
    transport : str
        ``"stdio"`` (read/write JSON-RPC on stdin/stdout) or
        ``"sse"`` (HTTP server with ``GET /health`` and ``POST /mcp``).
    """
    server = MCPServer(palace)

    if transport == "stdio":
        _stdio_server(server)
    elif transport == "sse":
        if not HAS_HTTP:
            raise RuntimeError("http.server not available on this platform")
        _sse_server(server, host, port)
    else:
        raise ValueError(f"Unknown transport: {transport!r}")
