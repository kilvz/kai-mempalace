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

VERSION = "3.3.6"


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

    def _kg_timeline(self, params: dict) -> list:
        return self.palace.kg.timeline(entity=params.get("entity"))

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
