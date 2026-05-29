"""File mining pipeline for MemPalace — mines files into palace drawers with entity extraction, dedup, and compression."""

import ast
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from kai_mempalace.dialect import aaak_compress
from kai_mempalace.entity_detector import EntityDetector
from kai_mempalace.entity_registry import EntityRegistry
from kai_mempalace.palace import Palace, _ENTITY_STOPLIST

logger = logging.getLogger(__name__)

# ── Entity metadata extraction (for diary_ingest etc.) ──────────────────────

_ENTITY_EXTRACT_WINDOW = 1000
_ENTITY_METADATA_LIMIT = 50


def _load_known_entities(palace: Palace) -> dict[str, dict]:
    """Load persisted known_entities from the knowledge graph."""
    from kai_mempalace.backends.knowledge_graph import KnowledgeGraph

    kg_db = getattr(palace, "_kg_db", None)
    if kg_db:
        kg = KnowledgeGraph(kg_db)
        entities: dict[str, dict] = {}
        for row in kg_db.execute(
            "SELECT DISTINCT subject AS name FROM kg_facts"
        ).fetchall():
            name: str = row[0]
            if not name or name in _ENTITY_STOPLIST:
                continue
            entities[name] = {"name": name, "in_kg": True}
        return entities
    return {}


def _extract_entities_for_metadata(
    text: str, palace: Palace, limit: int = _ENTITY_METADATA_LIMIT
) -> list[dict]:
    """Extract named-entity metadata from text for drawer metadata.

    Returns a list of dicts, each with keys: ``name``, ``type``, ``count``.
    """
    from kai_mempalace.entity_detector import _get_coca_filter, _apply_known_systems_prepass
    from kai_mempalace.i18n import get_entity_patterns

    coca = _get_coca_filter()
    ep = get_entity_patterns()
    stopwords = frozenset(w.lower() for w in ep.get("stopwords", []))
    known = _load_known_entities(palace)

    cleaned, sys_counts = _apply_known_systems_prepass(text)

    counter: dict[str, int] = {}
    for name, count in sys_counts.items():
        if name not in _ENTITY_STOPLIST and name.lower() not in stopwords and name.lower() not in coca:
            counter[name] = counter.get(name, 0) + count

    for m in re.finditer(r"\b[A-Z][a-zA-Z'\-]{2,50}\b", cleaned):
        word = m.group()
        low = word.lower()
        if word in _ENTITY_STOPLIST:
            continue
        if low in stopwords:
            continue
        if low in coca:
            continue
        counter[word] = counter.get(word, 0) + 1

    sorted_entities = sorted(counter.items(), key=lambda x: -x[1])
    result: list[dict] = []
    for name, count in sorted_entities[:limit]:
        entry: dict[str, Any] = {"name": name, "type": "unknown", "count": count}
        if name in known:
            entry["in_kg"] = True
        result.append(entry)
    return result

# ── File type classifications ──────────────────────────────────────────────

_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".rst", ".log", ".cfg", ".ini", ".conf",
})

_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp", ".hxx", ".cxx", ".cc", ".hh",
    ".rb", ".php", ".swift", ".kt", ".scala", ".ex", ".exs",
    ".sh", ".bash", ".zsh", ".ps1", ".lua", ".pl", ".pm", ".r",
})

_STRUCTURED_EXTENSIONS = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
})

# ── Regex patterns ─────────────────────────────────────────────────────────

_CONVERSATION_LINE_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9_ ]+?):\s+(.+)$", re.MULTILINE,
)

_DOUBLE_NEWLINE_RE = re.compile(r"\n\s*\n")

_PYTHON_LINE_COMMENT_RE = re.compile(r"^\s*#.*$", re.MULTILINE)

_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/", re.MULTILINE)


def _classify_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext in _CODE_EXTENSIONS:
        return "code"
    if ext in _STRUCTURED_EXTENSIONS:
        return "structured"
    return "text"


def _get_source_path(filepath: str) -> str:
    return str(Path(filepath).resolve())


def _already_mined(palace: "Palace", filepath: str) -> bool:
    resolved = _get_source_path(filepath)
    drawers = palace.list_drawers(limit=10000)
    return any(d.get("source_file", "") == resolved for d in drawers)


# ── Code comment extraction ────────────────────────────────────────────────


def _extract_python_comments(source: str) -> list[dict[str, Any]]:
    results = []
    seen_positions: set[int] = set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("Failed to parse Python file, falling back to regex")
        return _extract_generic_comments(source, "#", False)

    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, (ast.Constant, ast.Str))):
        doc = tree.body[0].value
        val = doc.value if hasattr(doc, "value") else doc.s
        if isinstance(val, str) and val.strip():
            pos = getattr(doc, "lineno", 1)
            results.append({"content": val.strip(), "type": "docstring", "line": pos})
            seen_positions.add(pos)

    for node in ast.walk(tree):
        doc = ast.get_docstring(node)
        if doc and doc.strip():
            pos = getattr(node, "lineno", 1)
            if pos not in seen_positions:
                node_type = type(node).__name__.lower()
                name = getattr(node, "name", "")
                header = f"{node_type}: {name}" if name else node_type
                results.append({
                    "content": f"{header}\n{doc.strip()}",
                    "type": "docstring",
                    "line": pos,
                })
                seen_positions.add(pos)

    for m in _PYTHON_LINE_COMMENT_RE.finditer(source):
        comment = m.group(0).strip()
        if len(comment) <= 3:
            continue
        line_num = source[:m.start()].count("\n") + 1
        if line_num in seen_positions:
            continue
        text = comment.lstrip("#").strip()
        if text:
            results.append({
                "content": text,
                "type": "comment",
                "line": line_num,
            })
            seen_positions.add(line_num)

    return results


def _extract_generic_comments(
    source: str,
    line_comment: str = "//",
    has_block: bool = True,
) -> list[dict[str, Any]]:
    results = []
    seen: set[int] = set()

    if has_block:
        for m in _BLOCK_COMMENT_RE.finditer(source):
            content = m.group(0)
            content = re.sub(r"^/\*\s*", "", content)
            content = re.sub(r"\s*\*/$", "", content)
            content = content.strip().strip("*").strip()
            if len(content) <= 3:
                continue
            line_num = source[:m.start()].count("\n") + 1
            if line_num not in seen:
                results.append({
                    "content": content,
                    "type": "block_comment",
                    "line": line_num,
                })
                seen.add(line_num)

    if line_comment:
        escaped = re.escape(line_comment)
        pattern = re.compile(r"^\s*" + escaped + r"\s*(.*?)$", re.MULTILINE)
        for m in pattern.finditer(source):
            text = m.group(1).strip()
            if len(text) <= 3:
                continue
            line_num = source[:m.start()].count("\n") + 1
            if line_num not in seen:
                results.append({
                    "content": text,
                    "type": "line_comment",
                    "line": line_num,
                })
                seen.add(line_num)

    return results


def _extract_code_comments(filepath: str, source: str) -> list[str]:
    ext = Path(filepath).suffix.lower()

    if ext == ".py":
        items = _extract_python_comments(source)
    elif ext in _CODE_EXTENSIONS - {".py"}:
        if ext in (".rb", ".pl", ".pm", ".sh", ".bash", ".zsh", ".ps1", ".r"):
            items = _extract_generic_comments(source, "#", has_block=False)
        elif ext in (".lua", ".ex", ".exs", "--"):
            items = _extract_generic_comments(source, "--", has_block=False)
        else:
            items = _extract_generic_comments(source, "//")
    else:
        return []

    return [item["content"] for item in items if item["content"].strip()]


# ── Text chunking ──────────────────────────────────────────────────────────


def _chunk_text(content: str, min_size: int = 50, max_size: int = 2000) -> list[str]:
    raw_chunks = _DOUBLE_NEWLINE_RE.split(content.strip())
    result = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        if len(chunk) < min_size:
            if result:
                result[-1] = result[-1] + "\n\n" + chunk
            else:
                result.append(chunk)
        elif len(chunk) > max_size:
            lines = chunk.split("\n")
            buf: list[str] = []
            buf_len = 0
            for line in lines:
                buf.append(line)
                buf_len += len(line) + 1
                if buf_len >= max_size:
                    result.append("\n".join(buf).strip())
                    buf = []
                    buf_len = 0
            if buf:
                result.append("\n".join(buf).strip())
        else:
            result.append(chunk)

    return result


# ── Internal helpers ───────────────────────────────────────────────────────


def _add_chunk(
    palace: "Palace",
    content: str,
    wing: str,
    room: str,
    source_file: str = "",
    metadata: Optional[dict] = None,
    compress: bool = False,
) -> Optional[str]:
    if len(content.strip()) < 10:
        return None

    dup = palace.check_duplicate(content, threshold=0.85)
    if dup:
        return None

    final_content = aaak_compress(content) if compress else content

    try:
        drawer_id = palace.add_drawer(
            wing=wing,
            room=room,
            content=final_content,
            metadata=metadata or {},
            source_file=source_file,
        )
    except (ValueError, RuntimeError):
        return None

    return drawer_id


# ── Public API ─────────────────────────────────────────────────────────────


def mine_file_into_palace(
    palace: "Palace",
    filepath: str,
    wing: str,
    room: str,
    min_chunk_size: int = 50,
    max_chunk_size: int = 2000,
) -> int:
    resolved = _get_source_path(filepath)

    if not os.path.isfile(resolved):
        logger.warning("File not found: %s", resolved)
        return 0

    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception as e:
        logger.error("Failed to read %s: %s", resolved, e)
        return 0

    if not content.strip():
        return 0

    file_type = _classify_file(resolved)
    metadata: dict[str, Any] = {"file_type": file_type, "source": resolved}

    if file_type == "code":
        chunks = _extract_code_comments(resolved, content)
    elif file_type == "structured":
        try:
            if resolved.endswith((".json",)):
                data = json.loads(content)
                pretty = json.dumps(data, indent=2, ensure_ascii=False)
                chunks = [pretty]
            else:
                chunks = [content.strip()]
        except (json.JSONDecodeError, ValueError):
            chunks = [content.strip()]
    else:
        chunks = _chunk_text(content, min_chunk_size, max_chunk_size)

    registry = EntityRegistry(palace)
    created = 0

    for chunk in chunks:
        if not chunk.strip():
            continue

        registry.register(chunk, source=resolved)

        drawer_id = _add_chunk(
            palace,
            chunk,
            wing=wing,
            room=room,
            source_file=resolved,
            metadata=metadata,
        )
        if drawer_id:
            created += 1

    logger.info("Mined %s -> %d/%d drawers in %s/%s",
                resolved, created, len(chunks), wing, room)
    return created


def mine_text_into_palace(
    palace: "Palace",
    text: str,
    wing: str,
    room: str,
    source: str = "",
    chunk: bool = True,
) -> int:
    if not text.strip():
        return 0

    chunks: list[str]
    if chunk:
        chunks = _chunk_text(text)
    else:
        chunks = [text.strip()]

    registry = EntityRegistry(palace)
    created = 0

    for c in chunks:
        if not c.strip():
            continue
        registry.register(c, source=source)
        drawer_id = _add_chunk(
            palace, c, wing=wing, room=room, source_file=source,
        )
        if drawer_id:
            created += 1

    return created


def mine_conversation(
    palace: "Palace",
    log_text: str,
    wing: str = "conversations",
    source: str = "",
) -> int:
    if not log_text.strip():
        return 0

    registry = EntityRegistry(palace)
    created = 0

    for m in _CONVERSATION_LINE_RE.finditer(log_text):
        speaker = m.group(1).strip()
        message = m.group(2).strip()
        if not message:
            continue

        registry.register(speaker, source=source)

        content = f"{speaker}: {message}"
        metadata = {"speaker": speaker, "type": "conversation"}

        registry.register(content, source=source)

        drawer_id = _add_chunk(
            palace,
            content,
            wing=wing,
            room=speaker.lower().replace(" ", "_"),
            source_file=source,
            metadata=metadata,
        )
        if drawer_id:
            created += 1

    return created


def batch_mine(
    palace: "Palace",
    directory: str,
    wing: str = "files",
    pattern: str = "*.txt,*.md,*.py,*.json,*.yaml,*.yml,*.cfg,*.ini,*.log",
    recursive: bool = True,
) -> dict:
    base = Path(directory).resolve()
    if not base.is_dir():
        return {
            "wing": wing,
            "files_processed": 0,
            "drawers_created": 0,
            "errors": [f"Directory not found: {directory}"],
        }

    patterns = [p.strip() for p in pattern.split(",") if p.strip()]
    files: list[Path] = []
    for p in patterns:
        if recursive:
            files.extend(base.rglob(p))
        else:
            files.extend(base.glob(p))

    seen: set[str] = set()
    unique: list[Path] = []
    for f in files:
        resolved = str(f.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)

    total_created = 0

    for fpath in unique:
        try:
            room = fpath.parent.relative_to(base).as_posix()
            if room == ".":
                room = "root"
        except ValueError:
            room = "root"

        n = mine_file_into_palace(palace, str(fpath), wing=wing, room=room)
        total_created += n

    return {
        "wing": wing,
        "files_processed": len(unique),
        "drawers_created": total_created,
        "errors": [],
    }


def mine_code_file(palace: "Palace", filepath: str, wing: str = "code") -> int:
    resolved = _get_source_path(filepath)
    ext = Path(resolved).suffix.lower()

    if ext not in _CODE_EXTENSIONS:
        logger.warning("Not a recognized code file: %s", resolved)
        return 0

    return mine_file_into_palace(palace, resolved, wing=wing, room=ext.lstrip("."))


# ── Object-oriented API ────────────────────────────────────────────────────


class FileMiner:
    """Object-oriented file miner with progress tracking."""

    def __init__(self, palace: "Palace"):
        self.palace = palace
        self._entity_registry = EntityRegistry(palace)
        self._stats: dict[str, int] = {
            "processed": 0,
            "created": 0,
            "skipped_dup": 0,
            "errors": 0,
        }

    def mine(self, filepath: str, wing: str, room: str) -> int:
        self._stats["processed"] += 1
        try:
            n = mine_file_into_palace(self.palace, filepath, wing, room)
            if n > 0:
                self._stats["created"] += n
            else:
                self._stats["skipped_dup"] += 1
            return n
        except Exception as e:
            logger.exception("Error mining %s: %s", filepath, e)
            self._stats["errors"] += 1
            return 0

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def reset_stats(self) -> None:
        self._stats = {
            "processed": 0,
            "created": 0,
            "skipped_dup": 0,
            "errors": 0,
        }
