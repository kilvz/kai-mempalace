"""Mine binary office-format documents (PDF, DOCX, PPTX, XLSX, RTF, EPUB) into the palace.

Requires optional dependencies:
  - ``markitdown`` for .pdf, .docx, .pptx, .xlsx, .epub
  - ``striprtf`` for .rtf
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from kai_mempalace.config import MempalaceConfig
from kai_mempalace.miner import _chunk_text, _extract_entities_for_metadata
from kai_mempalace.palace import file_already_mined, get_closets_collection, mine_lock

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".rtf", ".epub"})
DEFAULT_MAX_FILE_SIZE = 500 * 1024 * 1024
DRAWER_UPSERT_BATCH_SIZE = 1000

_SKIP_FILENAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
_ENCRYPTED_PATTERNS = re.compile(r"(encrypt|decrypt|password|protected)", re.IGNORECASE)
_SKIP_DIRS = frozenset({".git", "__pycache__", "node_modules", ".venv", "venv", ".opencode", ".claude"})


class ExtractionStatus(enum.Enum):
    OK = "ok"
    SKIP_TOO_LARGE = "skip:too_large"
    SKIP_CLOUD_ONLY = "skip:cloud_only"
    SKIP_EMPTY = "skip:empty"
    SKIP_NO_MARKITDOWN = "skip:no_markitdown"
    SKIP_NO_STRIPRTF = "skip:no_striprtf"
    SKIP_ENCRYPTED = "skip:encrypted"
    SKIP_PERMISSION = "skip:permission"
    SKIP_BROKEN_SYMLINK = "skip:broken_symlink"
    SKIP_UNRECOGNIZED = "skip:unrecognized"
    SKIP_EXTRACTION_ERROR = "skip:extraction_error"
    SKIP_MISSING_FORMAT_DEPS = "skip:missing_format_deps"
    SKIP_NETWORK_TIMEOUT = "skip:network_timeout"
    SKIP_UNREADABLE = "skip:unreadable"


_TRANSIENT_MISSING_DEP_STATUSES = frozenset({
    ExtractionStatus.SKIP_NO_MARKITDOWN,
    ExtractionStatus.SKIP_NO_STRIPRTF,
    ExtractionStatus.SKIP_MISSING_FORMAT_DEPS,
    ExtractionStatus.SKIP_NETWORK_TIMEOUT,
})


def decode_robust(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        pass
    return raw.decode("utf-8", errors="replace")


def is_icloud_dataless(path: Path) -> bool:
    if path.suffix.lower() == ".icloud":
        return True
    try:
        flags = getattr(path.lstat(), "st_flags", 0)
    except OSError:
        return False
    return bool(flags & 0x40000000)


def _extract_via_markitdown(path: Path) -> Optional[str]:
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise
    converter = MarkItDown()
    result = converter.convert(str(path))
    text = getattr(result, "text_content", None) or getattr(result, "markdown", None)
    if text is None or not isinstance(text, str):
        return None
    return text


def _extract_via_striprtf(path: Path) -> Optional[str]:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        raise
    raw = path.read_bytes()
    source = decode_robust(raw)
    text = rtf_to_text(source)
    if not isinstance(text, str) or text == "":
        return None
    return text


def extract_text(
    path: Union[Path, str],
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> tuple[Optional[str], ExtractionStatus]:
    p = Path(path).expanduser()

    if p.is_symlink() and not p.exists():
        logger.info("skip:broken_symlink %s", p)
        return None, ExtractionStatus.SKIP_BROKEN_SYMLINK

    if is_icloud_dataless(p):
        logger.info("skip:cloud_only %s", p)
        return None, ExtractionStatus.SKIP_CLOUD_ONLY

    try:
        stat = p.stat()
    except PermissionError:
        logger.info("skip:permission (stat) %s", p)
        return None, ExtractionStatus.SKIP_PERMISSION
    except FileNotFoundError:
        if p.is_symlink():
            return None, ExtractionStatus.SKIP_BROKEN_SYMLINK
        logger.info("skip:unreadable (file gone) %s", p)
        return None, ExtractionStatus.SKIP_UNREADABLE
    except OSError as exc:
        logger.info("skip:unreadable %s — %s", p, exc)
        return None, ExtractionStatus.SKIP_UNREADABLE

    if stat.st_size == 0:
        logger.debug("skip:empty %s", p)
        return None, ExtractionStatus.SKIP_EMPTY

    if stat.st_size > max_file_size:
        logger.info("skip:too_large %s (%d bytes > %d)", p, stat.st_size, max_file_size)
        return None, ExtractionStatus.SKIP_TOO_LARGE

    if p.suffix.lower() not in SUPPORTED_FORMATS:
        logger.debug("skip:unrecognized %s (suffix=%s)", p, p.suffix)
        return None, ExtractionStatus.SKIP_UNRECOGNIZED

    is_rtf = p.suffix.lower() == ".rtf"
    try:
        if is_rtf:
            text = _extract_via_striprtf(p)
        else:
            text = _extract_via_markitdown(p)
    except ImportError:
        if is_rtf:
            logger.warning("skip:no_striprtf %s — install with: pip install striprtf", p)
            return None, ExtractionStatus.SKIP_NO_STRIPRTF
        logger.warning("skip:no_markitdown %s — install with: pip install markitdown", p)
        return None, ExtractionStatus.SKIP_NO_MARKITDOWN
    except TimeoutError:
        logger.info("skip:network_timeout %s", p)
        return None, ExtractionStatus.SKIP_NETWORK_TIMEOUT
    except PermissionError:
        logger.info("skip:permission %s", p)
        return None, ExtractionStatus.SKIP_PERMISSION
    except Exception as exc:
        if type(exc).__name__ == "MissingDependencyException":
            logger.warning("skip:missing_format_deps %s — %s", p, str(exc)[:200])
            return None, ExtractionStatus.SKIP_MISSING_FORMAT_DEPS
        msg = str(exc)
        if _ENCRYPTED_PATTERNS.search(msg):
            logger.info("skip:encrypted %s — %s", p, msg[:120])
            return None, ExtractionStatus.SKIP_ENCRYPTED
        logger.warning("skip:extraction_error %s — %s: %s", p, type(exc).__name__, msg[:200])
        return None, ExtractionStatus.SKIP_EXTRACTION_ERROR

    if not text:
        transformer = "striprtf" if is_rtf else "markitdown"
        logger.info("skip:extraction_error %s — %s returned None/empty", p, transformer)
        return None, ExtractionStatus.SKIP_EXTRACTION_ERROR

    return text, ExtractionStatus.OK


def scan_formats(directory: Union[Path, str]) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        return []

    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name in _SKIP_FILENAMES:
                continue
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            if p.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            found.append(p)

    found.sort()
    return found


def _register_format_sentinel(
    closets_col, source_file: str, wing: str, agent: str
) -> None:
    sentinel_id = f"sentinel_format_{hashlib.sha256(source_file.encode()).hexdigest()[:24]}"
    try:
        closets_col._conn.execute(
            "INSERT OR REPLACE INTO closets (id, content, metadata, source_file, wing, room) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                sentinel_id,
                "[empty]",
                json.dumps({
                    "wing": wing,
                    "room": "documents",
                    "source_file": source_file,
                    "added_by": agent,
                    "filed_at": datetime.now().isoformat(),
                    "ingest_mode": "extract",
                    "extract_mode": "format",
                    "is_sentinel": True,
                }),
                source_file,
                wing,
                "documents",
            ),
        )
        closets_col._conn.commit()
    except Exception:
        logger.debug("Sentinel write failed for %s", source_file, exc_info=True)


def _register_skip_sentinel_if_appropriate(
    closets_col, source_file: str, wing: str, agent: str, status: ExtractionStatus
) -> None:
    if status in _TRANSIENT_MISSING_DEP_STATUSES:
        return
    _register_format_sentinel(closets_col, source_file, wing, agent)


def _format_drawer_id(wing: str, room: str, source_file: str, chunk_index: int) -> str:
    key = f"{wing}|{room}|{source_file}|{chunk_index}"
    return f"drawer_format_{hashlib.sha256(key.encode()).hexdigest()[:24]}"


def mine_formats(
    format_dir: str,
    palace_path: str,
    wing: Optional[str] = None,
    agent: str = "mempalace",
    limit: int = 0,
    dry_run: bool = False,
) -> dict[str, int]:
    palace_config = MempalaceConfig()

    format_path = Path(format_dir).expanduser().resolve()
    if not wing:
        wing = format_path.name.lower().replace(" ", "_").replace("-", "_")

    files: list[Path] = []
    closets_col = None
    total_drawers = 0
    files_skipped = 0
    files_with_text = 0
    files_errored = 0
    status_counts: dict[str, int] = defaultdict(int)

    try:
        files = scan_formats(format_path)
        if limit > 0:
            files = files[:limit]

        print(f"\n{'=' * 55}")
        print("  MemPalace Mine — Format extraction")
        print(f"{'=' * 55}")
        print(f"  Wing:    {wing}")
        print(f"  Source:  {format_path}")
        print(f"  Files:   {len(files)}")
        print(f"  Palace:  {palace_path}")
        if dry_run:
            print("  DRY RUN — nothing will be filed")
        print(f"{'-' * 55}\n")

        closets_col = get_closets_collection(palace_path) if not dry_run else None

        for i, filepath in enumerate(files, 1):
            source_file = str(filepath)
            try:
                try:
                    source_mtime = os.path.getmtime(source_file)
                except OSError:
                    source_mtime = None

                if not dry_run and file_already_mined(
                    closets_col._conn, source_file, check_mtime=True, extract_mode="format"
                ):
                    files_skipped += 1
                    continue

                text, status = extract_text(filepath)
                status_counts[status.name] += 1

                if status != ExtractionStatus.OK or not text:
                    if not dry_run:
                        _register_skip_sentinel_if_appropriate(closets_col, source_file, wing, agent, status)
                    print(f"  - [{i:4}/{len(files)}] {filepath.name[:50]:50} {status.name}")
                    continue

                raw_chunks = _chunk_text(
                    text,
                    min_size=palace_config.min_chunk_size,
                    max_size=palace_config.chunk_size,
                )
                chunks = [
                    {"content": c, "chunk_index": i, "source_file": source_file}
                    for i, c in enumerate(raw_chunks)
                ]
                if not chunks:
                    if not dry_run:
                        _register_format_sentinel(closets_col, source_file, wing, agent)
                    print(f"  - [{i:4}/{len(files)}] {filepath.name[:50]:50} EMPTY_AFTER_CHUNK")
                    continue

                room = filepath.parent.name.lower().replace(" ", "_") if filepath.parent.name else "documents"
                files_with_text += 1

                if dry_run:
                    print(f"    [DRY RUN] {filepath.name} → {len(chunks)} drawers")
                    total_drawers += len(chunks)
                    continue

                drawers_added = 0
                with mine_lock(source_file):
                    if file_already_mined(
                        closets_col._conn, source_file, check_mtime=True, extract_mode="format"
                    ):
                        files_skipped += 1
                        continue

                    for batch_start in range(0, len(chunks), DRAWER_UPSERT_BATCH_SIZE):
                        batch = chunks[batch_start:batch_start + DRAWER_UPSERT_BATCH_SIZE]
                        for chunk in batch:
                            drawer_id = _format_drawer_id(wing, room, source_file, chunk["chunk_index"])
                            content = chunk["content"]
                            entities = _extract_entities_for_metadata(content)
                            meta = {
                                "wing": wing,
                                "room": room,
                                "source_file": source_file,
                                "chunk_index": chunk["chunk_index"],
                                "added_by": agent,
                                "filed_at": datetime.now().isoformat(),
                                "ingest_mode": "extract",
                                "extract_mode": "format",
                            }
                            if source_mtime is not None:
                                meta["source_mtime"] = source_mtime
                            if entities:
                                meta["entities"] = entities
                            closets_col._conn.execute(
                                "INSERT OR REPLACE INTO closets (id, content, metadata, source_file, wing, room) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    drawer_id,
                                    content,
                                    json.dumps(meta),
                                    source_file,
                                    wing,
                                    room,
                                ),
                            )
                        closets_col._conn.commit()
                        drawers_added += len(batch)

                total_drawers += drawers_added
                print(f"  + [{i:4}/{len(files)}] {filepath.name[:50]:50} +{drawers_added}")

            except Exception as exc:
                files_errored += 1
                logger.warning("mine_formats: error processing %s — %s: %s", source_file, type(exc).__name__, str(exc)[:200])
                print(f"  ! [{i:4}/{len(files)}] {filepath.name[:50]:50} ERROR: {type(exc).__name__}")
                continue

    except KeyboardInterrupt:
        print("\n  Mine interrupted by user (Ctrl-C).")
    except Exception as exc:
        logger.warning("mine_formats: unexpected exception — %s: %s", type(exc).__name__, str(exc)[:200])
        print(f"\n  Mine aborted ({type(exc).__name__}: {str(exc)[:120]})", file=sys.stderr)

    print(f"\n{'=' * 55}")
    print("  Summary")
    print(f"{'-' * 55}")
    print(f"  Files seen:        {len(files)}")
    print(f"  Files extracted:   {files_with_text}")
    print(f"  Files skipped:     {files_skipped}")
    print(f"  Files errored:     {files_errored}")
    print(f"  Total drawers:     {total_drawers}")
    if status_counts:
        print("  Extraction status:")
        for name, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {name:30} {count}")
    print(f"{'=' * 55}\n")

    return {"files": len(files), "drawers": total_drawers, "errors": files_errored}
