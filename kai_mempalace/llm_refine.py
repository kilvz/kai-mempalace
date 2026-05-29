"""
llm_refine.py — Optional LLM refinement of regex-detected entities.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass

from kai_mempalace.llm_client import LLMError, LLMProvider


BATCH_SIZE = 25
CONTEXT_LINES_PER_CANDIDATE = 3
CONTEXT_WINDOW_CHARS = 240

VALID_LABELS = {"PERSON", "PROJECT", "TOPIC", "COMMON_WORD", "AMBIGUOUS"}


SYSTEM_PROMPT = """You are helping organize a user's memory palace by classifying capitalized tokens found in their files.

For each candidate, pick exactly ONE label:
- PERSON: a specific real person the user knows (colleague, family, character they write about)
- PROJECT: a named product, codebase, or effort the user works on
- TOPIC: a recurring theme or subject (not a person, not a project) — cities, technologies, concepts
- COMMON_WORD: an English word, verb, or fragment that isn't a named entity at all (e.g. "Created", "Before", "Never")
- AMBIGUOUS: context is insufficient to decide between two of the above

Frameworks, runtimes, APIs, cloud services, vendors, and third-party products
(e.g. Angular, OpenAPI, Terraform, Bun, Google) are TOPIC unless the context
clearly says this is the user's own named codebase, product, or active effort.

Use the provided context lines to disambiguate. A capitalized word that only appears in metadata ("Created: 2026-04-24") is COMMON_WORD. A name that appears with pronouns and dialogue is PERSON.

Respond with JSON only. Schema:
{"classifications": [{"name": "<exact candidate name>", "label": "<LABEL>", "reason": "<one short sentence>"}]}

One entry per candidate, same order as the input."""


@dataclass
class RefineResult:
    merged: dict
    reclassified: int
    dropped: int
    errors: list[str]
    batches_completed: int
    batches_total: int
    cancelled: bool


def _collect_contexts(
    corpus_lines: list[str], name: str, max_lines: int = CONTEXT_LINES_PER_CANDIDATE
) -> list[str]:
    needle = re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
    seen: set[str] = set()
    out: list[str] = []
    for line in corpus_lines:
        if not needle.search(line):
            continue
        trimmed = line.strip()[:CONTEXT_WINDOW_CHARS]
        if not trimmed or trimmed in seen:
            continue
        seen.add(trimmed)
        out.append(trimmed)
        if len(out) >= max_lines:
            break
    return out


def _build_user_prompt(candidates_with_contexts: list[tuple[str, str, list[str]]]) -> str:
    parts: list[str] = ["CANDIDATES:"]
    for i, (name, current_type, contexts) in enumerate(candidates_with_contexts, 1):
        parts.append(f"\n{i}. {name}  (currently: {current_type})")
        if contexts:
            for c in contexts:
                parts.append(f"   > {c}")
        else:
            parts.append("   > (no context available)")
    return "\n".join(parts)


def _extract_json_candidates(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    candidates: list[str] = [text]
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE):
        candidate = match.group(1).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for start, opener in ((i, ch) for i, ch in enumerate(text) if ch in "{["):
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1].strip()
                    if candidate and candidate not in candidates:
                        candidates.append(candidate)
                    break
    return candidates


def _parse_response(text: str, expected_names: list[str]) -> dict[str, tuple[str, str]]:
    data = None
    for candidate in _extract_json_candidates(text):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return {}
    entries = data.get("classifications") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return {}
    name_to_label: dict[str, tuple[str, str]] = {}
    expected_set = {n.lower(): n for n in expected_names}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("candidate")
        label = entry.get("label") or entry.get("type") or entry.get("classification")
        reason = entry.get("reason") or ""
        if not isinstance(name, str) or not isinstance(label, str):
            continue
        canonical = expected_set.get(name.lower(), name)
        lbl = label.strip().upper()
        if lbl not in VALID_LABELS:
            lbl = "AMBIGUOUS"
        name_to_label[canonical] = (lbl, reason.strip()[:120])
    return name_to_label


def _apply_classifications(
    detected: dict,
    decisions: dict[str, tuple[str, str]],
    allow_project_promotions: bool = True,
) -> tuple[dict, int, int]:
    label_to_bucket = {
        "PERSON": "people",
        "PROJECT": "projects",
        "TOPIC": "topics",
        "AMBIGUOUS": "uncertain",
    }
    bucket_to_type = {
        "people": "person",
        "projects": "project",
        "topics": "topic",
        "uncertain": "uncertain",
    }
    all_entries: list[tuple[str, dict]] = []
    for bucket, items in detected.items():
        for e in items:
            all_entries.append((bucket, e))
    reclassified = 0
    dropped = 0
    new_detected: dict[str, list[dict]] = {
        "people": [],
        "projects": [],
        "topics": [],
        "uncertain": [],
    }
    for old_bucket, entry in all_entries:
        decision = decisions.get(entry["name"])
        if decision is None:
            new_detected.setdefault(old_bucket, []).append(entry)
            continue
        label, reason = decision
        if label == "COMMON_WORD":
            dropped += 1
            continue
        target_bucket = label_to_bucket[label]
        if (
            label == "PROJECT"
            and not allow_project_promotions
            and not _is_authoritative_project(entry)
        ):
            target_bucket = "uncertain"
        updated = dict(entry)
        signals = list(updated.get("signals", []))
        signals.append(f"LLM: {label.lower()} — {reason}" if reason else f"LLM: {label.lower()}")
        updated["signals"] = signals
        if target_bucket != old_bucket:
            reclassified += 1
            updated["type"] = bucket_to_type.get(target_bucket, "uncertain")
        new_detected[target_bucket].append(updated)
    return new_detected, reclassified, dropped


def _build_corpus_origin_preamble(corpus_origin: dict | None) -> str:
    if not corpus_origin:
        return ""
    result = corpus_origin.get("result") or {}
    if not result.get("likely_ai_dialogue"):
        return ""
    lines = ["\n\nCORPUS CONTEXT (corpus-origin detection):"]
    platform = result.get("primary_platform")
    if platform:
        lines.append(f"- This corpus is AI-dialogue from {platform}.")
    user_name = result.get("user_name")
    if user_name:
        lines.append(
            f"- The corpus author (the human user) is named '{user_name}'. "
            f"Treat this name as PERSON."
        )
    personas = result.get("agent_persona_names") or []
    if personas:
        lines.append(
            "- The user has assigned these persona names to AI agents in "
            f"this corpus: {', '.join(personas)}."
        )
        lines.append(
            "- Persona names refer to AI agents, not biological people. "
            "Classify them as PERSON (a downstream step tags them as "
            "agent personas)."
        )
    return "\n".join(lines)


def _is_authoritative_person(entry: dict) -> bool:
    signals = " ".join(entry.get("signals", [])).lower()
    return "commit" in signals and "repo" in signals


def _is_authoritative_project(entry: dict) -> bool:
    signals = " ".join(entry.get("signals", [])).lower()
    manifest_markers = ("package.json", "pyproject.toml", "cargo.toml", "go.mod")
    return any(marker in signals for marker in manifest_markers) or "commit" in signals


def _print_progress(batch_idx: int, total: int, current_name: str) -> None:
    width = 40
    filled = int(width * batch_idx / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    msg = f"\r  LLM refine: [{bar}] batch {batch_idx}/{total}  current: {current_name[:30]:<30}"
    sys.stderr.write(msg)
    sys.stderr.flush()


def refine_entities(
    detected: dict,
    corpus_text: str,
    provider: LLMProvider,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
    allow_project_promotions: bool = True,
    corpus_origin: dict | None = None,
) -> RefineResult:
    candidates: list[tuple[str, str]] = []
    current_type = {"people": "person", "projects": "project", "uncertain": "uncertain"}
    for bucket in ("people", "projects", "uncertain"):
        for e in detected.get(bucket, []):
            if bucket == "people" and _is_authoritative_person(e):
                continue
            if bucket == "projects" and _is_authoritative_project(e):
                continue
            candidates.append((e["name"], current_type[bucket]))
    corpus_lines = corpus_text.splitlines() if corpus_text else []
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name, kind in candidates:
        if name not in seen:
            seen.add(name)
            unique.append((name, kind))
    if not unique:
        return RefineResult(
            merged=detected,
            reclassified=0,
            dropped=0,
            errors=[],
            batches_completed=0,
            batches_total=0,
            cancelled=False,
        )
    batches: list[list[tuple[str, str, list[str]]]] = []
    for i in range(0, len(unique), batch_size):
        chunk = unique[i : i + batch_size]
        enriched = [(name, kind, _collect_contexts(corpus_lines, name)) for name, kind in chunk]
        batches.append(enriched)
    all_decisions: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    completed = 0
    cancelled = False
    system_prompt = SYSTEM_PROMPT + _build_corpus_origin_preamble(corpus_origin)
    for idx, batch in enumerate(batches, 1):
        if show_progress and batch:
            _print_progress(idx - 1, len(batches), batch[0][0])
        user_prompt = _build_user_prompt(batch)
        try:
            resp = provider.classify(system_prompt, user_prompt, json_mode=True)
        except KeyboardInterrupt:
            cancelled = True
            break
        except LLMError as e:
            errors.append(f"batch {idx}: {e}")
            continue
        names_in_batch = [name for name, _, _ in batch]
        decisions = _parse_response(resp.text, names_in_batch)
        if not decisions:
            errors.append(f"batch {idx}: could not parse response")
        all_decisions.update(decisions)
        completed += 1
        if show_progress:
            _print_progress(idx, len(batches), batch[-1][0])
    if show_progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    merged, reclassified, dropped = _apply_classifications(
        detected,
        all_decisions,
        allow_project_promotions=allow_project_promotions,
    )
    return RefineResult(
        merged=merged,
        reclassified=reclassified,
        dropped=dropped,
        errors=errors,
        batches_completed=completed,
        batches_total=len(batches),
        cancelled=cancelled,
    )


def collect_corpus_text(
    project_dir: str,
    max_files: int = 30,
    max_bytes_per_file: int = 20_000,
) -> str:
    from pathlib import Path

    from kai_mempalace.entity_detector import PROSE_EXTENSIONS, SKIP_DIRS

    root = Path(project_dir).expanduser().resolve()
    if not root.is_dir():
        return ""
    candidates: list[tuple[float, Path]] = []
    for dirpath, dirs, files in _walk_prose(root, SKIP_DIRS):
        for fname in files:
            p = dirpath / fname
            if p.suffix.lower() not in PROSE_EXTENSIONS:
                continue
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            candidates.append((mtime, p))
    candidates.sort(reverse=True)
    selected = [p for _, p in candidates[:max_files]]
    chunks: list[str] = []
    for p in selected:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                chunks.append(f.read(max_bytes_per_file))
        except OSError:
            continue
    return "\n".join(chunks)


def _walk_prose(root, skip_dirs):
    import os
    from pathlib import Path

    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        yield Path(dirpath), dirs, files
