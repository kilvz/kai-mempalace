"""
corpus_origin.py — Detect whether a corpus is an AI-dialogue record.

Two-tier detection:
  Tier 1 — cheap heuristic (regex, no API)
  Tier 2 — LLM-assisted (via LLMProvider)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


_AI_UNAMBIGUOUS_TERMS = [
    "Anthropic",
    "Claude Code",
    "Claude 3",
    "Claude 4",
    "claude mcp",
    "CLAUDE.md",
    ".claude/",
    "ChatGPT",
    "GPT-4",
    "GPT-3",
    "GPT-5",
    "OpenAI",
    "gpt-4o",
    "gpt-4-turbo",
    "o1-preview",
    "o3",
    "gemini-pro",
    "gemini-1.5",
    "Google AI",
    "Mixtral",
    "Cohere",
    "MCP",
    "LLM",
    "RAG",
    "fine-tune",
    "context window",
    "embedding",
]

_AI_AMBIGUOUS_TERMS = [
    "Claude",
    "Opus",
    "Sonnet",
    "Haiku",
    "Gemini",
    "Bard",
    "Llama",
    "Mistral",
]

_TURN_MARKERS = [
    r"\buser\s*:\s*",
    r"\bassistant\s*:\s*",
    r"\bhuman\s*:\s*",
    r"\bai\s*:\s*",
    r"\b>>>\s*User\b",
    r"\b>>>\s*Assistant\b",
]


def _brand_pattern(term: str) -> str:
    escaped = re.escape(term)
    prefix = r"\b" if term[0].isalnum() or term[0] == "_" else ""
    suffix = r"\b" if term[-1].isalnum() or term[-1] == "_" else ""
    return prefix + escaped + suffix


@dataclass
class CorpusOriginResult:
    likely_ai_dialogue: bool
    confidence: float
    primary_platform: Optional[str]
    user_name: Optional[str] = None
    agent_persona_names: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_origin_heuristic(samples: list[str]) -> CorpusOriginResult:
    combined = "\n\n".join(samples)
    total_chars = max(1, len(combined))

    unambiguous_hits: dict[str, int] = {}
    total_unambiguous = 0
    for term in _AI_UNAMBIGUOUS_TERMS:
        matches = re.findall(_brand_pattern(term), combined, re.IGNORECASE)
        if matches:
            unambiguous_hits[term] = len(matches)
            total_unambiguous += len(matches)

    ambiguous_hits: dict[str, int] = {}
    total_ambiguous = 0
    for term in _AI_AMBIGUOUS_TERMS:
        matches = re.findall(_brand_pattern(term), combined, re.IGNORECASE)
        if matches:
            ambiguous_hits[term] = len(matches)
            total_ambiguous += len(matches)

    turn_hits = 0
    turn_types_found = set()
    for pattern in _TURN_MARKERS:
        matches = re.findall(pattern, combined, re.IGNORECASE)
        if matches:
            turn_hits += len(matches)
            turn_types_found.add(pattern)

    has_ai_context = total_unambiguous > 0 or turn_hits > 0
    counted_brand_hits = total_unambiguous + (total_ambiguous if has_ai_context else 0)

    brand_density = counted_brand_hits / (total_chars / 1000)
    turn_density = turn_hits / (total_chars / 1000)

    evidence: list[str] = []
    shown_hits = dict(unambiguous_hits)
    if has_ai_context:
        shown_hits.update(ambiguous_hits)
    if shown_hits:
        top_terms = sorted(shown_hits.items(), key=lambda x: -x[1])[:5]
        evidence.append("AI brand terms: " + ", ".join(f"'{k}' ({v}x)" for k, v in top_terms))
    elif ambiguous_hits and not has_ai_context:
        suppressed = sorted(ambiguous_hits.items(), key=lambda x: -x[1])[:3]
        evidence.append(
            "Ambiguous terms present but suppressed (no co-occurring AI signal): "
            + ", ".join(f"'{k}' ({v}x)" for k, v in suppressed)
        )
    if turn_hits:
        evidence.append(
            f"Turn markers detected: {turn_hits} occurrences across {len(turn_types_found)} pattern types"
        )

    MEANINGFUL_TEXT_FLOOR = 150

    if brand_density >= 0.5 or turn_density >= 2.0:
        return CorpusOriginResult(
            likely_ai_dialogue=True,
            confidence=min(0.95, 0.6 + 0.1 * (brand_density + turn_density)),
            primary_platform=None,
            evidence=evidence,
        )
    if counted_brand_hits == 0 and turn_hits == 0 and total_chars >= MEANINGFUL_TEXT_FLOOR:
        narrative_evidence = list(evidence) + [
            f"no unambiguous AI signal across {total_chars} chars of text — pure narrative"
        ]
        return CorpusOriginResult(
            likely_ai_dialogue=False,
            confidence=0.9,
            primary_platform=None,
            evidence=narrative_evidence,
        )
    reason = "weak signal" if (counted_brand_hits or turn_hits) else "insufficient text"
    return CorpusOriginResult(
        likely_ai_dialogue=True,
        confidence=0.4,
        primary_platform=None,
        evidence=evidence
        + [
            f"{reason} — applying default-stance (ai_dialogue=True, low confidence). "
            "Tier 2 LLM check recommended to confirm or override."
        ],
    )


_SYSTEM_PROMPT = """You are analyzing a corpus of text to determine whether it is a \
record of conversations with an AI agent (e.g. Claude, ChatGPT, Gemini, custom LLM \
apps), or some other kind of text (personal narrative, story, research notes, \
journal, code, etc.).

Use your pre-existing knowledge of well-known AI platforms. You don't need the \
corpus to explain what Claude or ChatGPT is — you already know. Your job is to \
detect evidence of their presence and identify what persona-names the user has \
assigned to the agent(s) they converse with.

CRITICAL distinction:
  - agent_persona_names are names the USER has assigned to the AI AGENT(S)
    they converse with.
  - Do NOT include the USER's own name in agent_persona_names.
  - If you can identify the user's name from context, put it in user_name
    (separate field). If unclear, leave user_name null.

Respond with JSON only (no prose before or after):
{
  "is_ai_dialogue_corpus": <true|false>,
  "confidence": <0.0 to 1.0>,
  "primary_platform": <"Claude (Anthropic)" | "ChatGPT (OpenAI)" | "Gemini (Google)" | other platform name | null>,
  "user_name": <user's name if clearly identifiable from context, else null>,
  "agent_persona_names": [<names the user has assigned to the AI AGENT(S)>],
  "evidence": [<short bullet strings explaining the decision>]
}

Default stance: if evidence is thin or mixed, return is_ai_dialogue_corpus=true \
with low confidence."""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def detect_origin_llm(samples: list[str], provider) -> CorpusOriginResult:
    from kai_mempalace.llm_client import LLMProvider

    max_excerpt_chars = 800
    excerpts = "\n\n---\n\n".join(
        f"[sample {i + 1}]\n{s[:max_excerpt_chars]}" for i, s in enumerate(samples[:20])
    )
    user_prompt = f"CORPUS EXCERPTS:\n\n{excerpts}\n\nAnalyze and respond with JSON."

    try:
        resp = provider.classify(system=_SYSTEM_PROMPT, user=user_prompt, json_mode=True)
        raw = getattr(resp, "text", "") or ""
    except Exception as e:
        return CorpusOriginResult(
            likely_ai_dialogue=True,
            confidence=0.3,
            primary_platform=None,
            evidence=[f"LLM provider error (fallback to default stance): {e}"],
        )

    parsed = _extract_json(raw)
    if not parsed or not isinstance(parsed, dict):
        return CorpusOriginResult(
            likely_ai_dialogue=True,
            confidence=0.3,
            primary_platform=None,
            evidence=["LLM response was not valid JSON (fallback to default stance)"],
        )

    user_name = parsed.get("user_name") or None
    personas = list(parsed.get("agent_persona_names") or [])
    if user_name:
        personas = [p for p in personas if p.lower() != user_name.lower()]
    return CorpusOriginResult(
        likely_ai_dialogue=bool(parsed.get("is_ai_dialogue_corpus", True)),
        confidence=float(parsed.get("confidence", 0.5)),
        primary_platform=parsed.get("primary_platform") or None,
        user_name=user_name,
        agent_persona_names=personas,
        evidence=list(parsed.get("evidence") or []),
    )
