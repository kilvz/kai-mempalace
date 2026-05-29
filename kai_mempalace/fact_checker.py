"""fact_checker.py — Verify text against known facts in the palace.

Checks AI responses, diary entries, and new content against the entity
registry and knowledge graph for three classes of issue:

  * similar_name          — text mentions a name that's one/two edits
                            away from another registered name.
  * relationship_mismatch — text asserts a role between two entities
                            while the KG records a different current role.
  * stale_fact            — text asserts a fact that the KG marks closed.

Purely offline. No network.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("kai_mempalace")

_RELATIONSHIP_PATTERNS = [
    re.compile(r"\b([A-Z][\w-]+)\s+is\s+([A-Z][\w-]+)'s\s+([a-z]{3,20})\b"),
    re.compile(r"\b([A-Z][\w-]+)'s\s+([a-z]{3,20})\s+is\s+([A-Z][\w-]+)\b"),
]


def _load_entity_names(palace_path: str = None) -> set:
    """Load entity names from the knowledge graph."""
    try:
        from kai_mempalace.backends.knowledge_graph import KnowledgeGraph
        from kai_mempalace.config import KaiPalaceConfig

        config = KaiPalaceConfig()
        kg_path = os.path.join(palace_path or config.palace_path, "knowledge_graph.sqlite3")
        kg = KnowledgeGraph(path=kg_path)
        facts = kg.query()
        names = set()
        for f in facts:
            if f.get("subject"):
                names.add(f["subject"])
            if f.get("object"):
                names.add(f["object"])
        kg.close()
        return names
    except Exception:
        return set()


def check_text(text: str, palace_path: str = None, config=None) -> list:
    if config is None:
        from kai_mempalace.config import KaiPalaceConfig
        config = KaiPalaceConfig()
    if palace_path is None:
        palace_path = config.palace_path

    if not text:
        return []

    issues: list = []
    entity_names = _load_entity_names(palace_path)

    issues.extend(_check_entity_confusion(text, entity_names))
    issues.extend(_check_kg_contradictions(text, palace_path))

    return issues


def _check_entity_confusion(text: str, all_names: set) -> list:
    if not all_names:
        return []

    mentioned: list = []
    for name in all_names:
        if re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE):
            mentioned.append(name)
    if not mentioned:
        return []

    issues: list = []
    seen_pairs: set = set()
    for name_a in mentioned:
        a_lower = name_a.lower()
        for name_b in all_names:
            if name_b == name_a:
                continue
            pair_key = tuple(sorted((name_a.lower(), name_b.lower())))
            if pair_key in seen_pairs:
                continue
            if name_b in mentioned:
                seen_pairs.add(pair_key)
                continue
            distance = _edit_distance(a_lower, name_b.lower())
            if 0 < distance <= 2:
                issues.append({
                    "type": "similar_name",
                    "detail": f"'{name_a}' mentioned — did you mean '{name_b}'? (edit distance {distance})",
                    "names": [name_a, name_b],
                    "distance": distance,
                })
                seen_pairs.add(pair_key)
    return issues


def _extract_claims(text: str) -> list:
    claims: list = []
    for pat in _RELATIONSHIP_PATTERNS:
        for match in pat.finditer(text):
            groups = match.groups()
            if pat is _RELATIONSHIP_PATTERNS[0]:
                subject, possessor, role = groups[0], groups[1], groups[2]
            else:
                possessor, role, subject = groups[0], groups[1], groups[2]
            claims.append({
                "subject": subject,
                "predicate": role.lower(),
                "object": possessor,
                "span": match.group(0),
            })
    return claims


def _check_kg_contradictions(text: str, palace_path: str) -> list:
    claims = _extract_claims(text)
    if not claims:
        return []

    try:
        from kai_mempalace.backends.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(path=palace_path)
    except Exception:
        return []

    issues: list = []
    for claim in claims:
        subject = claim["subject"]
        claim_pred = claim["predicate"]
        claim_obj = claim["object"]
        try:
            facts = kg.query_entity(subject, direction="outgoing")
        except Exception:
            logger.debug("KG lookup failed for subject %r", subject, exc_info=True)
            continue
        if not facts:
            continue

        current_facts = [f for f in facts if f.get("valid_to") is None or str(f["valid_to"]) > datetime.now(timezone.utc).date().isoformat()]

        for fact in current_facts:
            if not _objects_match(fact.get("object"), claim_obj):
                continue
            kg_pred = (fact.get("predicate") or "").lower()
            if kg_pred and kg_pred != claim_pred:
                issues.append({
                    "type": "relationship_mismatch",
                    "detail": f"Text says '{claim['span']}' but KG records {subject} {kg_pred} {fact.get('object')}",
                    "entity": subject,
                    "claim": {"predicate": claim_pred, "object": claim_obj},
                    "kg_fact": {"predicate": kg_pred, "object": fact.get("object")},
                })

        now_iso = datetime.now(timezone.utc).date().isoformat()
        for fact in facts:
            if fact.get("valid_to") is None:
                continue
            kg_pred = (fact.get("predicate") or "").lower()
            if kg_pred != claim_pred:
                continue
            if not _objects_match(fact.get("object"), claim_obj):
                continue
            valid_to = fact.get("valid_to")
            if valid_to and str(valid_to) < now_iso:
                issues.append({
                    "type": "stale_fact",
                    "detail": f"Text says '{claim['span']}' but KG marks this fact closed on {valid_to}",
                    "entity": subject,
                    "valid_to": valid_to,
                })

    return issues


def _objects_match(kg_obj, claim_obj: str) -> bool:
    if kg_obj is None or not claim_obj:
        return False
    return str(kg_obj).strip().lower() == claim_obj.strip().lower()


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]
