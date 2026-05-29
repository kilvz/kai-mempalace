"""Canonical entity registry backed by KnowledgeGraph."""

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Optional

from kai_mempalace.backends.knowledge_graph import KnowledgeGraph
from kai_mempalace.entity_detector import EntityDetector

logger = logging.getLogger(__name__)

_WIKI_UA = "KaiMempalace/1.0 (entity-research)"
COMMON_ENGLISH_WORDS: set[str] = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them",
    "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "great", "between", "need", "large", "often",
    "without", "thing", "much", "many", "right", "same", "tell", "very", "why",
    "too", "own", "through", "long", "where", "might", "show", "part", "still",
    "every", "read", "hand", "high", "place", "little", "world", "house",
    "keep", "last", "never", "start", "life", "always", "tree", "city",
    "country", "ask", "group", "number", "night", "point", "small", "away",
    "home", "big", "find", "old", "man"
}


_NAME_INDICATOR_PHRASES = [
    " is a name", " is a given name", " is a surname", " is a nickname",
    "is an english", "is a masculine", "is a feminine", "refers to a name",
    "commonly used as a name",
]
_PLACE_INDICATOR_PHRASES = [
    " is a city", " is a town", " is a village", " is a country",
    " is a state", " is a province", " is an island", " is a river",
    " is a mountain", " is a lake", " is a region", " is a county",
    " is a municipality", " is a borough",
]


def _wikipedia_lookup(word: str, timeout: float = 5.0) -> dict:
    """
    Look up a word via Wikipedia REST API.
    Returns inferred type (person/place/concept/unknown) + confidence + summary.
    """
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(word)}"
        req = urllib.request.Request(url, headers={"User-Agent": _WIKI_UA})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        page_type = data.get("type", "")
        extract = data.get("extract", "").lower()
        title = data.get("title", word)

        if page_type == "disambiguation":
            desc = data.get("description", "").lower()
            if any(p in desc for p in ["name", "given name"]):
                return {
                    "inferred_type": "person",
                    "confidence": 0.65,
                    "wiki_summary": extract[:200],
                    "wiki_title": title,
                    "note": "disambiguation page with name entries",
                }
            return {
                "inferred_type": "ambiguous",
                "confidence": 0.4,
                "wiki_summary": extract[:200],
                "wiki_title": title,
            }

        if any(phrase in extract for phrase in _NAME_INDICATOR_PHRASES):
            confidence = 0.90 if f"{word.lower()} is a" in extract or f"{word.lower()} (name" in extract else 0.80
            return {
                "inferred_type": "person",
                "confidence": confidence,
                "wiki_summary": extract[:200],
                "wiki_title": title,
            }

        if any(phrase in extract for phrase in _PLACE_INDICATOR_PHRASES):
            return {
                "inferred_type": "place",
                "confidence": 0.80,
                "wiki_summary": extract[:200],
                "wiki_title": title,
            }

        return {
            "inferred_type": "concept",
            "confidence": 0.60,
            "wiki_summary": extract[:200],
            "wiki_title": title,
        }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                "inferred_type": "unknown",
                "confidence": 0.3,
                "wiki_summary": None,
                "wiki_title": None,
                "note": "not found in Wikipedia",
            }
        return {"inferred_type": "unknown", "confidence": 0.0, "wiki_summary": None}
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as exc:
        logger.debug("wikipedia lookup failed for %r: %s", word, exc)
        return {"inferred_type": "unknown", "confidence": 0.0, "wiki_summary": None}


class EntityRegistry:
    """Canonical entity registry backed by KnowledgeGraph."""

    def __init__(self, palace: "Palace"):
        self._palace = palace
        self._detector = EntityDetector()
        self._synonyms: dict[str, str] = {}

    @property
    def _kg(self) -> KnowledgeGraph:
        return self._palace.kg

    def register(self, text: str, source: str = "") -> int:
        """Detect entities in text and register them in the KG.

        Returns number of new entities registered (not duplicates).

        1. Run entity_detector on text
        2. For each detected entity, resolve to canonical name
        3. If new, add to KG via palace.kg.add(entity, 'is_a', entity_type)
        4. Track source of where entity was found
        """
        if not text or not text.strip():
            return 0

        detected = self._detector.detect(text)
        count = 0

        for entity in detected:
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "entity")
            if not name:
                continue

            if self.resolve(name) is not None:
                continue

            self._kg.add(subject=name, predicate="is_a", object=entity_type, source=source)
            if source:
                self._kg.add(subject=name, predicate="found_in", object=source, source=source)

            count += 1

        return count

    def research(self, word: str, allow_network: bool = False, auto_confirm: bool = False) -> dict:
        """
        Research an unknown word.

        By default this is **local-only**: it checks the wiki cache in the KG
        and returns ``"unknown"`` for uncached words.  Pass
        ``allow_network=True`` to explicitly opt in to an outbound
        Wikipedia lookup.

        Caches result.  If *auto_confirm* is ``False``, marks the entry
        as unconfirmed (needs user review).
        """
        if not word or not word.strip():
            return {
                "inferred_type": "unknown", "confidence": 0.0,
                "wiki_summary": None, "wiki_title": None,
                "word": word, "confirmed": False,
            }

        word = word.strip()

        cached = self._kg.query(entity=word, predicate="wikipedia", direction="outgoing")
        if cached:
            stored = json.loads(cached[0]["object"])
            stored["word"] = word
            return stored

        if not allow_network:
            return {
                "inferred_type": "unknown",
                "confidence": 0.0,
                "wiki_summary": None,
                "wiki_title": None,
                "word": word,
                "confirmed": False,
                "note": "network lookup disabled — pass allow_network=True to query Wikipedia",
            }

        result = _wikipedia_lookup(word)
        result["word"] = word
        result["confirmed"] = auto_confirm

        self._kg.add(
            subject=word, predicate="wikipedia",
            object=json.dumps(result),
            source="wikipedia",
        )

        return result

    def confirm_research(self, word: str, entity_type: str, relationship: str = "", context: str = "personal") -> dict:
        """Mark a researched word as confirmed and add to people registry."""
        cached = self._kg.query(entity=word, predicate="wikipedia", direction="outgoing")
        if cached:
            stored = json.loads(cached[0]["object"])
            stored["confirmed"] = True
            stored["confirmed_type"] = entity_type
            self._kg.add(
                subject=word, predicate="wikipedia",
                object=json.dumps(stored),
                source="wikipedia",
            )

        if entity_type == "person":
            self._kg.add(subject=word, predicate="is_a", object="person", source="wikipedia")
            if word.lower() in COMMON_ENGLISH_WORDS:
                self._kg.add(subject=word, predicate="ambiguous_flag", object="true", source="wikipedia")

        return {"word": word, "confirmed": True, "confirmed_type": entity_type}

    def resolve(self, name: str) -> Optional[str]:
        """Resolve a name to its canonical form.

        Checks synonyms map first, then tries case-insensitive KG lookup.
        Follows merged_into chains. Returns canonical name or None if unknown.
        """
        if not name:
            return None

        name = name.strip()

        if name in self._synonyms:
            return self._synonyms[name]

        for candidate in (name, name.lower(), name.title()):
            facts = self._kg.query(entity=candidate, predicate="is_a", direction="outgoing")
            if not facts:
                continue

            merged = self._kg.query(entity=candidate, predicate="merged_into",
                                    direction="outgoing")
            if merged:
                target = merged[0]["object"]
                while True:
                    chain = self._kg.query(entity=target, predicate="merged_into",
                                           direction="outgoing")
                    if not chain:
                        break
                    target = chain[0]["object"]

                self._synonyms[name] = target
                return target

            if candidate != name:
                self._synonyms[name] = candidate
            return candidate

        return None

    def get_entity(self, name: str) -> Optional[dict]:
        """Get full entity info from KG.

        Returns dict with name, type, aliases, first_seen, last_seen, source, fact_count
        or None if not found.
        """
        canonical = self.resolve(name)
        if canonical is None:
            return None

        facts = self._kg.query(entity=canonical)
        if not facts:
            return None

        entity_type = "entity"
        aliases = set()
        first_seen = None
        last_seen = None
        source = ""

        for f in facts:
            if f["predicate"] == "is_a":
                entity_type = f["object"]
            elif f["predicate"] == "alias":
                aliases.add(f["object"])

            if f["source"] and not source:
                source = f["source"]

            if f["created_at"]:
                if first_seen is None or f["created_at"] < first_seen:
                    first_seen = f["created_at"]
                if last_seen is None or f["created_at"] > last_seen:
                    last_seen = f["created_at"]

        return {
            "name": canonical,
            "type": entity_type,
            "aliases": sorted(aliases),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "source": source,
            "fact_count": len(facts),
        }

    def add_synonym(self, alias: str, canonical: str) -> None:
        """Register an alias for an entity."""
        if not alias or not canonical:
            return

        alias = alias.strip()
        canonical = canonical.strip()

        self._synonyms[alias] = canonical

        if self.resolve(canonical):
            self._kg.add(subject=canonical, predicate="alias", object=alias)

    def find_related(self, name: str, max_distance: int = 2) -> list[dict]:
        """Find entities related to this one via KG traversal.

        Returns list of {entity, relationship, distance} dicts.
        """
        canonical = self.resolve(name)
        if canonical is None:
            return []

        visited = {canonical}
        results = []
        current = [(canonical, 0)]

        for _ in range(max_distance):
            nxt = []
            for node, dist in current:
                for f in self._kg.query(entity=node, direction="outgoing"):
                    if f["predicate"] == "is_a":
                        continue
                    if f["object"] not in visited:
                        visited.add(f["object"])
                        entry = {
                            "entity": f["object"],
                            "relationship": f["predicate"],
                            "distance": dist + 1,
                        }
                        results.append(entry)
                        if dist + 1 < max_distance:
                            nxt.append((f["object"], dist + 1))

                for f in self._kg.query(entity=node, direction="incoming"):
                    if f["subject"] not in visited:
                        visited.add(f["subject"])
                        entry = {
                            "entity": f["subject"],
                            "relationship": f"{f['predicate']}(inverse)",
                            "distance": dist + 1,
                        }
                        results.append(entry)
                        if dist + 1 < max_distance:
                            nxt.append((f["subject"], dist + 1))

            current = nxt

        return results

    def merge(self, source_name: str, target_name: str) -> bool:
        """Merge source entity into target (all relations move to target).

        Returns True if merged, False if either entity not found.
        """
        source_canon = self.resolve(source_name)
        target_canon = self.resolve(target_name)

        if source_canon is None or target_canon is None:
            return False

        if source_canon == target_canon:
            return True

        out_facts = self._kg.query(entity=source_canon, direction="outgoing")
        for f in out_facts:
            if f["predicate"] in ("is_a", "merged_into"):
                continue
            self._kg.add(subject=target_canon, predicate=f["predicate"],
                         object=f["object"], source=f.get("source", ""))

        in_facts = self._kg.query(entity=source_canon, direction="incoming")
        for f in in_facts:
            if f["predicate"] == "merged_into":
                continue
            self._kg.add(subject=f["subject"], predicate=f["predicate"],
                         object=target_canon, source=f.get("source", ""))

        self._kg.add(subject=source_canon, predicate="merged_into", object=target_canon)

        is_a_facts = [f for f in out_facts if f["predicate"] == "is_a"]
        for f in is_a_facts:
            self._kg.invalidate(subject=source_canon, predicate="is_a", object=f["object"])

        self._synonyms[source_canon] = target_canon

        return True

    def suggest_merge(self, threshold: float = 0.85) -> list[tuple[str, str, float]]:
        """Find entities that might be the same (similar names).

        Returns list of (name_a, name_b, similarity) tuples.
        Similarity based on edit distance and shared relationships.
        """
        facts = self._kg.query()
        entities = set()
        for f in facts:
            if f["predicate"] == "is_a":
                entities.add(f["subject"])

        entity_list = sorted(entities)
        suggestions = []

        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                a, b = entity_list[i], entity_list[j]

                if a in self._synonyms or b in self._synonyms:
                    if self._synonyms.get(a, a) == self._synonyms.get(b, b):
                        continue

                name_sim = SequenceMatcher(None, a.lower(), b.lower()).ratio()

                a_preds = set()
                b_preds = set()
                for f in self._kg.query(entity=a, direction="outgoing"):
                    if f["predicate"] != "is_a":
                        a_preds.add(f"{f['predicate']}:{f['object']}")
                for f in self._kg.query(entity=b, direction="outgoing"):
                    if f["predicate"] != "is_a":
                        b_preds.add(f"{f['predicate']}:{f['object']}")

                rel_sim = 0.0
                if a_preds and b_preds:
                    rel_sim = len(a_preds & b_preds) / len(a_preds | b_preds)

                combined = max(name_sim, rel_sim)
                if combined >= threshold:
                    suggestions.append((a, b, combined))

        suggestions.sort(key=lambda x: -x[2])
        return suggestions
