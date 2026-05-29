"""i18n — Language dictionaries for Kai MemPalace."""

import json
from pathlib import Path
from typing import Optional

_LANG_DIR = Path(__file__).parent
_strings: dict = {}
_current_lang: str = "en"
_entity_cache: dict = {}


def _canonical_lang(lang: str) -> Optional[str]:
    if not lang:
        return None
    target = lang.strip().lower()
    for path in _LANG_DIR.glob("*.json"):
        if path.stem.lower() == target:
            return path.stem
    return None


def available_languages() -> list[str]:
    return sorted(p.stem for p in _LANG_DIR.glob("*.json"))


def load_lang(lang: str = "en") -> dict:
    global _strings, _current_lang
    canonical = _canonical_lang(lang)
    if canonical is None:
        canonical = "en"
    lang_file = _LANG_DIR / f"{canonical}.json"
    _strings = json.loads(lang_file.read_text(encoding="utf-8"))
    _current_lang = canonical
    return _strings


def t(key: str, **kwargs) -> str:
    if not _strings:
        load_lang("en")
    parts = key.split(".", 1)
    if len(parts) == 2:
        section, name = parts
        val = _strings.get(section, {}).get(name, key)
    else:
        val = _strings.get(key, key)
    if kwargs and isinstance(val, str):
        try:
            val = val.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return val


def current_lang() -> str:
    return _current_lang


def get_regex() -> dict:
    if not _strings:
        load_lang("en")
    return _strings.get("regex", {})


def _load_entity_section(lang: str) -> dict:
    canonical = _canonical_lang(lang)
    if canonical is None:
        return {}
    lang_file = _LANG_DIR / f"{canonical}.json"
    try:
        data = json.loads(lang_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data.get("entity", {}) or {}


def _script_boundary(chars: str) -> str:
    return (
        rf"(?:(?<=[{chars}])(?=[^{chars}])"
        rf"|(?<=[^{chars}])(?=[{chars}])"
        rf"|^(?=[{chars}])"
        rf"|(?<=[{chars}])$)"
    )


def _expand_b(pattern: str, boundary_chars: str) -> str:
    if not boundary_chars:
        return pattern
    return pattern.replace(r"\b", _script_boundary(boundary_chars))


def _wrap_candidate(raw_pat: str, boundary_chars: str) -> str:
    if boundary_chars:
        b = _script_boundary(boundary_chars)
        return f"{b}({raw_pat}){b}"
    return rf"\b({raw_pat})\b"


def _collect_entity_section(section: dict, acc: dict) -> None:
    boundary_chars = section.get("boundary_chars")
    if section.get("candidate_pattern"):
        acc["candidate_patterns"].append(
            _wrap_candidate(section["candidate_pattern"], boundary_chars)
        )
    if section.get("multi_word_pattern"):
        acc["multi_word_patterns"].append(
            _wrap_candidate(section["multi_word_pattern"], boundary_chars)
        )
    if section.get("direct_address_pattern"):
        acc["direct_address"].append(_expand_b(section["direct_address_pattern"], boundary_chars))
    acc["person_verbs"].extend(
        _expand_b(p, boundary_chars) for p in section.get("person_verb_patterns", [])
    )
    acc["pronouns"].extend(
        _expand_b(p, boundary_chars) for p in section.get("pronoun_patterns", [])
    )
    acc["dialogue"].extend(
        _expand_b(p, boundary_chars) for p in section.get("dialogue_patterns", [])
    )
    acc["project_verbs"].extend(
        _expand_b(p, boundary_chars) for p in section.get("project_verb_patterns", [])
    )
    acc["stopwords"].update(w.lower() for w in section.get("stopwords", []))


def get_entity_patterns(languages=("en",)) -> dict:
    if not languages:
        languages = ("en",)
    languages = tuple(_canonical_lang(lang) or lang for lang in languages)
    key = languages
    if key in _entity_cache:
        return _entity_cache[key]

    acc = {
        "candidate_patterns": [],
        "multi_word_patterns": [],
        "person_verbs": [],
        "pronouns": [],
        "dialogue": [],
        "direct_address": [],
        "project_verbs": [],
        "stopwords": set(),
    }

    found_any = False
    for lang in languages:
        section = _load_entity_section(lang)
        if not section:
            continue
        found_any = True
        _collect_entity_section(section, acc)

    if not found_any:
        _collect_entity_section(_load_entity_section("en"), acc)

    merged = {
        "candidate_patterns": acc["candidate_patterns"],
        "multi_word_patterns": acc["multi_word_patterns"],
        "person_verb_patterns": _dedupe(acc["person_verbs"]),
        "pronoun_patterns": _dedupe(acc["pronouns"]),
        "dialogue_patterns": _dedupe(acc["dialogue"]),
        "direct_address_patterns": acc["direct_address"],
        "project_verb_patterns": _dedupe(acc["project_verbs"]),
        "stopwords": sorted(acc["stopwords"]),
    }
    _entity_cache[key] = merged
    return merged


def _dedupe(items: list) -> list:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


load_lang("en")
