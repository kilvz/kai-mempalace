"""onboarding.py — MemPalace first-run setup.

Asks the user:
  1. How they're using MemPalace (work / personal / combo)
  2. Who the people in their life are (names, nicknames, relationships)
  3. What their projects are
  4. What they want their wings called

Seeds the entity registry with confirmed data so MemPalace knows your world
from minute one — before a single session is indexed.
"""

from pathlib import Path

from kai_mempalace.entity_detector import EntityDetector
from kai_mempalace.backends.knowledge_graph import KnowledgeGraph

DEFAULT_WINGS = {
    "work": ["projects", "clients", "team", "decisions", "research"],
    "personal": ["family", "health", "creative", "reflections", "relationships"],
    "combo": ["family", "work", "health", "creative", "projects", "reflections"],
}

COMMON_ENGLISH_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
    "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them",
    "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us", "great", "many", "need", "too", "very", "every",
    "thing", "find", "right", "still", "between", "own", "never", "must",
    "say", "much", "ask", "long", "off", "here", "why", "under", "same",
    "next", "really", "should", "last", "let", "keep", "hand", "place",
    "while", "high", "world", "each", "tell", "set", "three", "run",
    "open", "together", "always", "move", "point", "old", "small",
    "around", "put", "however", "yet", "begin", "better", "best",
    "real", "enough", "though", "until", "always", "away", "face",
    "seem", "big", "another", "close", "end", "far", "few", "group",
    "help", "large", "later", "leave", "life", "light", "line",
    "live", "man", "mean", "might", "more", "most", "name", "never",
    "next", "night", "number", "often", "once", "order", "part",
    "possible", "present", "problem", "public", "quite", "rather",
    "reason", "result", "right", "room", "show", "side", "since",
    "state", "still", "such", "sure", "take", "thing", "think",
    "though", "thus", "today", "together", "top", "true", "turn",
    "used", "value", "various", "view", "voice", "way", "week",
    "whether", "whole", "without", "woman", "word", "work", "year",
}


def _hr():
    print(f"\n{'─' * 58}")


def _header(text):
    print(f"\n{'=' * 58}")
    print(f"  {text}")
    print(f"{'=' * 58}")


def _ask(prompt, default=None):
    if default:
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default
    return input(f"  {prompt}: ").strip()


def _yn(prompt, default="y"):
    val = input(f"  {prompt} [{'Y/n' if default == 'y' else 'y/N'}]: ").strip().lower()
    if not val:
        return default == "y"
    return val.startswith("y")


def _ask_mode() -> str:
    _header("Welcome to MemPalace")
    print("""
  MemPalace is a personal memory system. To work well, it needs to know
  a little about your world.
""")
    print("  How are you using MemPalace?")
    print()
    print("    [1]  Work     — notes, projects, clients, colleagues, decisions")
    print("    [2]  Personal — diary, family, health, relationships, reflections")
    print("    [3]  Both     — personal and professional mixed")
    print()
    while True:
        choice = input("  Your choice [1/2/3]: ").strip()
        if choice == "1":
            return "work"
        elif choice == "2":
            return "personal"
        elif choice == "3":
            return "combo"
        print("  Please enter 1, 2, or 3.")


def _ask_people(mode: str) -> tuple[list, dict]:
    people = []
    aliases = {}
    if mode in ("personal", "combo"):
        _hr()
        print("""
  Personal world — who are the important people in your life?
  Format: name, relationship (e.g. "Riley, daughter" or just "Devon")
  Type 'done' when finished.
""")
        while True:
            entry = input("  Person: ").strip()
            if entry.lower() in ("done", ""):
                break
            parts = [p.strip() for p in entry.split(",", 1)]
            name = parts[0]
            relationship = parts[1] if len(parts) > 1 else ""
            if name:
                nick = input(f"  Nickname for {name}? (or enter to skip): ").strip()
                if nick:
                    aliases[nick] = name
                people.append({"name": name, "relationship": relationship, "context": "personal"})

    if mode in ("work", "combo"):
        _hr()
        print("""
  Work world — who are the colleagues, clients, or collaborators
  you'd want to find in your notes?
  Format: name, role (e.g. "Ben, co-founder" or just "Sarah")
  Type 'done' when finished.
""")
        while True:
            entry = input("  Person: ").strip()
            if entry.lower() in ("done", ""):
                break
            parts = [p.strip() for p in entry.split(",", 1)]
            name = parts[0]
            role = parts[1] if len(parts) > 1 else ""
            if name:
                people.append({"name": name, "relationship": role, "context": "work"})
    return people, aliases


def _ask_projects(mode: str) -> list:
    if mode == "personal":
        return []
    _hr()
    print("""
  What are your main projects? (Type 'done' when finished.)
""")
    projects = []
    while True:
        proj = input("  Project: ").strip()
        if proj.lower() in ("done", ""):
            break
        if proj:
            projects.append(proj)
    return projects


def _ask_wings(mode: str) -> list:
    defaults = DEFAULT_WINGS[mode]
    _hr()
    print(f"""
  Wings are the top-level categories in your memory palace.
  Suggested wings for {mode} mode: {', '.join(defaults)}
  Press enter to keep these, or type your own comma-separated list.
""")
    custom = input("  Wings: ").strip()
    if custom:
        return [w.strip() for w in custom.split(",") if w.strip()]
    return defaults


def _scan_for_detection(directory: str) -> list:
    """Walk directory and return list of text file contents for entity detection."""
    import os
    from pathlib import Path
    texts = []
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    text_exts = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".json", ".yaml", ".yml"}
    for root, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fn in filenames:
            if Path(fn).suffix.lower() not in text_exts:
                continue
            fp = Path(root) / fn
            if fp.is_symlink():
                continue
            try:
                texts.append(fp.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
    return texts


def _auto_detect(directory: str, known_people: list, detector: EntityDetector) -> list:
    known_names = {p["name"].lower() for p in known_people}
    try:
        texts = _scan_for_detection(directory)
        if not texts:
            return []
        all_detected = []
        for text in texts:
            all_detected.extend(detector.detect(text))
        seen = set()
        new_people = []
        for e in all_detected:
            name = e.get("name", "")
            if not name or name.lower() in known_names:
                continue
            normalized = name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            if e.get("confidence", 0) >= 0.7:
                new_people.append(e)
        return new_people
    except Exception:
        return []


def _warn_ambiguous(people: list) -> list:
    ambiguous = []
    for p in people:
        if p["name"].lower() in COMMON_ENGLISH_WORDS:
            ambiguous.append(p["name"])
    return ambiguous


def _seed_kg(kg: KnowledgeGraph, people: list, projects: list, aliases: dict):
    for p in people:
        name = p["name"]
        kg.add_entity(name=name, entity_type="person")
        if p.get("relationship"):
            kg.add(subject=name, predicate="relationship", object=p["relationship"])
        if p.get("context"):
            kg.add(subject=name, predicate="context", object=p["context"])

    for alias, canonical in aliases.items():
        kg.add(subject=canonical, predicate="alias", object=alias)

    for proj in projects:
        kg.add_entity(name=proj, entity_type="project")


def run_onboarding(
    kg_path: str = None,
    directory: str = ".",
    auto_detect: bool = True,
) -> None:
    from kai_mempalace.config import KaiPalaceConfig

    config = KaiPalaceConfig()
    kg = KnowledgeGraph(path=kg_path or config.kg_path)
    detector = EntityDetector()

    mode = _ask_mode()
    people, aliases = _ask_people(mode)
    projects = _ask_projects(mode)
    wings = _ask_wings(mode)

    if auto_detect and _yn("\nScan your files for additional names we might have missed?"):
        directory = _ask("Directory to scan", default=directory)
        detected = _auto_detect(directory, people, detector)
        if detected:
            _hr()
            print(f"\n  Found {len(detected)} additional name candidates:\n")
            for e in detected:
                print(f"    {e.get('name', ''):20} confidence={e.get('confidence', 0):.0%}")
            print()
            if _yn("  Add any of these to your registry?"):
                for e in detected:
                    ans = input(f"    {e.get('name', '')} — (p)erson, (s)kip? ").strip().lower()
                    if ans == "p":
                        rel = input(f"    Relationship/role for {e.get('name', '')}? ").strip()
                        ctx = (
                            "personal" if mode == "personal"
                            else "work" if mode == "work"
                            else input("    Context — (p)ersonal or (w)ork? ").strip().lower()
                            .replace("w", "work").replace("p", "personal")
                        )
                        people.append({"name": e["name"], "relationship": rel, "context": ctx})

    ambiguous = _warn_ambiguous(people)
    if ambiguous:
        _hr()
        print(f"""
  Heads up — these names are also common English words:
    {', '.join(ambiguous)}
  MemPalace will check the context before treating them as person names.
""")

    _seed_kg(kg, people, projects, aliases)

    _header("Setup Complete")
    print()
    print(f"  Mode: {mode}")
    print(f"  People: {len(people)}")
    print(f"  Projects: {len(projects)}")
    print(f"  Wings: {', '.join(wings)}")
    print(f"  KG path: {kg_path or config.kg_path}")
    print()


def quick_setup(
    mode: str,
    people: list,
    projects: list = None,
    aliases: dict = None,
    kg_path: str = None,
) -> None:
    from kai_mempalace.config import KaiPalaceConfig
    config = KaiPalaceConfig()
    kg = KnowledgeGraph(path=kg_path or config.kg_path)
    _seed_kg(kg, people, projects or [], aliases or {})
