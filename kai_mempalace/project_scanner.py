"""project_scanner.py — Detect projects and people from real signal.

Scans a codebase for build manifests (package.json, pyproject.toml,
Cargo.toml, go.mod) and git history to discover project names and
contributors. Pure local, no API.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", "coverage", ".terraform", "vendor",
    "target", ".mempalace", ".cache", ".pytest_cache", ".mypy_cache",
    ".ruff_cache",
}

MAX_DEPTH = 6
MAX_COMMITS_PER_REPO = 1000
GIT_TIMEOUT = 10


@dataclass
class ProjectInfo:
    name: str
    repo_root: Path
    manifest: Optional[str] = None
    has_git: bool = False
    total_commits: int = 0
    user_commits: int = 0
    is_mine: bool = False

    @property
    def confidence(self) -> float:
        if self.is_mine:
            return 0.99
        if self.has_git and self.total_commits > 0:
            return 0.7
        return 0.85

    def to_signal(self) -> str:
        parts: list[str] = []
        if self.manifest:
            parts.append(self.manifest)
        if self.has_git:
            if self.is_mine and self.user_commits:
                parts.append(f"{self.user_commits} of your commits")
            elif self.user_commits:
                parts.append(f"{self.user_commits}/{self.total_commits} yours")
            else:
                parts.append(f"{self.total_commits} commits (none by you)")
        return ", ".join(parts) or "repo"


@dataclass
class PersonInfo:
    name: str
    total_commits: int = 0
    emails: set[str] = field(default_factory=set)
    repos: set[str] = field(default_factory=set)

    @property
    def confidence(self) -> float:
        if self.total_commits >= 100 or len(self.repos) >= 3:
            return 0.99
        if self.total_commits >= 20:
            return 0.85
        return 0.65

    def to_signal(self) -> str:
        r = len(self.repos)
        return f"{self.total_commits} commit{'s' if self.total_commits != 1 else ''} across {r} repo{'s' if r != 1 else ''}"


def _read_manifest(path: Path) -> Optional[str]:
    try:
        if path.name == "package.json":
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return data.get("name")
        elif path.name == "pyproject.toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    return None
            data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
            project = data.get("project", {})
            return project.get("name") or data.get("tool", {}).get("poetry", {}).get("name")
        elif path.name == "Cargo.toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    return None
            data = tomllib.loads(path.read_text(encoding="utf-8", errors="replace"))
            pkg = data.get("package", {})
            if isinstance(pkg, list):
                return pkg[0].get("name") if pkg else None
            return pkg.get("name")
        elif path.name == "go.mod":
            first_line = path.read_text(encoding="utf-8", errors="replace").split("\n")[0]
            m = re.match(r"module\s+(\S+)", first_line)
            if m:
                return m.group(1).split("/")[-1]
        elif path.name == "Gemfile":
            return path.parent.name
        elif path.name in ("setup.py", "setup.cfg"):
            return path.parent.name
    except Exception:
        return None
    return None


def _git_log(repo_path: Path) -> tuple[int, int, bool]:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--max-count=" + str(MAX_COMMITS_PER_REPO + 1)],
            capture_output=True, text=True, timeout=GIT_TIMEOUT,
            cwd=str(repo_path),
        )
        if result.returncode != 0:
            return 0, 0, False
        lines = [l for l in result.stdout.split("\n") if l.strip()]
        total = len(lines)
        return total, total, True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return 0, 0, False


def _find_manifests(root: Path, depth: int = 0) -> list[Path]:
    if depth > MAX_DEPTH:
        return []
    manifests = []
    try:
        for item in root.iterdir():
            if item.name in SKIP_DIRS or item.name.startswith("."):
                continue
            if item.is_dir():
                manifests.extend(_find_manifests(item, depth + 1))
            elif item.is_file() and item.name in (
                "package.json", "pyproject.toml", "Cargo.toml",
                "go.mod", "Gemfile", "setup.py", "setup.cfg",
            ):
                manifests.append(item)
    except (PermissionError, OSError):
        pass
    return manifests


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _git_authors(repo: Path) -> list[tuple[str, str]]:
    out = _run_git(
        repo,
        "log",
        f"--max-count={MAX_COMMITS_PER_REPO}",
        "--format=%aN|%aE",
    )
    result = []
    for line in out.splitlines():
        if "|" in line:
            name, email = line.split("|", 1)
            result.append((name.strip(), email.strip()))
    return result


def _run_git(cwd: Path, *args: str, timeout: int = GIT_TIMEOUT) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return r.stdout if r.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


_BOT_NAME_PATTERNS = [
    r"\[bot\]",
    r"^dependabot",
    r"^renovate",
    r"^github-actions",
    r"^actions-user",
    r"-bot$",
    r"\bbot$",
    r"^bot-",
    r"^snyk",
    r"^greenkeeper",
    r"^semantic-release",
    r"^allcontributors",
    r"-autoroll$",
    r"^auto-format",
    r"^pre-commit-ci",
]
_BOT_EMAIL_PATTERNS = [
    r"bot@",
    r"-bot@",
    r"\[bot\]@",
]
_BOT_RE_NAMES = [re.compile(p) for p in _BOT_NAME_PATTERNS]
_BOT_RE_EMAILS = [re.compile(p) for p in _BOT_EMAIL_PATTERNS]


def _is_bot(name: str, email: str) -> bool:
    ln, le = name.lower(), email.lower()
    return any(rx.search(ln) for rx in _BOT_RE_NAMES) or any(rx.search(le) for rx in _BOT_RE_EMAILS)


def _looks_like_real_name(name: str) -> bool:
    if not name or " " not in name:
        return False
    parts = name.split()
    if len(parts) < 2:
        return False
    return parts[0][:1].isupper() and parts[-1][:1].isupper()


def scan(root: str) -> tuple[list[ProjectInfo], list[PersonInfo]]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        return [], []

    projects = []
    people_map: dict[str, PersonInfo] = {}

    def _add_person(name: str, email: str, repo: Path) -> None:
        if not name or not email:
            return
        if _is_bot(name, email):
            return
        if name not in people_map:
            people_map[name] = PersonInfo(name=name)
        p = people_map[name]
        p.emails.add(email)
        p.repos.add(str(repo))

    def _gather_people(repo: Path) -> None:
        authors = _git_authors(repo)
        for author_name, author_email in authors:
            _add_person(author_name, author_email, repo)

    if _is_git_repo(root_path):
        total, user, has_git = _git_log(root_path)
        manifests = _find_manifests(root_path)
        manifest_name = None
        if manifests:
            manifest_name = _read_manifest(manifests[0])
        name = manifest_name or root_path.name
        projects.append(ProjectInfo(
            name=name, repo_root=root_path, manifest=manifest_name,
            has_git=has_git, total_commits=total, user_commits=user,
        ))
        _gather_people(root_path)

    manifests = _find_manifests(root_path)
    for m in manifests:
        if m.parent == root_path and _is_git_repo(root_path):
            continue
        name = _read_manifest(m)
        if name:
            is_git = _is_git_repo(m.parent)
            tc, uc, _ = _git_log(m.parent) if is_git else (0, 0, False)
            projects.append(ProjectInfo(
                name=name, repo_root=m.parent, manifest=name,
                has_git=is_git, total_commits=tc, user_commits=uc,
            ))
            if is_git:
                _gather_people(m.parent)

    people = sorted(people_map.values(), key=lambda p: -p.confidence)

    # Count total commits per person across all repos they appear in
    for p in people:
        total = 0
        for repo_str in p.repos:
            authors = _git_authors(Path(repo_str))
            total += sum(1 for n, _ in authors if n == p.name)
        p.total_commits = total

    return projects, people


def to_detected_dict(projects: list, people: list) -> dict:
    return {
        "people": [
            {
                "name": p.name,
                "signal": p.to_signal(),
                "confidence": p.confidence,
                "emails": sorted(p.emails),
                "repos": sorted(p.repos),
            }
            for p in people
        ],
        "projects": [
            {"name": p.name, "confidence": p.confidence, "repo": str(p.repo_root)}
            for p in projects
        ],
        "uncertain": [],
    }
