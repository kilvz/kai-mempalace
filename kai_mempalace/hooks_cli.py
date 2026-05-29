"""Hook logic for MemPalace — session-start, stop, and precompact hooks for Claude Code / Codex."""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from kai_mempalace.config import MempalaceConfig

logger = logging.getLogger(__name__)

SAVE_INTERVAL = 15
STATE_DIR = Path.home() / ".mempalace" / "hook_state"
PALACE_ROOT = Path.home() / ".mempalace"


def _detached_popen_kwargs() -> dict:
    kwargs: dict = {"stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        flags = 0
        for name in ("DETACHED_PROCESS", "CREATE_NEW_PROCESS_GROUP", "CREATE_BREAKAWAY_FROM_JOB"):
            flags |= getattr(subprocess, name, 0)
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _palace_root_exists() -> bool:
    return PALACE_ROOT.is_dir()


def _mempalace_python() -> str:
    env_python = os.environ.get("MEMPALACE_PYTHON", "")
    if env_python and os.path.isfile(env_python) and os.access(env_python, os.X_OK):
        return env_python
    parents = Path(__file__).resolve().parents
    if len(parents) > 3:
        venv_bin = parents[3] / "bin" / "python"
        if venv_bin.is_file():
            return str(venv_bin)
    if len(parents) > 1:
        project_venv = parents[1] / "venv" / "bin" / "python"
        if project_venv.is_file():
            return str(project_venv)
    return sys.executable


STOP_BLOCK_REASON = (
    "MemPalace auto-save checkpoint. "
    "Use mempalace_diary_write (session summary) and mempalace_add_drawer "
    "(quotes, decisions, code) to save session content. "
    "Do NOT use native auto-memory files."
)

PRECOMPACT_BLOCK_REASON = (
    "MemPalace emergency save — compaction imminent. "
    "Use mempalace_diary_write (thorough summary) and mempalace_add_drawer "
    "(ALL quotes, decisions, code, context) to save ALL content before context is lost. "
    "Do NOT use native auto-memory files."
)


def _sanitize_session_id(session_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", session_id)
    return sanitized or "unknown"


def _validate_transcript_path(transcript_path: str) -> Optional[Path]:
    if not transcript_path:
        return None
    path = Path(transcript_path).expanduser().resolve()
    if path.suffix not in (".jsonl", ".json"):
        return None
    if ".." in Path(transcript_path).parts:
        return None
    return path


def _count_human_messages(transcript_path: str) -> int:
    path = _validate_transcript_path(transcript_path)
    if path is None:
        if transcript_path:
            logger.warning("transcript_path rejected: %r", transcript_path)
        return 0
    if not path.is_file():
        return 0
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message", {})
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            if "<command-message>" in content:
                                continue
                        elif isinstance(content, list):
                            text = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                            if "<command-message>" in text:
                                continue
                        count += 1
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            msg_text = payload.get("message", "")
                            if isinstance(msg_text, str) and "<command-message>" not in msg_text:
                                count += 1
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return 0
    return count


_state_dir_initialized = False


def _log(message: str):
    if not _palace_root_exists():
        return
    global _state_dir_initialized
    try:
        if not _state_dir_initialized:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                STATE_DIR.chmod(0o700)
            except (OSError, NotImplementedError):
                pass
            _state_dir_initialized = True
        log_path = STATE_DIR / "hook.log"
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        try:
            log_path.chmod(0o600)
        except (OSError, NotImplementedError):
            pass
    except OSError:
        pass


def _output(data: dict):
    payload = (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    fd = 1
    offset = 0
    try:
        while offset < len(payload):
            try:
                offset += os.write(fd, payload[offset:])
            except InterruptedError:
                continue
        return
    except OSError:
        pass
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _get_mine_targets() -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    mempal_dir = os.environ.get("MEMPAL_DIR", "")
    if mempal_dir:
        resolved = Path(mempal_dir).expanduser().resolve()
        if resolved.is_dir():
            targets.append((str(resolved), "projects"))
    return targets


_MINE_PID_DIR = STATE_DIR / "mine_pids"
_MINE_PID_FILE_ENV = "MEMPALACE_MINE_PID_FILE"
_MINE_TIMEOUT_HOURS_ENV = "MEMPALACE_MINE_TIMEOUT_HOURS"
_MINE_TIMEOUT_HOURS_DEFAULT = 2.0


def _mine_slot_timeout_secs() -> float:
    raw = os.environ.get(_MINE_TIMEOUT_HOURS_ENV, "")
    if raw:
        try:
            hours = float(raw)
            return max(0.0, hours) * 3600
        except ValueError:
            return 0.0
    return _MINE_TIMEOUT_HOURS_DEFAULT * 3600


def _pid_file_for_cmd(cmd: list[str]) -> Path:
    try:
        idx = cmd.index("mine")
        key = " ".join(cmd[idx:])
    except ValueError:
        key = " ".join(cmd)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return _MINE_PID_DIR / f"mine_{digest}.pid"


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _mine_already_running(cmd: list[str]) -> bool:
    pid_file = _pid_file_for_cmd(cmd)
    try:
        recorded = pid_file.read_text().strip()
    except OSError:
        return False
    if not recorded:
        return False
    parts = recorded.split(None, 1)
    if not parts[0].isdigit():
        return False
    pid = int(parts[0])
    if not _pid_alive(pid):
        return False
    timeout_secs = _mine_slot_timeout_secs()
    if timeout_secs > 0:
        if len(parts) > 1 and parts[1]:
            try:
                start_ts = float(parts[1])
            except ValueError:
                return False
        else:
            try:
                start_ts = pid_file.stat().st_mtime
            except OSError:
                return True
        if time.time() - start_ts > timeout_secs:
            return False
    return True


def _create_mine_slot_with_placeholder(pid_file: Path) -> Path:
    fd = os.open(str(pid_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write(f"{os.getpid()} {int(time.time())}")
    except OSError:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            pid_file.unlink()
        except OSError:
            pass
        raise
    return pid_file


def _claim_mine_slot(cmd: list[str]) -> Optional[Path]:
    pid_file = _pid_file_for_cmd(cmd)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _create_mine_slot_with_placeholder(pid_file)
    except FileExistsError:
        pass
    if _mine_already_running(cmd):
        return None
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return None
    try:
        return _create_mine_slot_with_placeholder(pid_file)
    except FileExistsError:
        return None


def _spawn_mine(cmd: list) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / "hook.log"
    pid_file = _claim_mine_slot(cmd)
    if pid_file is None:
        _log(f"Skipping mine: target already running ({' '.join(cmd[-3:])})")
        return
    child_env = os.environ.copy()
    child_env[_MINE_PID_FILE_ENV] = str(pid_file)
    with open(log_path, "a") as log_f:
        try:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=child_env, **_detached_popen_kwargs())
        except OSError:
            try:
                pid_file.unlink()
            except OSError:
                pass
            raise
    try:
        pid_file.write_text(f"{proc.pid} {int(time.time())}")
    except OSError:
        pass


def _maybe_auto_ingest():
    targets = _get_mine_targets()
    if not targets:
        return
    for mine_dir, mode in targets:
        try:
            _spawn_mine([_mempalace_python(), "-m", "kai_mempalace", "mine", mine_dir, "--mode", mode])
        except OSError:
            pass


def _mine_sync():
    targets = _get_mine_targets()
    if not targets:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STATE_DIR / "hook.log"
    for mine_dir, mode in targets:
        try:
            with open(log_path, "a") as log_f:
                subprocess.run(
                    [_mempalace_python(), "-m", "kai_mempalace", "mine", mine_dir, "--mode", mode],
                    stdout=log_f, stderr=log_f, timeout=60,
                )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _desktop_toast(body: str, title: str = "MemPalace"):
    try:
        subprocess.Popen(
            ["notify-send", "--app-name=MemPalace", "--icon=brain", title, body],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **_detached_popen_kwargs(),
        )
    except OSError:
        pass


_RECENT_MSG_COUNT = 30


def _extract_recent_messages(transcript_path: str, count: int = _RECENT_MSG_COUNT) -> list[str]:
    path = Path(transcript_path).expanduser()
    if not path.is_file():
        return []
    messages = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    msg = entry.get("message") or entry.get("event_message") or {}
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
                        if not isinstance(content, str) or not content.strip():
                            continue
                        if "<command-message>" in content or "<system-reminder>" in content:
                            continue
                        messages.append(content.strip()[:200])
                    elif entry.get("type") == "event_msg":
                        payload = entry.get("payload", {})
                        if isinstance(payload, dict) and payload.get("type") == "user_message":
                            text = payload.get("message", "")
                            if isinstance(text, str) and text.strip():
                                if "<command-message>" not in text:
                                    messages.append(text.strip()[:200])
                except (json.JSONDecodeError, AttributeError):
                    pass
    except OSError:
        return []
    return messages[-count:]


_THEME_STOPWORDS = frozenset(
    "the a an and or but in on at to for of is it i me my you your we our "
    "this that with from by was were be been are not no yes can do did dont "
    "will would should could have has had lets let just also like so if then "
    "ok okay sure yeah hey hi here there what when where how why which some "
    "all any each every about into out up down over after before between "
    "get got make made need want use used using check look see run try "
    "know think right now still already really very much more most too "
    "file files code one two new first last next thing things way well".split()
)


def _extract_themes(messages: list[str], max_themes: int = 3) -> list[str]:
    words: Counter[str] = Counter()
    for msg in messages:
        for word in msg.lower().split():
            clean = word.strip(".,;:!?\"'`()[]{}#<>/\\-_=+@$%^&*~")
            if len(clean) >= 4 and clean not in _THEME_STOPWORDS and clean.isalpha():
                words[clean] += 1
    return [w for w, _ in words.most_common(max_themes)]


def _save_diary_direct(transcript_path: str, session_id: str, wing: str = "", toast: bool = False) -> dict:
    messages = _extract_recent_messages(transcript_path)
    if not messages:
        _log("No recent messages to save")
        return {"count": 0}

    themes = _extract_themes(messages)
    now = datetime.now()
    topics = "|".join(m[:80] for m in messages[-10:])
    entry = (
        f"CHECKPOINT:{now.strftime('%Y-%m-%d')}|session:{session_id}"
        f"|msgs:{len(messages)}|recent:{topics}"
    )

    try:
        from kai_mempalace.mcp_server import tool_diary_write

        result = tool_diary_write(agent_name="session-hook", entry=entry, topic="checkpoint", wing=wing)
        if result.get("success"):
            _log(f"Diary checkpoint saved: {result.get('entry_id', '?')}")
            try:
                ack_file = STATE_DIR / "last_checkpoint"
                ack_file.write_text(json.dumps({"msgs": len(messages), "ts": now.isoformat()}), encoding="utf-8")
            except OSError:
                pass
            if toast:
                _desktop_toast(f"Checkpoint saved — {len(messages)} messages archived")
            return {"count": len(messages), "themes": themes}
        else:
            _log(f"Diary checkpoint failed: {result.get('error', 'unknown')}")
    except Exception as e:
        _log(f"Diary checkpoint error: {e}")
    return {"count": 0}


def _ingest_transcript(transcript_path: str):
    path = Path(transcript_path).expanduser()
    if not path.is_file() or path.stat().st_size < 100:
        return
    try:
        MempalaceConfig()
    except Exception:
        return
    try:
        _spawn_mine([_mempalace_python(), "-m", "kai_mempalace", "mine", str(path.parent), "--mode", "convos", "--wing", "sessions"])
        _log(f"Transcript ingest started: {path.name}")
    except OSError:
        pass


SUPPORTED_HARNESSES = {"claude-code", "codex"}


def _parse_harness_input(data: dict, harness: str) -> dict:
    if harness not in SUPPORTED_HARNESSES:
        print(f"Unknown harness: {harness}", file=sys.stderr)
        sys.exit(1)
    return {
        "session_id": _sanitize_session_id(str(data.get("session_id", "unknown"))),
        "stop_hook_active": data.get("stop_hook_active", False),
        "transcript_path": str(data.get("transcript_path", "")),
    }


_ENCODED_PARENT_PREFIXES = (
    "git-", "dev-", "projects-", "Projects-", "src-", "code-", "work-", "Documents-",
)


def _wing_from_jsonl_cwd(transcript_path: str) -> Optional[str]:
    try:
        path = Path(transcript_path).expanduser()
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= 200:
                    break
                line = line.strip()
                if not line or '"cwd"' not in line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                cwd = data.get("cwd")
                if not cwd or not isinstance(cwd, str):
                    continue
                cwd_norm = cwd.replace("\\", "/").rstrip("/")
                if not cwd_norm:
                    continue
                project = cwd_norm.rsplit("/", 1)[-1]
                if project:
                    slug = project.lower().replace(" ", "_").replace("-", "_")
                    return f"wing_{slug}"
    except OSError:
        pass
    return None


def _wing_from_transcript_path(transcript_path: str) -> str:
    cwd_wing = _wing_from_jsonl_cwd(transcript_path)
    if cwd_wing:
        return cwd_wing

    normalized = transcript_path.replace("\\", "/")

    match = re.search(r"/\.claude/projects/-([^/]+)", normalized)
    if match:
        encoded = match.group(1)
        m = re.match(r"(?:Users|home)-[^-]+-(.+)", encoded)
        if m:
            encoded = m.group(1)
        for prefix in _ENCODED_PARENT_PREFIXES:
            if encoded.startswith(prefix):
                encoded = encoded[len(prefix):]
                break
        project = encoded.lower().replace(" ", "_").replace("-", "_")
        if project:
            return f"wing_{project}"

    match = re.search(r"-Projects-([^/]+?)(?:/|$)", normalized)
    if match:
        project = match.group(1).lower().replace(" ", "_").replace("-", "_")
        return f"wing_{project}"

    return "wing_sessions"


def hook_stop(data: dict, harness: str):
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    stop_hook_active = parsed["stop_hook_active"]
    transcript_path = parsed["transcript_path"]

    if not MempalaceConfig().hooks_auto_save:
        _output({})
        return

    if str(stop_hook_active).lower() in ("true", "1", "yes"):
        silent_guard = True
        try:
            silent_guard = MempalaceConfig().hook_silent_save
        except AttributeError as exc:
            _log(f"WARNING: could not read hook_silent_save: {exc}; defaulting to silent mode")
        if not silent_guard:
            _output({})
            return

    exchange_count = _count_human_messages(transcript_path)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    last_save_file = STATE_DIR / f"{session_id}_last_save"
    last_save = 0
    if last_save_file.is_file():
        try:
            last_save = int(last_save_file.read_text().strip())
        except (ValueError, OSError):
            last_save = 0

    since_last = exchange_count - last_save
    _log(f"Session {session_id}: {exchange_count} exchanges, {since_last} since last save")

    if since_last >= SAVE_INTERVAL and exchange_count > 0:
        _log(f"TRIGGERING SAVE at exchange {exchange_count}")

        try:
            config = MempalaceConfig()
            silent = config.hook_silent_save
            toast = config.hook_desktop_toast
        except Exception:
            silent = True
            toast = False

        project_wing = _wing_from_transcript_path(transcript_path)

        if silent:
            result = {"count": 0}
            if transcript_path:
                result = _save_diary_direct(transcript_path, session_id, wing=project_wing, toast=toast)
                _ingest_transcript(transcript_path)
            _maybe_auto_ingest()
            count = result.get("count", 0)
            if count > 0:
                try:
                    last_save_file.write_text(str(exchange_count), encoding="utf-8")
                except OSError:
                    pass
                themes = result.get("themes", [])
                tag = " — " + ", ".join(themes) if themes else ""
                _output({"systemMessage": f"✦ {count} memories woven into the palace{tag}"})
            else:
                _output({})
        else:
            try:
                last_save_file.write_text(str(exchange_count), encoding="utf-8")
            except OSError:
                pass
            if transcript_path:
                _ingest_transcript(transcript_path)
            _maybe_auto_ingest()
            reason = STOP_BLOCK_REASON + f" Write diary entry to wing={project_wing}."
            _output({"decision": "block", "reason": reason})
    else:
        _output({})


def hook_session_start(data: dict, harness: str):
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    _log(f"SESSION START for session {session_id}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _output({})


def hook_precompact(data: dict, harness: str):
    if not _palace_root_exists():
        _output({})
        return
    parsed = _parse_harness_input(data, harness)
    session_id = parsed["session_id"]
    transcript_path = parsed["transcript_path"]

    if not MempalaceConfig().hooks_auto_save:
        _output({})
        return

    _log(f"PRE-COMPACT triggered for session {session_id}")

    if transcript_path:
        _ingest_transcript(transcript_path)
    _mine_sync()
    _output({})


def run_hook(hook_name: str, harness: str):
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        _log("WARNING: Failed to parse stdin JSON, proceeding with empty data")
        data = {}

    hooks = {
        "session-start": hook_session_start,
        "stop": hook_stop,
        "precompact": hook_precompact,
    }
    handler = hooks.get(hook_name)
    if handler is None:
        print(f"Unknown hook: {hook_name}", file=sys.stderr)
        sys.exit(1)
    handler(data, harness)
