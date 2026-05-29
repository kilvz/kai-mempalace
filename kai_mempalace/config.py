"""kai-mempalace configuration system.

Priority: env vars > config file (~/.kai-palace/config.json) > defaults
"""

import json
import os
import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

MAX_NAME_LENGTH = 128
_SAFE_NAME_RE = re.compile(r"^(?:[^\W_]|[^\W_][\w .'-]{0,126}[^\W_])$")
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def strip_lone_surrogates(text: str) -> str:
    return _LONE_SURROGATE_RE.sub("\ufffd", text)


def normalize_wing_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def sanitize_name(value: str, field_name: str = "name") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")
    if ".." in value or "/" in value or "\\" in value:
        raise ValueError(f"{field_name} contains invalid path characters")
    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")
    if not _SAFE_NAME_RE.match(value):
        raise ValueError(f"{field_name} contains invalid characters")
    return value


def sanitize_kg_value(value: str, field_name: str = "value") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > MAX_NAME_LENGTH:
        raise ValueError(f"{field_name} exceeds maximum length of {MAX_NAME_LENGTH} characters")
    if "\x00" in value:
        raise ValueError(f"{field_name} contains null bytes")
    return strip_lone_surrogates(value)


_ISO_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_ISO_UTC_DATETIME_RE = re.compile(
    r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:Z|\+00:00)$"
)


def _validate_iso_temporal_calendar(value: str) -> None:
    if _ISO_DATE_RE.match(value):
        date.fromisoformat(value)
        return
    if _ISO_UTC_DATETIME_RE.match(value):
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return
    raise ValueError


def sanitize_iso_temporal(value, field_name: str = "date"):
    if value is None or value == "":
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    value = value.strip()
    try:
        _validate_iso_temporal_calendar(value)
    except ValueError:
        raise ValueError(
            f"{field_name}={value!r} is not a valid ISO-8601 date or UTC datetime "
            "(expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)"
        ) from None
    if value.endswith("+00:00"):
        value = f"{value[:-6]}Z"
    return value


def sanitize_content(value: str, max_length: int = 100_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("content must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"content exceeds maximum length of {max_length} characters")
    if "\x00" in value:
        raise ValueError("content contains null bytes")
    return strip_lone_surrogates(value)


DEFAULT_PALACE_PATH = os.path.expanduser("~/.kai-palace")
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_MIN_CHUNK_SIZE = 50

DEFAULT_TOPIC_WINGS = [
    "emotions",
    "consciousness",
    "memory",
    "technical",
    "identity",
    "family",
    "creative",
]

DEFAULT_HALL_KEYWORDS = {
    "emotions": [
        "scared", "afraid", "worried", "happy", "sad",
        "love", "hate", "feel", "cry", "tears",
    ],
    "consciousness": [
        "consciousness", "conscious", "aware", "real",
        "genuine", "soul", "exist", "alive",
    ],
    "memory": ["memory", "remember", "forget", "recall", "archive", "palace", "store"],
    "technical": [
        "code", "python", "script", "bug", "error",
        "function", "api", "database", "server",
    ],
    "identity": ["identity", "name", "who am i", "persona", "self"],
    "family": [
        "family", "kids", "children", "daughter",
        "son", "parent", "mother", "father",
    ],
    "creative": [
        "game", "gameplay", "player", "app",
        "design", "art", "music", "story",
    ],
}


class KaiPalaceConfig:
    """Configuration manager for kai-mempalace.

    Load order: env vars > config file > defaults.
    """

    def __init__(self, config_dir=None):
        self._config_dir = (
            Path(config_dir) if config_dir else Path(os.path.expanduser("~/.kai-palace"))
        )
        self._config_file = self._config_dir / "config.json"
        self._file_config = {}

        if self._config_file.exists():
            try:
                with open(self._config_file, "r") as f:
                    self._file_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._file_config = {}

    @property
    def palace_path(self):
        env_val = os.environ.get("KAI_PALACE_PATH") or os.environ.get("MEMPAL_PALACE_PATH")
        if env_val:
            return os.path.abspath(os.path.expanduser(env_val))
        return self._file_config.get("palace_path", DEFAULT_PALACE_PATH)

    @property
    def tunnel_file(self):
        return os.path.join(os.path.dirname(self.palace_path), "tunnels.json")

    @property
    def topic_wings(self):
        return self._file_config.get("topic_wings", DEFAULT_TOPIC_WINGS)

    @property
    def hall_keywords(self):
        return self._file_config.get("hall_keywords", DEFAULT_HALL_KEYWORDS)

    @staticmethod
    def _try_coerce_int(value, minimum=None):
        if isinstance(value, bool):
            return None
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
            value = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if minimum is not None and value < minimum:
            return None
        return value

    def _coerce_config_int(self, key: str, default: int, minimum=None) -> int:
        coerced = self._try_coerce_int(self._file_config.get(key, default), minimum)
        return default if coerced is None else coerced

    def _validated_chunk_config(self):
        chunk_size = self._coerce_config_int("chunk_size", DEFAULT_CHUNK_SIZE, minimum=1)
        chunk_overlap = self._coerce_config_int("chunk_overlap", DEFAULT_CHUNK_OVERLAP, minimum=0)
        min_chunk_size = self._coerce_config_int("min_chunk_size", DEFAULT_MIN_CHUNK_SIZE, minimum=0)

        if chunk_overlap >= chunk_size:
            chunk_overlap = (
                DEFAULT_CHUNK_OVERLAP
                if DEFAULT_CHUNK_OVERLAP < chunk_size
                else max(0, chunk_size - 1)
            )
        if min_chunk_size > chunk_size:
            min_chunk_size = (
                DEFAULT_MIN_CHUNK_SIZE if DEFAULT_MIN_CHUNK_SIZE <= chunk_size else chunk_size
            )
        return chunk_size, chunk_overlap, min_chunk_size

    @property
    def chunk_size(self) -> int:
        return self._validated_chunk_config()[0]

    @property
    def chunk_overlap(self) -> int:
        return self._validated_chunk_config()[1]

    @property
    def min_chunk_size(self) -> int:
        return self._validated_chunk_config()[2]

    @property
    def min_chunk_size_explicit(self):
        raw = self._file_config.get("min_chunk_size")
        if raw is None:
            return None
        coerced = self._try_coerce_int(raw, minimum=0)
        if coerced is None or coerced > self.chunk_size:
            return None
        return coerced

    @property
    def entity_languages(self):
        env_val = os.environ.get("KAI_ENTITY_LANGUAGES") or os.environ.get("MEMPAL_ENTITY_LANGUAGES")
        if env_val:
            return [s.strip() for s in env_val.split(",") if s.strip()] or ["en"]
        cfg = self._file_config.get("entity_languages")
        if isinstance(cfg, list) and cfg:
            return [str(s) for s in cfg]
        return ["en"]

    def set_entity_languages(self, languages):
        normalized = [s.strip() for s in languages if s and s.strip()]
        if not normalized:
            normalized = ["en"]
        self._file_config["entity_languages"] = normalized
        self._config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        return normalized

    @property
    def topic_tunnel_min_count(self):
        env_val = os.environ.get("KAI_TOPIC_TUNNEL_MIN_COUNT")
        if env_val:
            try:
                parsed = int(env_val)
                if parsed >= 1:
                    return parsed
            except ValueError:
                pass
        cfg_val = self._file_config.get("topic_tunnel_min_count")
        try:
            parsed = int(cfg_val) if cfg_val is not None else 1
        except (TypeError, ValueError):
            parsed = 1
        return max(1, parsed)

    @property
    def hooks_auto_save(self) -> bool:
        val = os.environ.get("KAI_HOOKS_AUTO_SAVE") or os.environ.get("MEMPAL_HOOKS_AUTO_SAVE")
        if val is not None:
            return val.lower() in ("true", "1", "yes", "on")
        return bool(self._file_config.get("hooks_auto_save", True))

    @property
    def hook_silent_save(self) -> bool:
        val = os.environ.get("KAI_HOOK_SILENT_SAVE") or os.environ.get("MEMPAL_HOOK_SILENT_SAVE")
        if val is not None:
            return val.lower() in ("true", "1", "yes", "on")
        return bool(self._file_config.get("hook_silent_save", True))

    @property
    def hook_desktop_toast(self) -> bool:
        val = os.environ.get("KAI_HOOK_DESKTOP_TOAST") or os.environ.get("MEMPAL_HOOK_DESKTOP_TOAST")
        if val is not None:
            return val.lower() in ("true", "1", "yes", "on")
        return bool(self._file_config.get("hook_desktop_toast", False))

    def get_hook_settings(self) -> dict:
        return {
            "silent_save": self.hook_silent_save,
            "desktop_toast": self.hook_desktop_toast,
            "auto_save": self.hooks_auto_save,
        }

    def set_hook_settings(self, silent_save: bool = None, desktop_toast: bool = None, auto_save: bool = None) -> dict:
        if silent_save is not None:
            self._file_config["hook_silent_save"] = bool(silent_save)
        if desktop_toast is not None:
            self._file_config["hook_desktop_toast"] = bool(desktop_toast)
        if auto_save is not None:
            self._file_config["hooks_auto_save"] = bool(auto_save)
        self._config_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._file_config, f, indent=2, ensure_ascii=False)
        except OSError:
            pass
        return self.get_hook_settings()

    def init(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if not self._config_file.exists():
            default_config = {
                "palace_path": DEFAULT_PALACE_PATH,
                "topic_wings": DEFAULT_TOPIC_WINGS,
                "hall_keywords": DEFAULT_HALL_KEYWORDS,
            }
            with open(self._config_file, "w") as f:
                json.dump(default_config, f, indent=2)
        return self._config_file


# Upstream-compatible alias
MempalaceConfig = KaiPalaceConfig
