"""
llm_client.py — Minimal provider abstraction for LLM-assisted entity refinement.

Three providers:
- ``ollama`` (default): local models via http://localhost:11434
- ``openai-compat``: any OpenAI-compatible /v1/chat/completions endpoint
- ``anthropic``: official Messages API

All use stdlib ``urllib`` — no external SDKs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _endpoint_is_local(url: Optional[str]) -> bool:
    if not url:
        return True
    try:
        host = (urlparse(url).hostname or "").lower()
    except (ValueError, AttributeError):
        return False
    if not host:
        return True
    if host in _LOCALHOST_HOSTS:
        return True
    if host.endswith(".local"):
        return True
    if host.startswith("10."):
        return True
    if host.startswith("192.168."):
        return True
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                if 16 <= int(parts[1]) <= 31:
                    return True
            except ValueError:
                pass
    if host.startswith("100."):
        parts = host.split(".")
        if len(parts) >= 2:
            try:
                if 64 <= int(parts[1]) <= 127:
                    return True
            except ValueError:
                pass
    if host.startswith("fc") or host.startswith("fd"):
        return True
    return False


class LLMError(RuntimeError):
    """Raised for any provider failure — transport, parse, auth, missing model."""


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: dict


class LLMProvider:
    name: str = "base"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        api_key_source: Optional[str] = None,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout
        self.api_key_source = api_key_source

    def classify(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        raise NotImplementedError

    def check_available(self) -> tuple[bool, str]:
        raise NotImplementedError

    @property
    def is_external_service(self) -> bool:
        return not _endpoint_is_local(self.endpoint)


def _http_post_json(url: str, body: dict, headers: dict, timeout: int) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise LLMError(f"HTTP {e.code} from {url}: {detail or e.reason}") from e
    except (URLError, OSError) as e:
        raise LLMError(f"Cannot reach {url}: {e}") from e
    except json.JSONDecodeError as e:
        raise LLMError(f"Malformed response from {url}: {e}") from e


class OllamaProvider(LLMProvider):
    name = "ollama"
    DEFAULT_ENDPOINT = "http://localhost:11434"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        timeout: int = 180,
        num_ctx: Optional[int] = None,
        **_: object,
    ):
        super().__init__(
            model=model,
            endpoint=endpoint or self.DEFAULT_ENDPOINT,
            timeout=timeout,
        )
        self.num_ctx = num_ctx

    def check_available(self) -> tuple[bool, str]:
        try:
            with urlopen(f"{self.endpoint}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
        except (URLError, HTTPError, OSError, json.JSONDecodeError) as e:
            return False, f"Cannot reach Ollama at {self.endpoint}: {e}"
        names = {m.get("name", "") for m in data.get("models", []) or []}
        wanted = {self.model, f"{self.model}:latest"}
        if not names & wanted:
            return (
                False,
                f"Model '{self.model}' not loaded in Ollama. Run: ollama pull {self.model}",
            )
        return True, "ok"

    def classify(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        options: dict = {"temperature": 0.1}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        if think is not None:
            body["think"] = think
        data = _http_post_json(f"{self.endpoint}/api/chat", body, headers={}, timeout=self.timeout)
        text = (data.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Empty response from Ollama (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


class OpenAICompatProvider(LLMProvider):
    name = "openai-compat"

    def __init__(
        self,
        model: str,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **_: object,
    ):
        if api_key:
            resolved_key = api_key
            source: Optional[str] = "flag"
        else:
            env_key = os.environ.get("OPENAI_API_KEY")
            resolved_key = env_key or None
            source = "env" if env_key else None
        super().__init__(
            model=model,
            endpoint=endpoint,
            api_key=resolved_key,
            timeout=timeout,
            api_key_source=source,
        )

    def _resolve_url(self) -> str:
        if not self.endpoint:
            raise LLMError("openai-compat provider requires --llm-endpoint")
        url = self.endpoint.rstrip("/")
        if url.endswith("/chat/completions"):
            return url
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        return f"{url}/chat/completions"

    def check_available(self) -> tuple[bool, str]:
        if not self.endpoint:
            return False, "no --llm-endpoint configured"
        base = self.endpoint.rstrip("/")
        base = base.removesuffix("/chat/completions").removesuffix("/v1")
        try:
            req = Request(f"{base}/v1/models")
            if self.api_key:
                req.add_header("Authorization", f"Bearer {self.api_key}")
            with urlopen(req, timeout=5):
                pass
        except (URLError, HTTPError, OSError) as e:
            return False, f"Cannot reach {self.endpoint}: {e}"
        return True, "ok"

    def classify(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = _http_post_json(self._resolve_url(), body, headers=headers, timeout=self.timeout)
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {e}") from e
        if not text:
            raise LLMError(f"Empty response from {self.name} (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    DEFAULT_ENDPOINT = "https://api.anthropic.com"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 120,
        **_: object,
    ):
        if api_key:
            resolved_key = api_key
            source: Optional[str] = "flag"
        else:
            env_key = os.environ.get("ANTHROPIC_API_KEY")
            resolved_key = env_key or None
            source = "env" if env_key else None
        super().__init__(
            model=model,
            endpoint=endpoint or self.DEFAULT_ENDPOINT,
            api_key=resolved_key,
            timeout=timeout,
            api_key_source=source,
        )

    def check_available(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "ANTHROPIC_API_KEY not set (use --llm-api-key or env)"
        return True, "ok"

    def classify(
        self,
        system: str,
        user: str,
        json_mode: bool = True,
        think: Optional[bool] = None,
    ) -> LLMResponse:
        if not self.api_key:
            raise LLMError("Anthropic provider requires ANTHROPIC_API_KEY env or --llm-api-key")
        sys_prompt = system
        if json_mode:
            sys_prompt += "\n\nRespond with valid JSON only, no prose."
        body = {
            "model": self.model,
            "max_tokens": 2048,
            "temperature": 0.1,
            "system": sys_prompt,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "X-API-Key": self.api_key,
            "anthropic-version": self.API_VERSION,
        }
        data = _http_post_json(
            f"{self.endpoint}/v1/messages", body, headers=headers, timeout=self.timeout
        )
        try:
            text = "".join(
                b.get("text", "") for b in data.get("content", []) or [] if b.get("type") == "text"
            )
        except (AttributeError, TypeError) as e:
            raise LLMError(f"Unexpected response shape: {e}") from e
        if not text:
            raise LLMError(f"Empty response from Anthropic (model={self.model})")
        return LLMResponse(text=text, model=self.model, provider=self.name, raw=data)


PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "openai-compat": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(
    name: str,
    model: str,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 120,
    **provider_kwargs: object,
) -> LLMProvider:
    cls = PROVIDERS.get(name)
    if cls is None:
        raise LLMError(f"Unknown provider '{name}'. Choices: {sorted(PROVIDERS.keys())}")
    return cls(model=model, endpoint=endpoint, api_key=api_key, timeout=timeout, **provider_kwargs)
