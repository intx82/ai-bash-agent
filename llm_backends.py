#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol
import re
from urllib.parse import urlsplit, urlunsplit

import requests


AGENT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "const": "message"},
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "message": {"type": "string"},
            },
            "required": ["type", "plan", "message"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "const": "tool"},
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "tool": {"type": "string", "const": "bash"},
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "message": {"type": "string"},
            },
            "required": ["type", "plan", "tool", "commands"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "const": "tool"},
                "plan": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 3,
                },
                "tool": {"type": "string", "const": "mcp"},
                "name": {"type": "string"},
                "arguments": {"type": "object"},
                "message": {"type": "string"},
            },
            "required": ["type", "plan", "tool", "name", "arguments"],
        },
    ],
}

AGENT_RESPONSE_FORMAT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_response",
        "schema": AGENT_RESPONSE_SCHEMA,
        "strict": True,
    },
}

AGENT_RESPONSE_FORMAT_JSON_OBJECT: Dict[str, Any] = {"type": "json_object"}


class ChatBackend(Protocol):
    def check_connection(self) -> bool: ...
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[Dict[str, str]]: ...


@dataclass
class BackendConfig:
    server: str
    model: str
    url: str
    api_key: Optional[str] = None
    prefer_json_schema: bool = True
    request_timeout_s: int = 4800


class LlamaCppBackend:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if cfg.api_key:
            self.session.headers["Authorization"] = f"Bearer {cfg.api_key}"

        self.id_slot = int(os.environ.get("LLAMA_ID_SLOT", "0"))
        self.cache_prompt = True

    def check_connection(self) -> bool:
        try:
            models_url = self.cfg.url.replace("/v1/chat/completions", "/v1/models")
            r = self.session.get(models_url, timeout=5)
            if r.status_code == 200:
                return True
            print(f"✗ llama.cpp /v1/models returned {r.status_code}: {r.text[:200]}")
            return False
        except Exception as e:
            print(f"✗ llama.cpp connection error: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[Dict[str, str]]:
        base_payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        bases: List[Dict[str, Any]] = []
        with_slot = dict(base_payload)
        with_slot["id_slot"] = self.id_slot
        with_slot["cache_prompt"] = self.cache_prompt
        bases.append(with_slot)
        bases.append(base_payload)

        rfs: List[Optional[Dict[str, Any]]] = []
        if self.cfg.prefer_json_schema:
            rfs.append(AGENT_RESPONSE_FORMAT_JSON_SCHEMA)
        rfs.append(AGENT_RESPONSE_FORMAT_JSON_OBJECT)
        rfs.append(None)

        last_err: Optional[str] = None

        for base in bases:
            for rf in rfs:
                payload = dict(base)
                if rf is not None:
                    payload["response_format"] = rf

                try:
                    timeout = (5, self.cfg.request_timeout_s)
                    r = self.session.post(self.cfg.url, json=payload, timeout=timeout)
                    if r.status_code != 200:
                        last_err = f"{r.status_code}: {r.text[:200]}"
                        continue

                    data = r.json()
                    msg = data["choices"][0]["message"]

                    content = msg.get("content", "") or ""

                    # Support both field names seen in llama.cpp ecosystem
                    reasoning = (
                        msg.get("reasoning")
                        or msg.get("reasoning_content")
                        or ""
                    )

                    # Fallback: extract <think>...</think> from content if present
                    if not reasoning and isinstance(content, str):
                        m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                        if m:
                            reasoning = m.group(1).strip()
                            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

                    return {
                        "content": content if isinstance(content, str) else "",
                        "reasoning": reasoning if isinstance(reasoning, str) else "",
                    }

                except Exception as e:
                    last_err = str(e)

        print(f"✗ llama.cpp request failed: {last_err}")
        return None


class OpenAICompatibleBackend:
    """Generic /v1/chat/completions backend without llama.cpp-only fields.

    Supports API-key-authenticated OpenAI and other compatible providers.
    The API key is held in memory and is never written into chat files.
    """
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if cfg.api_key:
            self.session.headers["Authorization"] = f"Bearer {cfg.api_key}"

    def check_connection(self) -> bool:
        try:
            parsed = urlsplit(self.cfg.url)
            if parsed.scheme not in ("https", "http") or not parsed.hostname:
                print("Invalid OpenAI-compatible chat URL")
                return False
            if not parsed.path.endswith("/chat/completions"):
                print("URL must end with /chat/completions")
                return False
            models_path = parsed.path[:-len("/chat/completions")] + "/models"
            models_url = urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))
            response = self.session.get(models_url, timeout=(5, 10))
            if response.ok:
                return True
            print(f"Connection test failed: HTTP {response.status_code}: {response.text[:180]}")
            return False
        except requests.RequestException as exc:
            print(f"Connection test failed: {type(exc).__name__}: {exc}")
            return False

    def chat_completion(self, messages: List[Dict[str, Any]], max_tokens: int,
                        temperature: float) -> Optional[Dict[str, str]]:
        base: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
        }
        # Reasoning models may reject temperature or max_tokens. Avoid sending
        # llama.cpp-only fields (id_slot/cache_prompt) to remote APIs.
        output_limits = ({"max_completion_tokens": max_tokens},
                         {"max_tokens": max_tokens})
        # The historical AGENT_RESPONSE_SCHEMA uses root oneOf, rejected by
        # OpenAI's strict Structured Outputs schema validation. Prefer JSON
        # object mode for remote servers rather than paying for a guaranteed
        # HTTP 400 on every request.
        formats: List[Optional[Dict[str, Any]]] = [
            AGENT_RESPONSE_FORMAT_JSON_OBJECT, None,
        ]
        last_error = "unknown error"
        for rf in formats:
            for limit in output_limits:
                payload = dict(base, **limit)
                if rf is not None:
                    payload["response_format"] = rf
                try:
                    response = self.session.post(
                        self.cfg.url, json=payload,
                        timeout=(8, self.cfg.request_timeout_s),
                    )
                    if not response.ok:
                        last_error = f"HTTP {response.status_code}: {response.text[:180]}"
                        # Never retry 401/403 with different payloads.
                        if response.status_code in (401, 403):
                            print(f"Remote model request failed: {last_error}")
                            return None
                        continue
                    message = response.json()["choices"][0]["message"]
                    content = message.get("content", "") or ""
                    if not isinstance(content, str):
                        content = ""
                    reasoning = (message.get("reasoning_content")
                                 or message.get("reasoning") or "")
                    return {"content": content, "reasoning":
                            reasoning if isinstance(reasoning, str) else ""}
                except (requests.RequestException, ValueError, KeyError, IndexError,
                        TypeError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
        print(f"Remote model request failed: {last_error}")
        return None


class OpenAIChatGPTBackend:
    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI backend selected, but python module 'openai' is not installed.") from e

        api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set.")

        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url and cfg.url:
            parts = urlsplit(cfg.url)
            if parts.path.endswith("/chat/completions") and parts.hostname == "api.openai.com":
                base_url = urlunsplit((parts.scheme, parts.netloc,
                                      parts.path[:-len("/chat/completions")], "", ""))
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url)
            if base_url
            else OpenAI(api_key=api_key)
        )

    def check_connection(self) -> bool:
        try:
            _ = self.client.models.list()
            return True
        except Exception as e:
            print(f"✗ OpenAI connection error: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[Dict[str, str]]:
        response_format_variants: List[Optional[Dict[str, Any]]] = []
        if self.cfg.prefer_json_schema:
            response_format_variants.append(AGENT_RESPONSE_FORMAT_JSON_SCHEMA)
        response_format_variants.append(AGENT_RESPONSE_FORMAT_JSON_OBJECT)
        response_format_variants.append(None)

        param_variants: List[Dict[str, Any]] = [
            {"max_completion_tokens": max_tokens, "temperature": temperature},
            {"max_completion_tokens": max_tokens},
            {"max_tokens": max_tokens, "temperature": temperature},
            {"max_tokens": max_tokens},
        ]

        last_err: Optional[str] = None
        for rf in response_format_variants:
            for pv in param_variants:
                try:
                    kwargs: Dict[str, Any] = {
                        "model": self.cfg.model,
                        "messages": messages,
                        **pv,
                    }
                    if rf is not None:
                        kwargs["response_format"] = rf
                    completion = self.client.chat.completions.create(**kwargs)
                    return {"content": completion.choices[0].message.content or "", "reasoning": ""}
                except Exception as e:
                    last_err = str(e)

        print(f"✗ OpenAI request failed: {last_err}")
        return None


def make_backend(cfg: BackendConfig) -> ChatBackend:
    if cfg.server == "llama.cpp":
        # User-provided OpenAI endpoints must not receive id_slot/cache_prompt.
        if urlsplit(cfg.url).hostname == "api.openai.com":
            return OpenAICompatibleBackend(cfg)
        return LlamaCppBackend(cfg)
    if cfg.server == "openai-compatible":
        return OpenAICompatibleBackend(cfg)
    if cfg.server in ("chatgpt", "openai"):
        return OpenAIChatGPTBackend(cfg)
    raise ValueError(f"Unknown server: {cfg.server}")