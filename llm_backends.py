#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import requests


# --------------------------
# Agent response schema (strict)
# --------------------------

AGENT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "const": "message"},
                "plan": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                "message": {"type": "string"},

                # Optional: publish code / markdown / long content safely
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },

                # Optional: small edits as a unified diff (preferred over sed)
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "diff": {"type": "string"},
                        },
                        "required": ["path", "diff"],
                    },
                },
            },
            "required": ["type", "plan", "message"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "type": {"type": "string", "const": "tool"},
                "plan": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
                "tool": {"type": "string", "enum": ["bash"]},
                "commands": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "message": {"type": "string"},

                # Allow these here too (sometimes useful to write/patch before executing)
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path": {"type": "string"},
                            "diff": {"type": "string"},
                        },
                        "required": ["path", "diff"],
                    },
                },
            },
            "required": ["type", "plan", "tool", "commands"],
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


# --------------------------
# Backend interface
# --------------------------

class ChatBackend(Protocol):
    def check_connection(self) -> bool: ...
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]: ...


@dataclass
class BackendConfig:
    server: str                      # "llama.cpp" or "chatgpt"
    model: str
    url: str                         # used for llama.cpp
    api_key: Optional[str] = None    # used for llama.cpp (optional)
    prefer_json_schema: bool = True  # try json_schema first when available
    request_timeout_s: int = 4800


# --------------------------
# llama.cpp backend (OpenAI-compatible endpoint)
# --------------------------
class LlamaCppBackend:
    """
    Talks to a llama.cpp OpenAI-compatible /v1/chat/completions server via requests.Session()
    to get HTTP keep-alive / connection reuse, and pins requests to a slot (id_slot)
    so the server doesn't accumulate multiple prompt caches.
    """

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if cfg.api_key:
            self.session.headers["Authorization"] = f"Bearer {cfg.api_key}"

        # Pin all requests to one slot unless overridden
        # (useful when server runs with --parallel > 1)
        self.id_slot = int(os.environ.get("LLAMA_ID_SLOT", "0"))

        # Explicitly request prompt caching (default is usually true, but be explicit)
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
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        base_payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,

            # Slot/KV-cache controls
            "id_slot": self.id_slot,
            "cache_prompt": self.cache_prompt,
        }

        # llama.cpp support varies; try schema -> json_object -> none
        payload_variants: List[Dict[str, Any]] = []

        if self.cfg.prefer_json_schema:
            p = dict(base_payload)
            p["response_format"] = AGENT_RESPONSE_FORMAT_JSON_SCHEMA
            payload_variants.append(p)

        p2 = dict(base_payload)
        p2["response_format"] = AGENT_RESPONSE_FORMAT_JSON_OBJECT
        payload_variants.append(p2)

        payload_variants.append(base_payload)

        last_err: Optional[str] = None
        for payload in payload_variants:
            try:
                # Separate connect/read timeouts: connect fast, read can be long.
                timeout = (5, self.cfg.request_timeout_s)
                r = self.session.post(self.cfg.url, json=payload, timeout=timeout)
                if r.status_code != 200:
                    last_err = f"{r.status_code}: {r.text[:200]}"
                    continue
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = str(e)

        print(f"✗ llama.cpp request failed: {last_err}")
        return None

# --------------------------
# OpenAI / ChatGPT backend (official SDK)
# --------------------------

class OpenAIChatGPTBackend:
    """
    Uses the official openai Python SDK (OpenAI()).
    API key from OPENAI_API_KEY env var.
    """

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "OpenAI backend selected, but python module 'openai' is not installed. "
                "Install it with: pip install openai"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY env var is not set.")

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def check_connection(self) -> bool:
        try:
            _ = self.client.models.list()
            return True
        except Exception as e:
            print(f"✗ OpenAI connection error: {e}")
            return False

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[str]:
        # Some models support json_schema, others only json_object, others none.
        response_format_variants: List[Optional[Dict[str, Any]]] = []
        if self.cfg.prefer_json_schema:
            response_format_variants.append(AGENT_RESPONSE_FORMAT_JSON_SCHEMA)
        response_format_variants.append(AGENT_RESPONSE_FORMAT_JSON_OBJECT)
        response_format_variants.append(None)

        # GPT-5 family often requires max_completion_tokens instead of max_tokens.
        # Also, some models reject temperature (or treat it differently).
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
                    return completion.choices[0].message.content
                except Exception as e:
                    last_err = str(e)
                    continue

        print(f"✗ OpenAI request failed: {last_err}")
        return None


def make_backend(cfg: BackendConfig) -> ChatBackend:
    if cfg.server == "llama.cpp":
        return LlamaCppBackend(cfg)
    if cfg.server in ("chatgpt", "openai"):
        return OpenAIChatGPTBackend(cfg)
    raise ValueError(f"Unknown server: {cfg.server}")