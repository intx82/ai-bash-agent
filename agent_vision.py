#!/usr/bin/env python3
"""Optional image feedback from MCP to a vision-capable OpenAI-compatible model.

AGENT_VISION_MODE=off|main|sidecar.  Never assume that the active LLM can see images.
Images are supplied only to the next inference, not embedded in persisted history.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MAX_IMAGE_BYTES = 8 * 1024 * 1024
SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/webp"}


def _image_part(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    images = result.get("images")
    if not isinstance(images, list):
        return None
    for item in images:
        if isinstance(item, dict) and "path" in item:
            return item
    return None


def _image_data_uri(info: Dict[str, Any]) -> str:
    mime = info.get("mime_type")
    if mime not in SUPPORTED_MIMES:
        raise ValueError("Unsupported image MIME type")
    path = Path(info["path"])
    if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError("Screenshot is missing or exceeds the 8 MiB vision limit")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("Empty screenshot")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def vision_mode() -> str:
    value = os.environ.get("AGENT_VISION_MODE", "off").lower().strip()
    return value if value in ("off", "main", "sidecar") else "off"


def image_feedback(result: Dict[str, Any], text_result: str) -> Tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
    """Return a report and optional one-turn multimodal message for main LLM.

    Sidecar vision runs once and converts the image to a text description for a
    text-only main model. Does not assume vision support from model names.
    """
    info = _image_part(result)
    if info is None:
        if result.get("images"):
            return ({"status": "unavailable", "reason": "MCP image could not be saved",
                     "images": result.get("images")}, None)
        return ({}, None)
    mode = vision_mode()
    if mode == "off":
        return ({"status": "not_analyzed", "reason":
                 "Image saved but not sent to the LLM. Set AGENT_VISION_MODE=main for a vision model, or sidecar for a separate vision endpoint."}, None)
    try:
        uri = _image_data_uri(info)
    except Exception as exc:
        return ({"status": "unavailable", "reason": str(exc)}, None)
    if mode == "main":
        return ({"status": "sent_to_main_model", "note":
                 "Image attached to the next inference; visual interpretation is not yet verified."},
                [{"type": "text", "text": text_result},
                 {"type": "image_url", "image_url": {"url": uri, "detail": "auto"}}])

    url = os.environ.get("AGENT_VISION_URL", "").strip()
    model = os.environ.get("AGENT_VISION_MODEL", "").strip()
    if not url or not model or not url.startswith(("http://", "https://")):
        return ({"status": "unavailable", "reason":
                 "Set AGENT_VISION_URL and AGENT_VISION_MODEL for sidecar mode."}, None)
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        return ({"status": "unavailable", "reason":
                 "Remote vision endpoint must use HTTPS to protect screenshots."}, None)
    try:
        import requests
        headers: Dict[str, str] = {}
        key = os.environ.get("AGENT_VISION_API_KEY", "").strip()
        if key:
            headers["Authorization"] = "Bearer " + key
        response = requests.post(url,
            headers=headers,
            json={"model": model, "stream": False, "max_tokens": 500,
                  "messages": [{"role": "user", "content": [
                      {"type": "text", "text":
                       "Describe what is actually visible in this FreeCAD viewport. "
                       "Mention observable geometry or anomalies, but do not infer precise dimensions from pixels. "
                       "Be concise and indicate uncertainty."},
                      {"type": "image_url", "image_url": {"url": uri, "detail": "auto"}}]}]},
            timeout=(10, 90))
        response.raise_for_status()
        description = response.json()["choices"][0]["message"]["content"]
        if not isinstance(description, str) or not description.strip():
            raise RuntimeError("Vision endpoint returned no text description")
        return ({"status": "analyzed_by_sidecar", "model": model,
                 "visual_description": description[:4000],
                 "note": "This is a model interpretation, not a measurement."}, None)
    except Exception as exc:
        return ({"status": "unavailable", "reason": f"Sidecar vision failed: {exc}"}, None)
