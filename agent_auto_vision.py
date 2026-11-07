#!/usr/bin/env python3
"""Opt-in FreeCAD screenshot -> OpenAI Vision MCP feedback.

The screenshot path is produced by the local MCP image bridge. Only images inside
WORKSPACE/mcp_images are eligible. No API key or image bytes enter the main LLM
context. The external upload is approved through the agent's confirmation UI.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


DEFAULT_QUESTION = (
    "Inspect the visible FreeCAD 3D viewport. Describe the objects and their "
    "positions, any visible gaps, overlaps, cutouts, holes or geometry anomalies. "
    "Estimate projected object bounding boxes and relative sizes in image pixels. "
    "Distinguish visible observations from assumptions. Do not infer exact "
    "millimeter dimensions from pixels; CAD properties are authoritative."
)
TOOL = "openai_vision.describe_image"
MAX_BYTES = 8 * 1024 * 1024


def _unwrap_result(response: Any) -> Optional[Dict[str, Any]]:
    """Extract the MCP function result from either nested or flat responses."""
    if not isinstance(response, dict):
        return None
    structured = response.get("structured")
    if isinstance(structured, dict):
        result = structured.get("result", structured)
        if isinstance(result, dict):
            return result
    for part in response.get("text", []):
        if isinstance(part, str):
            try:
                parsed = json.loads(part)
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict):
                return parsed.get("result", parsed) if isinstance(parsed.get("result", parsed), dict) else None
    return None


def _config(mcp: Any) -> Dict[str, Any]:
    path = getattr(mcp, "config_path", None)
    if path is None:
        return {}
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    options = config.get("automatic_vision", {})
    if not isinstance(options, dict):
        raise ValueError("automatic_vision must be an object")
    return options


class AutoVision:
    """One controller per agent turn, with a bounded upload budget."""

    def __init__(self, mcp: Any, workdir: str):
        options = _config(mcp)
        self.enabled = options.get("enabled", False)
        if type(self.enabled) is not bool:
            raise ValueError("automatic_vision.enabled must be a boolean")
        self.calls = 0
        self.denied = False
        self.mcp = mcp
        self.workspace = Path(workdir).resolve()
        self.image_dir = (self.workspace / "mcp_images").resolve()
        self.max_calls = options.get("max_calls_per_turn", 4)
        self.max_image_bytes = options.get("max_image_bytes", MAX_BYTES)
        self.detail = options.get("detail", "high")
        self.max_objects = options.get("max_objects", 8)
        self.question = options.get("question", DEFAULT_QUESTION)
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 20:
            raise ValueError("automatic_vision.max_calls_per_turn must be 1..20")
        if type(self.max_image_bytes) is not int or not 1 <= self.max_image_bytes <= MAX_BYTES:
            raise ValueError("automatic_vision.max_image_bytes must be 1..8388608")
        if self.detail not in ("auto", "low", "high", "original"):
            raise ValueError("automatic_vision.detail must be auto, low, high, or original")
        if type(self.max_objects) is not int or not 1 <= self.max_objects <= 12:
            raise ValueError("automatic_vision.max_objects must be 1..12")
        if not isinstance(self.question, str) or not 1 <= len(self.question) <= 4000:
            raise ValueError("automatic_vision.question must contain 1..4000 characters")

    def analyze(
        self,
        tool_name: str,
        response: Any,
        approve: Callable[[str], bool],
        emit: Callable[[str, Any], None],
        check_cancel: Callable[[], None],
    ) -> Optional[Dict[str, Any]]:
        """Return vision report or None when no eligible screenshot exists.

        This method never changes whether the originating FreeCAD tool succeeded.
        An OpenAI failure is reported as vision.status=failed, not a CAD error.
        """
        if not self.enabled or not tool_name.startswith("freecad."):
            return None
        if not isinstance(response, dict) or not response.get("images"):
            return None
        if self.denied:
            return {"status": "denied", "reason": "OpenAI image upload was declined earlier in this turn"}
        if self.calls >= self.max_calls:
            return {"status": "skipped", "reason": "Automatic vision per-turn limit reached"}
        if not hasattr(self.mcp, "has_tool") or not self.mcp.has_tool(TOOL):
            reason = (self.mcp.explain_unavailable_tool(TOOL)
                      if hasattr(self.mcp, "explain_unavailable_tool") else "OpenAI vision MCP not connected")
            return {"status": "unavailable", "reason": reason}
        candidates = [x for x in response.get("images", []) if isinstance(x, dict) and isinstance(x.get("path"), str)]
        if not candidates:
            return {"status": "unavailable", "reason": "Screenshot was returned but could not be saved"}
        image = candidates[0]
        try:
            path = Path(image["path"]).resolve(strict=True)
            # Only accept files the MCP bridge saved into this turn's image folder.
            if not path.is_relative_to(self.image_dir):
                raise ValueError("Screenshot path is outside the MCP image directory")
            if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise ValueError("Screenshot must be a PNG/JPEG/WEBP regular file")
            size = path.stat().st_size
            if not 0 < size <= self.max_image_bytes:
                raise ValueError(f"Screenshot size {size} exceeds the configured limit")
        except (OSError, ValueError) as exc:
            return {"status": "skipped", "reason": str(exc)}

        check_cancel()
        # Always ask before automatically uploading a screenshot; the MCP
        # read_only_tools allowlist must not silently waive this approval.
        prompt = ("Send FreeCAD screenshot to OpenAI Vision for paid analysis? "
                  f"({path.name}, {size} bytes) (y/N): ")
        if not approve(prompt):
            self.denied = True
            emit("MCP tool denied", "Automatic OpenAI screenshot upload declined")
            return {"status": "denied", "reason": "User declined OpenAI screenshot upload"}
        check_cancel()
        self.calls += 1
        args = {"image_path": str(path), "question": self.question,
                "detail": self.detail, "max_objects": self.max_objects}
        emit("MCP tool running", f"Auto vision: {TOOL}({path.name})")
        try:
            response = self.mcp.call_tool(TOOL, args)
            payload = _unwrap_result(response)
            if not isinstance(payload, dict):
                raise RuntimeError("OpenAI Vision returned no structured result")
            if (response.get("isError") is True or response.get("ok") is False
                    or payload.get("ok") is False):
                return {"status": "failed", "reason": str(payload.get("error") or "Vision tool returned an error")[:500]}
            if not isinstance(payload.get("description"), str) or not payload["description"].strip():
                return {"status": "failed", "reason": "OpenAI Vision returned no description"}
            # Provider returns only bounded, model-generated text and metadata, not pixels.
            wanted = ("model", "width_px", "height_px", "description", "objects", "relative_sizes",
                      "observations", "limitations", "measurement_note")
            report = {"status": "analyzed_by_openai_mcp", "tool": TOOL, "image_path": str(path)}
            report.update({key: payload[key] for key in wanted if key in payload})
            emit("MCP tool result", {"automatic_vision": report})
            return report
        except Exception as exc:
            return {"status": "failed", "reason": f"OpenAI Vision call failed: {type(exc).__name__}: {exc}"[:500]}
