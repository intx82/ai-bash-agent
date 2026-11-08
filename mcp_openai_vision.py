#!/usr/bin/env python3
"""OpenAI image-understanding MCP server for the multi-MCP agent.

- describe_image: visual content + *estimated* boxes and pixel relationships.
- image_info: exact file dimensions without an OpenAI request.
- compare_regions: exact arithmetic on provided rectangles.

All image uploads require user confirmation in the agent's MCP manager.
No images, API keys, or base64 data are returned in tool results.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("openai-vision")

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_OBJECTS = 12
VALID_DETAILS = ("low", "high", "auto", "original")
SUPPORTED_FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


class VisionInputError(ValueError):
    pass


def _roots() -> List[Path]:
    # The agent passes its isolated workspace as MCP_VISION_ROOT. Optional extra
    # allowlisted directories can be added for user images, separated by os.pathsep.
    configured = os.environ.get("MCP_VISION_ROOT", os.environ.get("MCP_ROOT", "."))
    additional = os.environ.get("MCP_VISION_EXTRA_ROOTS", "")
    values = [configured, *[s for s in additional.split(os.pathsep) if s]]
    return [Path(value).expanduser().resolve() for value in values]


def _image_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise VisionInputError("image_path must be a non-empty path")
    roots = _roots()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    full = candidate.resolve()
    if not any(full == root or root in full.parents for root in roots):
        raise VisionInputError(
            "Image is outside approved MCP vision directories. "
            "Set MCP_VISION_EXTRA_ROOTS to allow another image directory."
        )
    if not full.is_file():
        raise VisionInputError(f"Not an existing regular image file: {path}")
    size = full.stat().st_size
    if not (0 < size <= MAX_IMAGE_BYTES):
        raise VisionInputError(f"Image must be 1..{MAX_IMAGE_BYTES} bytes (actual {size})")
    return full


def _read_image(path: str, with_bytes: bool = False) -> Dict[str, Any]:
    from PIL import Image, UnidentifiedImageError

    full = _image_path(path)
    try:
        with Image.open(full) as im:
            fmt = im.format
            width, height = im.size
            if fmt not in SUPPORTED_FORMATS:
                raise VisionInputError("Only PNG, JPEG and WEBP are supported")
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise VisionInputError("Image exceeds configured pixel limit")
            if getattr(im, "n_frames", 1) != 1:
                raise VisionInputError("Animated or multipage images are not supported")
            # Decodes the file to verify that it is not corrupt.
            im.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise VisionInputError(f"Cannot decode image: {exc}") from exc
    result: Dict[str, Any] = {
        "ok": True, "path": str(full), "width_px": width,
        "height_px": height, "mime_type": SUPPORTED_FORMATS[fmt],
        "file_bytes": full.stat().st_size,
    }
    if with_bytes:
        result["_data_uri"] = (
            f"data:{SUPPORTED_FORMATS[fmt]};base64,"
            + base64.b64encode(full.read_bytes()).decode("ascii")
        )
    return result


def _pixel_box(norm: List[int], width: int, height: int) -> Optional[List[int]]:
    if not isinstance(norm, (list, tuple)) or len(norm) != 4:
        return None
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in norm):
        return None
    x1, y1, x2, y2 = (max(0, min(1000, float(x))) for x in norm)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1 * width / 1000), round(y1 * height / 1000),
            round(x2 * width / 1000), round(y2 * height / 1000)]


def _region(name: str, box: List[int]) -> Dict[str, Any]:
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    return {"name": name, "bbox_px": box, "width_px": w, "height_px": h,
            "center_px": [round((x1+x2)/2, 1), round((y1+y2)/2, 1)],
            "area_bbox_px2": w*h}


def _relations(regions: List[Dict[str, Any]], max_pairs: int = 36) -> List[Dict[str, Any]]:
    results = []
    for i, a in enumerate(regions):
        for b in regions[i+1:]:
            if len(results) >= max_pairs:
                return results
            dx = round(b["center_px"][0] - a["center_px"][0], 1)
            dy = round(b["center_px"][1] - a["center_px"][1], 1)
            results.append({
                "a": a["name"], "b": b["name"],
                "center_delta_px": [dx, dy],
                "center_distance_px": round(math.hypot(dx, dy), 2),
                "width_ratio_a_to_b": round(a["width_px"] / b["width_px"], 4) if b["width_px"] else None,
                "height_ratio_a_to_b": round(a["height_px"] / b["height_px"], 4) if b["height_px"] else None,
                "bbox_area_ratio_a_to_b": round(a["area_bbox_px2"] / b["area_bbox_px2"], 4) if b["area_bbox_px2"] else None,
            })
    return results


VISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "bbox_1000": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        }, "required": ["name", "description", "bbox_1000", "confidence"],
            "additionalProperties": False}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["description", "objects", "observations", "limitations"],
    "additionalProperties": False,
}


def _openai_describe(image: Dict[str, Any], question: str, detail: str,
                     max_objects: int) -> Dict[str, Any]:
    # Import lazily to let the MCP tools register without API credentials.
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set for the OpenAI vision MCP server")
    from openai import OpenAI
    model = os.environ.get("MCP_OPENAI_VISION_MODEL", "gpt-5.4-mini")
    client = OpenAI(api_key=key, timeout=90.0, max_retries=1)
    instructions = (
        "Describe the image faithfully. Identify up to " + str(max_objects) +
        " prominent objects or visible regions. For each, estimate a 2D axis-aligned "
        "bounding box [left,top,right,bottom] with coordinates on a 0..1000 scale "
        "relative to the original image. Coordinates are ESTIMATES, not computer-vision "
        "measurements. Do not hallucinate hidden or occluded geometry. For 3D perspective "
        "images, report only projected 2D screen bounds, never inferred real-world dimensions. "
        "State uncertainty and avoid precise-sounding claims about location or size. "
        "If objects cannot be isolated, return an empty objects array."
    )
    response = client.responses.create(
        model=model,
        store=False,
        instructions=instructions,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": question},
            {"type": "input_image", "image_url": image["_data_uri"], "detail": detail},
        ]}],
        text={"format": {"type": "json_schema", "name": "image_description",
                         "strict": True, "schema": VISION_SCHEMA}},
        max_output_tokens=1500,
    )
    answer = getattr(response, "output_text", "")
    if not answer:
        raise RuntimeError("OpenAI returned no image description")
    return json.loads(answer)


def describe_image_impl(image_path: str, question: str = "Describe this image and compare visible object sizes and positions.",
                        detail: str = "high", max_objects: int = 8) -> Dict[str, Any]:
    if detail not in VALID_DETAILS:
        raise VisionInputError(f"detail must be one of {VALID_DETAILS}")
    if type(max_objects) is not int or not 1 <= max_objects <= MAX_OBJECTS:
        raise VisionInputError(f"max_objects must be in 1..{MAX_OBJECTS}")
    if not isinstance(question, str) or len(question) > 4000:
        raise VisionInputError("question must be text shorter than 4001 chars")
    image = _read_image(image_path, with_bytes=True)
    description = _openai_describe(image, question, detail, max_objects)
    regions = []
    for item in description.get("objects", [])[:max_objects]:
        if not isinstance(item, dict):
            continue
        bbox = _pixel_box(item.get("bbox_1000", []), image["width_px"], image["height_px"])
        if bbox:
            regions.append({**_region(item["name"], bbox),
                            "description": item.get("description", ""),
                            "confidence": item.get("confidence", "low"),
                            "bbox_source": "model_estimate"})
    image.pop("_data_uri", None)
    return {
        **image, "model": os.environ.get("MCP_OPENAI_VISION_MODEL", "gpt-5.4-mini"),
        "description": description.get("description", ""),
        "objects": regions, "relative_sizes": _relations(regions),
        "observations": description.get("observations", []),
        "limitations": description.get("limitations", []),
        "measurement_note": "Image pixel dimensions are exact. Object bounding boxes and their derived pixel sizes/ratios are model estimates. For CAD dimensions, use FreeCAD object properties, not screenshots.",
    }


@mcp.tool()
def image_info(image_path: str) -> Dict[str, Any]:
    """Get EXACT image width/height (pixels), format and byte size. No OpenAI request."""
    try:
        return _read_image(image_path)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
async def describe_image(image_path: str, question: str = "Describe the visible objects and their relative sizes in pixels.",
                         detail: str = "high", max_objects: int = 8) -> Dict[str, Any]:
    """Use OpenAI vision to describe an image and estimate object bounding boxes, pixel sizes, relative positions and ratios. Boundaries are approximate. Sends the image to OpenAI; ask user approval."""
    try:
        return await asyncio.to_thread(describe_image_impl, image_path, question, detail, max_objects)
    except Exception as exc:
        # Don't return image data, API keys or whole provider errors to the model.
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}


@mcp.tool()
def compare_regions(image_path: str, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """EXACT pixel arithmetic on USER-SUPPLIED rectangles [{name,bbox_px:[left,top,right,bottom]},...]. No AI inference or API call. Compare widths, heights, distances and ratios."""
    try:
        image = _read_image(image_path)
        if not isinstance(regions, list) or not (2 <= len(regions) <= MAX_OBJECTS):
            raise VisionInputError(f"Provide 2..{MAX_OBJECTS} rectangle objects")
        measured = []
        for obj in regions:
            if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
                raise VisionInputError("Every region needs a string name")
            box = obj.get("bbox_px")
            if not isinstance(box, list) or len(box) != 4 or any(type(x) is not int for x in box):
                raise VisionInputError("bbox_px must be four integer pixel coordinates")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image["width_px"] and 0 <= y1 < y2 <= image["height_px"]):
                raise VisionInputError("Region coordinates must be inside the image")
            measured.append(_region(obj["name"], box))
        return {**image, "regions": measured, "relative_sizes": _relations(measured),
                "measurement_note": "Pixel arithmetic is exact for the supplied rectangles; no claim is made that the rectangles match object boundaries."}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
