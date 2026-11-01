#!/usr/bin/env python3
"""Read-only FreeCAD geometry verification for object creation and editing.

Runs only after an approved FreeCAD mutation. It never changes the document.
The independent get_object readback plus an optional fixed-code geometry probe
are separate from the creation response, avoiding unsupported success claims.
"""
from __future__ import annotations

import base64
import json
import math
import re
from typing import Any, Dict, Optional


_MARKER = "AGENT_GEOMETRY_CHECK="


def _is_error(res: Any) -> bool:
    if not isinstance(res, dict):
        return True
    if res.get("isError") is True or res.get("ok") is False:
        return True
    s = res.get("structured")
    if isinstance(s, dict):
        x = s.get("result", s)
        if isinstance(x, dict) and (x.get("ok") is False or x.get("success") is False):
            return True
    for t in res.get("text", []):
        try:
            x = json.loads(t)
            if isinstance(x, dict) and (x.get("ok") is False or x.get("success") is False):
                return True
        except (TypeError, ValueError):
            continue
    return False


def response_failed(res: Any) -> bool:
    """MCP protocol and FreeCAD operation failures, including success=false."""
    return _is_error(res)


def _find_data(res: Dict[str, Any]) -> Dict[str, Any]:
    """Interpret structured MCP tool results and JSON-encoded text blocks."""
    s = res.get("structured")
    if isinstance(s, dict):
        x = s.get("result", s)
        if isinstance(x, dict):
            return x
    for t in res.get("text", []):
        try:
            x = json.loads(t)
            if isinstance(x, dict):
                return x.get("result", x) if isinstance(x.get("result", x), dict) else x
        except (TypeError, ValueError):
            continue
    return {}


def _readback_properties(data: Any, requested: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort extraction across freecad-mcp serialization versions."""
    if not isinstance(data, dict):
        return {}
    containers = [data]
    for field in ("object", "properties", "Properties", "data", "result"):
        if isinstance(data.get(field), dict):
            containers.append(data[field])
            child = data[field]
            if isinstance(child.get("properties"), dict):
                containers.append(child["properties"])
            if isinstance(child.get("Properties"), dict):
                containers.append(child["Properties"])
    result = {}
    for key in requested:
        for container in containers:
            if key in container:
                value = container[key]
                if isinstance(value, dict):
                    value = value.get("value", value.get("Value", value))
                result[key] = value
                break
    return result


def _numeric(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    if isinstance(v, str):
        # FreeCAD serializations sometimes represent quantities as '50.00 mm'.
        m = re.fullmatch(r"\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(?:mm)?\s*", v)
        if m:
            return float(m.group(1))
    return None


def _probe_code(doc_name: str, obj_name: str, keys: list[str]) -> str:
    # Only repr-quoted strings enter the executed script. No LLM-generated code.
    return '''import FreeCAD as App
import json as _json
import base64 as _b64
_d = App.getDocument(''' + repr(doc_name) + ''')
_o = _d.getObject(''' + repr(obj_name) + ''') if _d else None
_r = {"exists": _o is not None, "properties": {}, "shape": {}}
if _o:
    _r["name"] = _o.Name
    _r["type_id"] = _o.TypeId
    for _key in ''' + repr(keys) + ''':
        try:
            _v = getattr(_o, _key)
            _r["properties"][_key] = _v.Value if hasattr(_v, "Value") else (float(_v) if isinstance(_v, (int, float)) else str(_v))
        except Exception as _e:
            _r["properties"][_key] = {"error": str(_e)}
    if hasattr(_o, "Shape"):
        try:
            _s = _o.Shape
            _r["shape"] = {"is_null": _s.isNull(), "is_valid": _s.isValid(),
                "volume_mm3": _s.Volume, "solids": len(_s.Solids),
                "bbox_mm": {"x": _s.BoundBox.XLength, "y": _s.BoundBox.YLength,
                            "z": _s.BoundBox.ZLength}}
        except Exception as _e:
            _r["shape"] = {"error": str(_e)}
print("''' + _MARKER + '''" + _b64.b64encode(_json.dumps(_r).encode("utf-8")).decode("ascii"))
'''


def _probe_result(res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = json.dumps({"structured": res.get("structured"), "text": res.get("text")}, ensure_ascii=False)
    m = re.search(re.escape(_MARKER) + r"([A-Za-z0-9+/=]+)", source)
    if not m:
        return None
    try:
        x = json.loads(base64.b64decode(m.group(1), validate=True))
        return x if isinstance(x, dict) else None
    except (ValueError, TypeError):
        return None


def verify_mutation(mcp: Any, tool_name: str, args: Dict[str, Any], mutation_response: Dict[str, Any]) -> Dict[str, Any]:
    """Verify just-created/edited FreeCAD objects using actual model state."""
    if tool_name not in ("freecad.create_object", "freecad.edit_object"):
        return {}
    if _is_error(mutation_response):
        return {"status": "not_run", "reason": "Creation/editing failed"}
    doc = args.get("doc_name")
    obj = args.get("obj_name")
    if not isinstance(doc, str) or not isinstance(obj, str):
        return {"status": "unverified", "reason": "Missing document/object name"}
    data = _find_data(mutation_response)
    if isinstance(data.get("object_name"), str):
        obj = data["object_name"]  # Factories can rename objects.
    requested = args.get("obj_properties") or {}
    if not isinstance(requested, dict):
        requested = {}
    result: Dict[str, Any] = {"status": "unverified", "document": doc, "object": obj,
                              "requested_properties": requested}
    try:
        state = mcp.call_tool("freecad.get_object", {"doc_name": doc,
                               "obj_name": obj, "include_screenshot": False})
    except Exception as exc:
        return {**result, "reason": f"get_object failed: {exc}"}
    if _is_error(state):
        return {**result, "reason": "get_object did not find/read the object", "readback": state}
    result["readback_ok"] = True
    observed = _readback_properties(_find_data(state), requested)

    # get_object is the first, independent state check. Use a fixed, read-only
    # FreeCAD GUI query to obtain actual shape validity and exact quantities.
    if hasattr(mcp, "has_tool") and mcp.has_tool("freecad.execute_code"):
        try:
            probe = mcp.call_tool("freecad.execute_code", {
                "code": _probe_code(doc, obj, list(requested)),
                "include_screenshot": False})
            x = _probe_result(probe)
            if x and not _is_error(probe):
                if not x.get("exists"):
                    return {**result, "status": "failed", "reason": "Object missing in FreeCAD document"}
                observed.update(x.get("properties", {}))
                result["geometry"] = x.get("shape") or {}
                result["type_id"] = x.get("type_id")
            else:
                result["probe_warning"] = "Could not decode FreeCAD shape probe; using get_object readback"
        except Exception as exc:
            result["probe_warning"] = str(exc)

    result["observed_properties"] = observed
    checked: Dict[str, Any] = {}
    mismatches: Dict[str, Any] = {}
    for key, wanted in requested.items():
        expected = _numeric(wanted)
        actual = _numeric(observed.get(key))
        if expected is None:
            continue  # Non-numeric data cannot be safely compared generically.
        if actual is None:
            continue  # Missing observation -> unverified, never assume success.
        checked[key] = {"requested": expected, "actual": actual}
        if not math.isclose(expected, actual, abs_tol=1e-5, rel_tol=1e-7):
            mismatches[key] = checked[key]
    result["checked_properties"] = checked
    if mismatches:
        return {**result, "status": "mismatch", "mismatches": mismatches}

    geometry = result.get("geometry") or {}
    if geometry.get("is_null") is True or geometry.get("is_valid") is False:
        return {**result, "status": "failed", "reason": "Shape is null or invalid"}
    if isinstance(geometry.get("error"), str):
        return {**result, "status": "unverified", "reason": geometry["error"]}
    # An axis-aligned Part::Box has a known non-zero expected solid volume.
    typ = args.get("obj_type") or result.get("type_id")
    if typ == "Part::Box" and all(x in checked for x in ("Length", "Width", "Height")):
        expected_volume = math.prod(checked[k]["requested"] for k in ("Length", "Width", "Height"))
        actual_volume = _numeric(geometry.get("volume_mm3"))
        if actual_volume is not None:
            result["expected_volume_mm3"] = expected_volume
            if not math.isclose(expected_volume, actual_volume, abs_tol=1e-4, rel_tol=1e-6):
                return {**result, "status": "mismatch", "reason": "Volume differs from requested dimensions"}
        if geometry.get("solids") is not None and geometry["solids"] != 1:
            return {**result, "status": "failed", "reason": "Box does not contain exactly one solid"}
    numeric_requested = {k for k, v in requested.items() if _numeric(v) is not None}
    if numeric_requested.issubset(checked) and geometry.get("is_valid") is True and not geometry.get("is_null"):
        # Part::Box must also have measurable, physically consistent volume.
        if typ == "Part::Box" and not isinstance(_numeric(geometry.get("volume_mm3")), float):
            result["reason"] = "Cannot confirm solid volume"
        else:
            result["status"] = "verified"
    else:
        result["reason"] = "Readback incomplete: cannot confirm dimensions and shape validity"
    return result
