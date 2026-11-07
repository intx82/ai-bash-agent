#!/usr/bin/env python3
"""FreeCAD scene tools composed from the existing FreeCAD and vision MCP servers.

The tools are *virtual MCP manager tools*: no new FreeCAD add-on, socket, or
GUI preview is required.  A scoped inspection reads the visibility snapshot,
isolates a named object, captures the viewport, calls OpenAI Vision with the
LLM's specific question, and restores the visibility in a finally block.
"""
from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_MARKER = "AGENT_SCENE_V22="
_VIEWS = frozenset({
    "Isometric", "Front", "Top", "Right", "Back", "Left", "Bottom",
    "Dimetric", "Trimetric",
})


TOOLS = {
    "list_parts": {
        "description": "List FreeCAD document objects with exact internal names, labels, visibility and approximate shape bounds (mm). Call this before targeting a part by name.",
        "inputSchema": {
            "type": "object", "properties": {"doc_name": {"type": "string"}},
            "required": ["doc_name"], "additionalProperties": False,
        },
    },
    "set_visibility": {
        "description": "Show or hide one or more named FreeCAD objects. This changes the GUI view but not the object's geometry. Names must be exact, as returned by list_parts.",
        "inputSchema": {
            "type": "object", "properties": {
                "doc_name": {"type": "string"},
                "object_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                "visible": {"type": "boolean"},
            }, "required": ["doc_name", "object_names", "visible"], "additionalProperties": False,
        },
    },
    "hide_parts": {
        "description": "Hide specified FreeCAD parts so the user or OpenAI Vision can see components behind them. Visibility stays changed until shown again.",
        "inputSchema": {
            "type": "object", "properties": {
                "doc_name": {"type": "string"},
                "object_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["doc_name", "object_names"], "additionalProperties": False,
        },
    },
    "show_parts": {
        "description": "Show specified previously hidden FreeCAD parts. Does not change their geometry.",
        "inputSchema": {
            "type": "object", "properties": {
                "doc_name": {"type": "string"},
                "object_names": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            }, "required": ["doc_name", "object_names"], "additionalProperties": False,
        },
    },
    "inspect_part": {
        "description": "Inspect EXACTLY one FreeCAD part with OpenAI Vision: temporarily isolate it (hide occluding objects), capture a screenshot focused on the part, ask a precise question, and ALWAYS restore original visibility. The model must choose the target and provide its question. This uploads a screenshot to a paid external API; approval is required. Pixel positions are approximate, not exact CAD dimensions.",
        "inputSchema": {
            "type": "object", "properties": {
                "doc_name": {"type": "string"}, "object_name": {"type": "string"},
                "question": {"type": "string", "minLength": 1},
                "view_name": {"type": "string", "enum": sorted(_VIEWS), "default": "Isometric"},
                "isolate": {"type": "boolean", "default": True},
                "detail": {"type": "string", "enum": ["auto", "low", "high", "original"], "default": "high"},
                "max_objects": {"type": "integer", "minimum": 1, "maximum": 12, "default": 8},
            }, "required": ["doc_name", "object_name", "question"],
            "additionalProperties": False,
        },
    },
}


def _string(value: Any, label: str, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be a nonempty string (up to {limit} characters)")
    if "\x00" in value:
        raise ValueError(f"{label} contains a NUL character")
    return value


def _names(value: Any) -> List[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= 100:
        raise ValueError("object_names must be a list of 1..100 names")
    names = [_string(item, "object name") for item in value]
    if len(set(names)) != len(names):
        raise ValueError("object_names contains duplicates")
    return names


def _result(data: Dict[str, Any], *, error: bool = False) -> Dict[str, Any]:
    return {"structured": {"result": {"ok": not error, **data}}, "text": [], "isError": error}


def _failed(res: Any) -> bool:
    if not isinstance(res, dict) or res.get("ok") is False or res.get("isError") is True:
        return True
    structured = res.get("structured")
    if isinstance(structured, dict):
        value = structured.get("result", structured)
        if isinstance(value, dict) and (value.get("ok") is False or value.get("success") is False):
            return True
    for item in res.get("text", []):
        if isinstance(item, str) and (
            item.lstrip().startswith("Failed to execute code:")
            or item.lstrip().startswith("Error executing code:")
        ):
            return True
    return False


def _decode(res: Any) -> Dict[str, Any]:
    if _failed(res):
        raise RuntimeError(f"FreeCAD execution error: {str(res)[:600]}")
    serialized = json.dumps({
        "structured": res.get("structured"), "text": res.get("text")
    }, ensure_ascii=False)
    m = re.search(re.escape(_MARKER) + r"([A-Za-z0-9+/=]+)", serialized)
    if not m:
        raise RuntimeError("FreeCAD returned no scene probe marker; no result can be verified")
    try:
        value = json.loads(base64.b64decode(m.group(1), validate=True))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("FreeCAD returned an invalid scene probe result") from exc
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise RuntimeError(f"FreeCAD scene operation failed: {value}")
    return value


def _emit_code(body: str) -> str:
    return ("import FreeCAD as App\n"
            "import json as _j\n"
            "import base64 as _b\n"
            "def _emit(_v):\n"
            f"    print({_MARKER!r} + _b.b64encode(_j.dumps(_v).encode('utf-8')).decode('ascii'))\n"
            + body)


def _doc_code(doc_name: str) -> str:
    return (f"_doc = App.listDocuments().get({doc_name!r})\n"
            "if _doc is None:\n"
            "    raise ValueError('FreeCAD document not found')\n")


def _snapshot_code(doc_name: str) -> str:
    return _emit_code(_doc_code(doc_name) + """_items = []
for _o in _doc.Objects:
    _vo = getattr(_o, 'ViewObject', None)
    if _vo is None or not hasattr(_vo, 'Visibility'):
        continue
    _record = {'name': _o.Name, 'label': str(_o.Label),
               'visible': bool(_vo.Visibility), 'type_id': str(_o.TypeId)}
    _shape = getattr(_o, 'Shape', None)
    if _shape is not None:
        try:
            _record['bbox_mm'] = {'x': float(_shape.BoundBox.XLength),
                                  'y': float(_shape.BoundBox.YLength),
                                  'z': float(_shape.BoundBox.ZLength)}
            _record['shape_valid'] = bool(_shape.isValid()) if not _shape.isNull() else False
        except Exception:
            pass
    _items.append(_record)
_emit({'ok': True, 'parts': _items})
""")


def _set_code(doc_name: str, visibility: Dict[str, bool]) -> str:
    return _emit_code(_doc_code(doc_name) + f"_requested = {visibility!r}\n" + """_targets = {}
for _name in _requested:
    _o = _doc.getObject(_name)
    if _o is None:
        raise ValueError('Missing FreeCAD object: ' + _name)
    _vo = getattr(_o, 'ViewObject', None)
    if _vo is None or not hasattr(_vo, 'Visibility'):
        raise ValueError('Object has no visibility property: ' + _name)
    _targets[_name] = _vo
_old = {n: bool(v.Visibility) for n, v in _targets.items()}
try:
    for _name, _vo in _targets.items():
        _vo.Visibility = bool(_requested[_name])
except BaseException:
    for _name, _vo in _targets.items():
        try:
            _vo.Visibility = _old[_name]
        except Exception:
            pass
    raise
_emit({'ok': True, 'before': _old,
       'after': {n: bool(v.Visibility) for n, v in _targets.items()}})
""")


def _part_children(doc: str, target: str) -> str:
    # Keep linked subobjects of the selected PartDesign/Group object visible if
    # already visible. A hidden child is not forcibly exposed.  OutList is
    # evaluated in FreeCAD, not via Python-side labels/guesswork.
    return _emit_code(_doc_code(doc) + f"_target = _doc.getObject({target!r})\n" + """if _target is None:
    raise ValueError('Selected FreeCAD object not found')
_keep = {_target.Name}
_force_show = {_target.Name}
_todo = list(getattr(_target, 'OutList', []))
while _todo:
    _child = _todo.pop()
    if _child.Name in _keep:
        continue
    _keep.add(_child.Name)
    _todo.extend(getattr(_child, 'OutList', []))
# An object inside a PartDesign Body / App::Part container may remain
# invisible if its parent is hidden. Include only container-like ancestors,
# not all dependency links (which could obscure the target).
_todo = list(getattr(_target, 'InList', []))
_seen_parents = set()
while _todo:
    _parent = _todo.pop()
    if _parent.Name in _seen_parents:
        continue
    _seen_parents.add(_parent.Name)
    if str(getattr(_parent, 'TypeId', '')) in (
        'App::Part', 'App::DocumentObjectGroup', 'PartDesign::Body',
        'App::LinkGroup'
    ):
        _keep.add(_parent.Name)
        _force_show.add(_parent.Name)
        _todo.extend(getattr(_parent, 'InList', []))
_emit({'ok': True, 'keep': sorted(_keep), 'force_show': sorted(_force_show)})
""")


class FreeCADSceneTools:
    """Composition layer called synchronously by MCPManager."""

    def __init__(self, manager: Any, workspace: str):
        self.manager = manager
        self.workspace = Path(workspace).resolve()
        self.image_dir = (self.workspace / "mcp_images").resolve()

    def _execute(self, code: str) -> Dict[str, Any]:
        response = self.manager.call_tool("freecad.execute_code", {
            "code": code, "include_screenshot": False,
        })
        return _decode(response)

    def _snapshot(self, doc_name: str) -> Dict[str, Any]:
        return self._execute(_snapshot_code(doc_name))

    def _set_visibility(self, doc: str, states: Dict[str, bool]) -> Dict[str, Any]:
        if not states:
            return {"ok": True, "before": {}, "after": {}}
        return self._execute(_set_code(doc, states))

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        doc = _string(arguments.get("doc_name"), "doc_name")
        if name == "list_parts":
            return _result(self._snapshot(doc))
        if name in ("set_visibility", "hide_parts", "show_parts"):
            names = _names(arguments.get("object_names"))
            if name == "set_visibility":
                visible = arguments.get("visible")
                if type(visible) is not bool:
                    raise ValueError("visible must be a boolean")
            else:
                visible = name == "show_parts"
            applied = self._set_visibility(doc, {x: visible for x in names})
            return _result({"doc_name": doc, "visibility": applied})
        if name == "inspect_part":
            return self._inspect_part(doc, arguments)
        raise ValueError(f"Unknown FreeCAD scene tool: {name}")

    def _inspect_part(self, doc: str, args: Dict[str, Any]) -> Dict[str, Any]:
        obj_name = _string(args.get("object_name"), "object_name")
        question = _string(args.get("question"), "question", 4000)
        view = args.get("view_name", "Isometric")
        if view not in _VIEWS:
            raise ValueError(f"Unsupported FreeCAD view: {view!r}")
        isolate = args.get("isolate", True)
        if type(isolate) is not bool:
            raise ValueError("isolate must be a boolean")
        detail = args.get("detail", "high")
        if detail not in ("auto", "low", "high", "original"):
            raise ValueError("detail must be auto, low, high or original")
        max_objects = args.get("max_objects", 8)
        if type(max_objects) is not int or not 1 <= max_objects <= 12:
            raise ValueError("max_objects must be 1..12")
        if not self.manager.has_tool("openai_vision.describe_image"):
            raise RuntimeError("OpenAI Vision MCP is not connected")

        snap = self._snapshot(doc)
        parts = snap.get("parts", [])
        if not isinstance(parts, list):
            raise RuntimeError("FreeCAD visibility snapshot missing parts")
        original = {x["name"]: x["visible"] for x in parts
                    if isinstance(x, dict) and isinstance(x.get("name"), str)
                    and type(x.get("visible")) is bool}
        if obj_name not in original:
            raise ValueError(f"FreeCAD object {obj_name!r} not found or has no GUI visibility; call freecad_scene.list_parts")

        target_record = next(x for x in parts if x.get("name") == obj_name)
        keep = {obj_name}
        force_show = {obj_name}
        if isolate:
            descendants = self._execute(_part_children(doc, obj_name))
            keep.update(descendants["keep"])
            force_show.update(descendants["force_show"])
        new_state = {name: (True if name in force_show else old if name in keep else False)
                     for name, old in original.items()} if isolate else {obj_name: True}
        changed = {name: value for name, value in new_state.items()
                   if original[name] != value}
        restore = {name: original[name] for name in changed}
        restored = False
        restore_error: Optional[str] = None
        report: Dict[str, Any] = {"document": doc, "object_name": obj_name,
            "cad_object": {key: value for key, value in target_record.items()
                           if key in ("name", "label", "type_id", "bbox_mm", "shape_valid")},
            "question": question, "view_name": view, "isolation": isolate,
            "temporarily_hidden": [n for n, v in changed.items() if not v],
            "temporarily_shown": [n for n, v in changed.items() if v],
            "visibility_restored": False}
        failure: Optional[BaseException] = None
        try:
            # The saved snapshot predates any GUI changes and enables recovery
            # even if the adjustment tool fails after partially applying.
            if changed:
                self._set_visibility(doc, changed)
            result = self.manager.call_tool("freecad.get_view", {
                "view_name": view, "focus_object": obj_name,
            })
            if _failed(result):
                raise RuntimeError(f"FreeCAD get_view failed: {str(result)[:500]}")
            images = result.get("images", []) if isinstance(result, dict) else []
            candidates = [x for x in images if isinstance(x, dict)
                          and isinstance(x.get("path"), str)]
            if not candidates:
                raise RuntimeError("FreeCAD get_view returned no saved screenshot")
            path = Path(candidates[0]["path"]).resolve(strict=True)
            if not path.is_relative_to(self.image_dir) or not path.is_file():
                raise RuntimeError("FreeCAD screenshot is outside the MCP image directory")
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                raise RuntimeError("Unsupported screenshot extension")
            if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
                raise RuntimeError("Screenshot is empty or exceeds 8 MiB")
            report["image_path"] = str(path)
            instruction = (
                f"Inspect the FreeCAD part with internal name {obj_name!r} in document {doc!r}. "
                f"The screenshot was captured with isolate={isolate}; other parts may be hidden. "
                "Describe only visible evidence, explicitly state uncertain or occluded areas. "
                "Report image-pixel positions/ratios only as estimates; never infer exact CAD dimensions. "
                "Targeted question: " + question
            )
            vision = self.manager.call_tool("openai_vision.describe_image", {
                "image_path": str(path), "question": instruction,
                "detail": detail, "max_objects": max_objects,
            })
            if _failed(vision):
                raise RuntimeError(f"OpenAI Vision failed: {str(vision)[:500]}")
            structured = vision.get("structured") or {}
            payload = structured.get("result", structured) if isinstance(structured, dict) else {}
            if not isinstance(payload, dict) or not isinstance(payload.get("description"), str):
                raise RuntimeError("OpenAI Vision returned no structured description")
            if payload.get("ok") is False:
                raise RuntimeError(f"OpenAI Vision returned error: {payload.get('error')}")
            report["vision"] = payload
            report["status"] = "analyzed"
        except Exception as exc:
            failure = exc
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"[:650]
        finally:
            try:
                if restore:
                    self._set_visibility(doc, restore)
                restored = True
            except Exception as exc:
                restore_error = f"{type(exc).__name__}: {exc}"[:650]
            report["visibility_restored"] = restored
            if restore_error:
                report["restoration_error"] = restore_error
                report["status"] = "failed"
                report["recovery_visibility"] = restore

        return _result(report, error=bool(failure or restore_error))
