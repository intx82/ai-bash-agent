# FreeCAD Python API guide for the MCP agent

This is an operational reference for the *running FreeCAD Python interpreter*, not an exhaustive API manual. FreeCAD versions and workbenches differ. When an unfamiliar object or method is involved, inspect it in FreeCAD before relying on it.

## QUICK REFERENCE — automatically supplied to the agent when FreeCAD is connected

- **Document objects are not Part topology objects.** `doc.getObject('Box')` is a document object; `doc.getObject('Box').Shape` is its geometric/topological shape; `.Shape.Edges[i]` is a `Part.Edge`. Do not assume they implement identical methods.
- **NEVER call `edge.isInstanceOf(...)`** on a `Part.Edge` (it does not provide this method). For topology, use `edge.ShapeType == 'Edge'`, `face.ShapeType == 'Face'`, etc. For the underlying edge curve, inspect `type(edge.Curve).__name__`; `edge.Curve` may represent a line, circle, B-spline, or another geometry type. Use `dir(edge)` / `dir(edge.Curve)` before calling uncertain methods.
- For **FreeCAD document objects**, use `obj.TypeId` and, if available, `obj.isDerivedFrom('Part::Feature')`; this is different from testing a `Part.Edge`. Prefer `getattr(obj, 'TypeId', None)` and runtime introspection if unsure.
- **Shape validation**: after `doc.recompute()`, read `shape.isNull()`, `shape.isValid()`, `shape.ShapeType`, `len(shape.Solids)`, `shape.Volume` and `shape.BoundBox` (as applicable). Empty/2D shapes may legitimately have zero volume; do not equate zero volume with invalid geometry.
- **Bounds**: `shape.BoundBox.XLength`, `YLength`, `ZLength` are the axis-aligned bounding extents in model units (normally mm), **not** screenshot pixel measurements.
- **Names**: `obj.Name` is the internal identifier, whereas `obj.Label` is displayed in the tree. Fetch via `doc.getObject(name)`; don't assume Label equals Name.
- **API failures**: On `AttributeError` or `TypeError`, don't repeat the same code. Inspect the object with a short read-only `freecad.execute_code` call using `include_screenshot=false`. Then correct the method/arguments and retry once. Check tool result `isError` and error text rather than reporting success.
- **Execution**: `freecad.execute_code` runs on the GUI thread for document edits and `Gui` calls. `execute_code_async` is only for off-thread computations; document writes and GUI operations must be handed to the GUI thread using the MCP addon's `commit()` helper. Use headless mode for long crash-prone geometry operations.
- **Prefer native FreeCAD tools** (`create_document`, `create_object`, `get_object`, `edit_object`) for straightforward operations over custom `execute_code`. After changes, read back geometry; don't infer correctness from an image alone.

**Small read-only inspection example** (`freecad.execute_code`, `include_screenshot=false`):

```python
import FreeCAD as App
import Part

doc = App.ActiveDocument
obj = doc.getObject('Box') if doc else None
if obj is not None and hasattr(obj, 'Shape'):
    shape = obj.Shape
    print('doc object:', type(obj).__name__, getattr(obj, 'TypeId', None))
    print('shape:', shape.ShapeType, 'valid:', shape.isValid())
    for i, edge in enumerate(shape.Edges[:5], 1):
        print(i, 'topology:', edge.ShapeType,
              'curve:', type(edge.Curve).__name__,
              'length:', edge.Length)
```

For unknown methods, a bounded inspection is safer than guessing:

```python
print(type(edge).__name__)
print([name for name in dir(edge) if 'type' in name.lower() or 'curve' in name.lower()])
print(type(edge.Curve).__name__)
```

### Visibility and occlusion-aware inspection

- `obj.ViewObject.Visibility` is a GUI/display property. `obj.ViewObject.hide()` and `.show()` are supported for GUI document objects. Visibility does not remove geometry or change a boolean cut. Use exact `obj.Name`, not display `Label`.
- A screenshot of the assembled scene may be **occluded**: the lid hides the internal mounting bosses or USB-C cutout. A visual model cannot confirm geometry it cannot see.
- If advertised by the live MCP catalog, use `freecad_scene.list_parts(doc_name)` and `freecad_scene.inspect_part(doc_name, object_name, question, isolate=true)` to get part-specific OpenAI Vision evidence with automatic visibility restoration.
- `freecad_scene.hide_parts` and `freecad_scene.show_parts` persistently toggle named part visibility for manual inspection. They do not change CAD geometry.
- `inspect_part` never provides exact millimeter dimensions; query FreeCAD object's geometry separately.

<!-- END PROMPT QUICK REFERENCE -->

## More examples

### Creating and validating a box

```python
import FreeCAD as App

doc = App.newDocument('Example')
box = doc.addObject('Part::Box', 'Box')
box.Length = 50
box.Width = 50
box.Height = 10
doc.recompute()
shape = box.Shape
print('type', shape.ShapeType)
print('null', shape.isNull(), 'valid', shape.isValid())
print('solids', len(shape.Solids), 'volume mm^3', shape.Volume)
print('size mm', shape.BoundBox.XLength,
      shape.BoundBox.YLength, shape.BoundBox.ZLength)
```

Expected box volume = 50 * 50 * 10 = 25000 mm^3. Shape `BoundBox` is axis-aligned and may differ from intrinsic feature dimensions after rotation.

### Distinguish geometry from topology

```python
import FreeCAD as App
import Part

obj = App.ActiveDocument.getObject('Box')
shape = obj.Shape
print('document type:', obj.TypeId)
print('topology type:', shape.ShapeType)
for idx, edge in enumerate(shape.Edges, 1):
    curve = edge.Curve
    print(idx, edge.ShapeType, type(curve).__name__, edge.Length)
```

Do not assume an edge is straight based on its bounding box or endpoint count: curves can have arbitrary geometry. Check `type(edge.Curve).__name__` on the *actual running FreeCAD version*, then use suitable methods for that class. There is no general `Part.Edge.isInstanceOf` method.

### Shape vs document object

```python
obj = App.ActiveDocument.getObject('Box')
print(obj.Name, obj.Label, obj.TypeId)
print(obj.isDerivedFrom('Part::Feature'))
print(obj.Shape.ShapeType)
```

`isDerivedFrom` is for FreeCAD document objects. Do not apply document-object predicates to `obj.Shape`, `obj.Shape.Edges`, `obj.Shape.Faces`, or `obj.Shape.Vertexes`.

### Editing via shape operations

```python
import Part
import FreeCAD as App

doc = App.ActiveDocument
base = Part.makeBox(50, 50, 10)
tool = Part.makeCylinder(2, 10)
result = base.cut(tool)
if result.isNull() or not result.isValid():
    raise RuntimeError('Invalid boolean result')
feature = doc.addObject('Part::Feature', 'CutBox')
feature.Shape = result
doc.recompute()
```

Check the location/orientation of a cutting tool before assuming it intersects the base. For arbitrary shapes, check solids, volume and bounding box after operations. Avoid hardcoded references to Edge1/Edge2 across topology-changing operations: topological naming may change.

### Save/export distinctions

```python
doc.saveAs('/path/to/model.FCStd')  # native FreeCAD document
Part.export([feature], '/path/to/model.step')  # STEP geometry export
```

### Diagnostics after a FreeCAD MCP tool error

Use `freecad.execute_code` in a **separate read-only request** (subject to the same permission policy) with `include_screenshot=false` and inspect `type`, `dir`, `getattr` on the actual target. Do not run speculative mutations repeatedly. Before retrying, inspect any partially created objects; tool execution may have succeeded partway through before an exception was raised.

## Authoritative references

- FreeCAD Part scripting: https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Part_scripting.md
- FreeCAD Part TopoShape: https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Part_TopoShape.md
- FreeCAD topological data scripting: https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Topological_data_scripting.md
- FreeCAD Python scripting tutorial: https://github.com/FreeCAD/FreeCAD-documentation/blob/main/wiki/Python_scripting_tutorial.md
- FreeCAD MCP tools: https://github.com/neka-nat/freecad-mcp/blob/main/docs/tools.md
- FreeCAD MCP execution modes: https://github.com/neka-nat/freecad-mcp/blob/main/docs/execution.md

Online API guides may refer to older FreeCAD versions. When version behavior matters, prefer the running interpreter's actual `type`, `dir`, `hasattr`, and `help()` (with output size kept bounded).
