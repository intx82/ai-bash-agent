# MCP agent: visual feedback and FreeCAD geometry verification

This change adds **no GTK screenshot preview**. Your FreeCAD window remains the visualization.

## Installation

Copy these files beside your current `agent.py`:

- `agent.py` — updated turn handler and system prompt
- `llm_backends.py` — multimodal chat message support, normalized OpenAI responses
- `agent_vision.py` — new optional visual feedback component
- `freecad_verify.py` — new FreeCAD geometry readback and validation

Leave `mcp_manager.py`, `mcp_bridge.py`, `mcp_server.py`, `mcp_servers.json`, and `agent_gui.py` unchanged.

## Visual feedback: three modes

Images returned by `freecad.get_view` are already saved by `mcp_bridge.py`. The new code makes their use explicit:

- `AGENT_VISION_MODE=off` (default): image is saved, **not analyzed**. The LLM is told not to claim it inspected the image.
- `AGENT_VISION_MODE=main`: send the saved image as an OpenAI-compatible `image_url` data URI in **one next inference call**. Only works if the model served by your main backend supports vision, with the appropriate image projector loaded in llama.cpp. The data URI is removed from stored conversation history immediately afterward.
- `AGENT_VISION_MODE=sidecar`: send the image to a **separate** vision-capable OpenAI-compatible `/v1/chat/completions` endpoint, then return its textual description to your normal text-only Qwen coding model.

Your current **Qwen3-Coder-Next** setup is text-only. Do not set `main` unless you switch to a vision-capable model and configure llama.cpp's multimodal projector.

### Sidecar example (recommended for text-only Qwen)

```bash
export AGENT_VISION_MODE=sidecar
export AGENT_VISION_URL=http://127.0.0.1:8090/v1/chat/completions
export AGENT_VISION_MODEL='your-vision-capable-model-name'
# Optional if the endpoint requires authentication:
export AGENT_VISION_API_KEY='...'
python3 agent.py
```

The endpoint at 8090 is just an example: **you must start your own vision model server**. Model must support image input. HTTPS is required for a remote, non-loopback sidecar, so screenshots are not sent over the internet unencrypted. Sidecar requests have a 90-second response timeout.

When visual feedback is not available, `TOOL_RESULT.vision.status` is `not_analyzed` or `unavailable`, never a fictitious visual observation. When supported, the model can explicitly call `freecad.get_view` to obtain a screenshot. The `--only-text-feedback` flag suppresses automatic screenshots from mutation tools but does not suppress `get_view`.

## Automatic FreeCAD geometry verification

After each successful `freecad.create_object` or `freecad.edit_object`, the agent automatically:

1. Calls `freecad.get_object` with `include_screenshot=false` to confirm the object exists.
2. If available, calls `freecad.execute_code` with an internal **fixed, read-only** query on FreeCAD's GUI thread. The query fetches object type, requested numeric properties, `Shape.isNull()`, `Shape.isValid()`, volume in mm³, number of solids and axis-aligned bounding-box size. Object/document names are safely quoted.
3. Compares the actual numeric properties with those requested (small floating-point tolerance). For `Part::Box`, also checks expected volume and exactly one solid when metrics are returned.

The tool result includes a `verification` object with `status`:

- `verified` — all requested numeric fields are confirmed and the shape reports valid; `Part::Box` volume and solid count also agree.
- `mismatch` — actual dimensions or volume differ from request (outer tool result `ok=false`).
- `failed` — actual shape is null, invalid, missing or malformed (outer `ok=false`).
- `unverified` — state or shape information could not be obtained or was incomplete. The model must **not** assert geometric correctness.
- `not_run` — creation/editing itself failed.

A valid CAD shape is **not a guarantee of engineering correctness**. For complex assemblies, collision, constraints, fillets, tolerances, manufacturability, or mass properties, use additional domain-specific checks.

The verification hook currently applies to `create_object` and `edit_object`. Direct `execute_code` geometry modifications are **not** automatically verified, because there is no general way to infer which documents/objects were touched.

For the 50 × 50 × 10 mm box, expected solid volume is **25,000 mm³**. Full 3D bounding-box metrics are included as diagnostics, although axis-aligned extents should not be treated as parametric Length/Width/Height for rotated objects.

## Tests

```bash
python3 test_vision_geometry.py
```

The tests are offline simulations of MCP, FreeCAD, llama.cpp and vision API responses. They do **not** validate your live FreeCAD addon or actual vision model and do not need `mcp` or FreeCAD installed on the test host.
