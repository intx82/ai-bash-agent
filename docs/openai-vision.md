# OpenAI vision MCP server for ai-bash-agent

The main Qwen/llama.cpp coding model remains unchanged. An independent **`openai_vision` MCP server** calls the OpenAI Responses API for screenshot inspection. No GTK preview or separate local vision model is needed.

## Files to install

Copy into the same directory as `agent.py`:

- `mcp_openai_vision.py` (new server)
- `mcp_servers.json` (adds enabled, optional `openai_vision` server to existing `local` and `freecad` entries)
- `agent.py` (one system prompt addition; existing geometry verification remains unchanged)

Do not replace your existing `mcp_manager.py`, `mcp_bridge.py`, `mcp_server.py`, `llm_backends.py` or `freecad_verify.py`.

## Install dependencies and set the key

Activate the existing Python venv:

```sh
python3 -m pip install --upgrade 'mcp' 'openai' 'Pillow'
export OPENAI_API_KEY='YOUR_KEY'
# Optional model override. gpt-5.4-mini accepts image inputs.
export MCP_OPENAI_VISION_MODEL='gpt-5.4-mini'
# Turn off the old sidecar path to avoid its missing-URL message.
export AGENT_VISION_MODE=off
python3 agent_gui.py
```

These are OpenAI **API** calls and incur usage charges separately from ChatGPT subscriptions. The image is uploaded **only when the `openai_vision.describe_image` MCP tool executes**. The multi-MCP manager asks for confirmation before running that tool. `image_info` and `compare_regions` are local-only tools, allowlisted as read-only.

`OPENAI_API_KEY` is read from the process environment and is **not** placed in `mcp_servers.json`. The server still starts and lists its tools without the key, but an API-backed call returns an actionable error.

## Example FreeCAD workflow

1. Ask the agent: "Create a 50 × 50 × 10 mm box in FreeCAD, verify its dimensions, then describe the isometric screenshot and estimate visible sizes in pixels."
2. Agent calls `freecad.create_document` and `freecad.create_object` (with existing `freecad_verify.py` verification).
3. Agent calls `freecad.get_view`, which saves a screenshot at e.g. `/tmp/ai-bash-agent-XXXX/mcp_images/mcp-view-YYYY.png`.
4. Agent calls `openai_vision.describe_image` with that **exact** image path. The manager asks you to approve sending the screenshot to OpenAI.
5. Agent receives a description and an array of object rectangles in pixels, plus width, height, area and center-distance ratios computed from **estimated** rectangles.

Example tool call (after `get_view`):

```json
{
  "type": "tool",
  "tool": "mcp",
  "name": "openai_vision.describe_image",
  "arguments": {
    "image_path": "/tmp/ai-bash-agent-XXXX/mcp_images/mcp-view-YYYY.png",
    "question": "Identify the box and other objects. Compare screen-projected widths and distances.",
    "detail": "high",
    "max_objects": 8
  }
}
```

Example output fields:

```json
{
  "ok": true,
  "width_px": 800,
  "height_px": 600,
  "description": "...",
  "objects": [
    {"name": "box", "bbox_px": [120, 80, 520, 390], "width_px": 400,
     "height_px": 310, "area_bbox_px2": 124000,
     "bbox_source": "model_estimate", "confidence": "medium"}
  ],
  "relative_sizes": [],
  "measurement_note": "..."
}
```

The above is illustrative only, not an actual image analysis result.

## Tools

- **`openai_vision.describe_image`**: Sends a PNG/JPEG/WEBP image to OpenAI and returns description, objects, *estimated* bounding boxes mapped to original pixel dimensions, and relative pixel sizes/offsets. Supports optional `question`, `detail` (`low`, `high`, `auto`, `original` depending on model) and `max_objects` (1–12).
- **`openai_vision.image_info`**: Gives *exact* source file pixel width, height and format; does **not** call OpenAI.
- **`openai_vision.compare_regions`**: Given **user-supplied** rectangles (`bbox_px: [x1,y1,x2,y2]`), computes their sizes, centers, distance and ratios with deterministic arithmetic; does **not** call OpenAI. This is precise only with respect to the supplied rectangles, not a guarantee the rectangle matches the object's outline.

### Important measurement limitation

A vision language model is **not** a precision object detector. The pixel dimensions of the *image* are exact; object bounding boxes and all derived pixel ratios from `describe_image` are **approximate model estimates**. Screen-projected distances are also affected by the FreeCAD camera angle and perspective. For exact CAD dimensions in millimeters, keep using FreeCAD's object properties and shape verification, not pixel ratios.

## Filesystem access

The server only opens image paths inside the agent workspace (`${WORKSPACE}`, configured as `MCP_VISION_ROOT`). The FreeCAD MCP screenshot folder is inside that workspace. To read images from another location, optionally set a colon-separated allowlist *before launching the agent*:

```sh
export MCP_VISION_EXTRA_ROOTS=/home/intl/wrk:/home/intl/Pictures
```

Images must be PNG/JPEG/WEBP, at most 8 MiB and at most 24 million pixels. Paths outside the allowed directories are rejected, including paths redirected outside using symlinks.

## Test

Offline mock tests (no API calls) are included:

```sh
python3 test_openai_vision_mcp.py
```

The mocked tests verify tool registration, path restrictions, 2D pixel calculations, image size validation, request formatting for the Responses API and missing-key errors. They do **not** confirm your account's API permissions or live FreeCAD screenshots.

## Troubleshooting

After installing the files, restart the agent and ask it to call `manager.list_servers`. `openai_vision` should be `connected`. If it is `failed`, check `manager.list_servers` error details (missing `mcp`, `Pillow`, launch path etc.). If it is connected but `describe_image` fails, check `OPENAI_API_KEY`, model availability and internet access.

The old `AGENT_VISION_MODE=sidecar` path is separate and no longer needed for this setup.
