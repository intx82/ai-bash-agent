"""Opt-in live FreeCAD -> screenshot -> paid OpenAI Vision smoke test.

Run: .venv/bin/python tests/test_freecad_vision_live.py --live
Requires the configured FreeCAD RPC server and OPENAI_API_KEY.
Creates a new unsaved document and leaves it open for manual inspection.
Normal unittest discovery skips this test; no CAD changes or API charges.
"""
from pathlib import Path
import argparse
import json
import os
import re
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def payload(result):
    """Normalize structured and text-only MCP responses; reject tool failures."""
    if result.get("isError"):
        raise AssertionError(f"MCP failure: {result}")
    data = result.get("structured")
    if data is None:
        for text in result.get("text", []):
            try:
                data = json.loads(text)
                break
            except (ValueError, TypeError):
                continue
    if isinstance(data, dict):
        data = data.get("result", data)
        if isinstance(data, dict) and (data.get("ok") is False or data.get("success") is False):
            raise AssertionError(f"Tool failure: {data}")
    return data


class FreeCADVisionLiveTest(unittest.TestCase):
    live_approved = False

    def test_cube_screenshot_vision(self):
        if not self.live_approved:
            self.skipTest("Live CAD changes and paid image upload require --live approval")
        from mcp_manager import MCPManager
        from freecad_verify import verify_mutation

        manager = MCPManager(str(ROOT))
        try:
            manager.start()
            status = payload(manager.call_tool("manager.list_servers", {}))
            print("MCP connections:", json.dumps(status), flush=True)
            for server in ("freecad", "openai_vision"):
                self.assertIn(server, manager.servers,
                              f"{server} disabled/unavailable: {manager.status()}")
            required = ("freecad.create_document", "freecad.create_object",
                        "freecad.get_object", "freecad.get_view",
                        "openai_vision.image_info", "openai_vision.describe_image")
            for name in required:
                self.assertTrue(manager.has_tool(name), f"Missing discovered tool: {name}")

            def call(tool_name, **arguments):
                result = manager.call_tool(tool_name, arguments)
                payload(result)
                return result

            doc = "VisionCube_" + uuid.uuid4().hex[:10]
            call("freecad.create_document", name=doc)
            print("Created unsaved test document:", doc, flush=True)
            arguments = dict(doc_name=doc, obj_type="Part::Box", obj_name="Cube",
                             obj_properties=dict(Length=20, Width=20, Height=20),
                             include_screenshot=False)
            created = call("freecad.create_object", **arguments)
            verification = verify_mutation(manager, "freecad.create_object", arguments, created)
            print("Geometry verification:", json.dumps(verification), flush=True)
            self.assertEqual(verification.get("status"), "verified", verification)

            screenshot = call("freecad.get_view", view_name="Isometric",
                              focus_object="Cube", width=1024, height=768)
            images = [item for item in screenshot.get("images", []) if item.get("path")]
            self.assertTrue(images, "Screenshot returned no saved image")
            image_path = images[0]["path"]
            self.assertTrue(Path(image_path).is_file(), image_path)
            info = payload(call("openai_vision.image_info", image_path=image_path))
            self.assertGreater(info["width_px"], 0)
            self.assertGreater(info["height_px"], 0)
            print("Screenshot:", image_path, flush=True)

            # One explicit upload only. Do not hint the expected shape in the prompt.
            answer = payload(call("openai_vision.describe_image", image_path=image_path,
                                  question="Identify the main visible 3D solid. Describe its shape and give its bounding box. Do not infer physical dimensions.",
                                  detail="high", max_objects=3))
            print("Vision response:", json.dumps(answer, indent=2), flush=True)
            self.assertIsInstance(answer, dict)
            self.assertTrue(answer.get("model"), "Missing model provenance")
            self.assertIsInstance(answer.get("description"), str)
            self.assertTrue(answer["description"].strip())
            self.assertEqual(answer["width_px"], info["width_px"])
            self.assertEqual(answer["height_px"], info["height_px"])
            self.assertIsInstance(answer.get("objects"), list)
            candidates = []
            for obj in answer["objects"]:
                text = obj.get("name", "") + " " + obj.get("description", "")
                if re.search(r"\b(cube|cuboid|box|block)\b", text, re.I):
                    candidates.append(obj)
            self.assertTrue(candidates, "Vision did not identify a cube/box-like solid")
            obj = candidates[0]
            self.assertIn(obj.get("confidence"), ("high", "medium"))
            box = obj.get("bbox_px")
            self.assertIsInstance(box, list)
            self.assertEqual(len(box), 4)
            self.assertTrue(all(type(x) is int for x in box))
            left, top, right, bottom = box
            self.assertTrue(0 <= left < right <= info["width_px"])
            self.assertTrue(0 <= top < bottom <= info["height_px"])
            self.assertEqual(obj.get("bbox_source"), "model_estimate")
            print("PASS: verified CAD cube, screenshot, and plausible Vision response.", flush=True)
        finally:
            manager.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Approve creating an unsaved CAD document and uploading its screenshot to paid OpenAI Vision")
    options = parser.parse_args()
    FreeCADVisionLiveTest.live_approved = options.live
    unittest.main(argv=[sys.argv[0]], verbosity=2)
