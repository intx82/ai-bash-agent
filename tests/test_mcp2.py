"""MCP 2.x regression tests for the active application.
Run from project root with unittest discovery. No paid APIs or real CAD.
"""
import asyncio
import base64
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp_bridge import MCPBridge, MCPBridgeConfig
from mcp.types import CallToolResult, TextContent, ImageContent

ROOT = Path(__file__).resolve().parent
SERVER_ROOT = ROOT.parent
FIXTURES = ROOT / "fixtures"


class ResultTests(unittest.TestCase):
    def test_result_contract_and_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = MCPBridge(MCPBridgeConfig(image_dir=tmp))
            result = CallToolResult(structured_content={"result": {"ok": False}},
                is_error=True, content=[TextContent(type="text", text="failed"),
                ImageContent(type="image", mime_type="image/png",
                             data=base64.b64encode(b"image").decode())])
            out = asyncio.run(bridge._normalize_result(result))
            self.assertEqual(set(out), {"structured", "text", "images", "isError"})
            self.assertTrue(out["isError"])
            self.assertEqual(out["structured"], {"result": {"ok": False}})
            self.assertEqual(Path(out["images"][0]["path"]).read_bytes(), b"image")

    def test_pagination_and_loop_guard(self):
        bridge = MCPBridge(MCPBridgeConfig())
        tool = SimpleNamespace(name="read", description=None, input_schema={"type": "object"})
        bridge._client = SimpleNamespace(list_tools=AsyncMock(side_effect=[
            SimpleNamespace(tools=[tool], next_cursor="a"),
            SimpleNamespace(tools=[], next_cursor=None)]))
        self.assertEqual(asyncio.run(bridge._list_tools())[0]["inputSchema"], {"type": "object"})
        bridge._client.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[], next_cursor="a"))
        with self.assertRaisesRegex(RuntimeError, "repeated"):
            asyncio.run(bridge._list_tools())

    def test_closed_connection(self):
        with self.assertRaisesRegex(RuntimeError, "not running"):
            MCPBridge(MCPBridgeConfig()).list_tools()


class LocalIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bridge = MCPBridge(MCPBridgeConfig(command=sys.executable,
            args=[str(SERVER_ROOT / "mcp_server.py")], env={**os.environ, "MCP_ROOT": self.tmp.name},
            startup_timeout=15, tool_timeout=10))
        self.bridge.start()

    def tearDown(self):
        self.bridge.close()
        self.assertFalse(self.bridge._thread.is_alive())
        self.tmp.cleanup()

    def call(self, name, **arguments):
        out = self.bridge.call_tool(name, arguments)
        self.assertFalse(out["isError"], out)
        data = out["structured"]
        return data.get("result", data)

    def test_files_discovery_and_safe_replacement(self):
        names = {t["name"] for t in self.bridge.list_tools()}
        self.assertTrue({"read_text", "write_text", "grep_files"} <= names)
        self.call("write_text", path="a.txt", content="foo\nfoo\n")
        ambiguous = self.bridge.call_tool("replace_in_file", {"path": "a.txt", "old": "foo", "new": "bar"})
        self.assertTrue(ambiguous["isError"])
        self.assertIn("ambiguous replace", " ".join(ambiguous["text"]))
        self.call("replace_in_file", path="a.txt", old="foo", new="bar", line_number=2)
        self.assertEqual(self.call("read_text", path="a.txt")["content"], "foo\nbar\n")
        found = self.call("grep_files", pattern="bar", path=".")
        self.assertTrue(found["complete"])
        self.assertEqual(found["count"], 1)

    def test_restart(self):
        self.bridge.close()
        self.bridge.start()
        self.assertTrue(self.bridge.list_tools())


if __name__ == "__main__":
    unittest.main()


class ReliabilityTests(unittest.TestCase):
    def bridge(self, python=sys.executable):
        return MCPBridge(MCPBridgeConfig(command=python,
            args=[str(FIXTURES / "reliability_server.py")], startup_timeout=15, tool_timeout=5))

    def test_timeout_cancels_and_connection_survives(self):
        import time
        bridge = self.bridge()
        bridge.start()
        try:
            bridge.cfg.tool_timeout = 0.15
            with self.assertRaises(TimeoutError):
                bridge.call_tool("slow", {})
            bridge.cfg.tool_timeout = 5
            time.sleep(0.1)
            out = bridge.call_tool("status", {})
            self.assertFalse(out["isError"], out)
            data = out["structured"]
            if data is None:
                import json
                data = json.loads(out["text"][0])
            self.assertTrue(data.get("result", data)["cancelled"], out)
        finally:
            bridge.close()

    def test_crash_and_explicit_reconnect(self):
        bridge = self.bridge()
        bridge.start()
        try:
            with self.assertRaises(Exception):
                bridge.call_tool("crash", {})
            bridge.close()
            bridge.start()
            self.assertFalse(bridge.call_tool("status", {})["isError"])
        finally:
            bridge.close()

    def test_separate_process_interoperability(self):
        bridge = self.bridge(sys.executable)
        bridge.start()
        try:
            self.assertIn("status", {tool["name"] for tool in bridge.list_tools()})
            self.assertFalse(bridge.call_tool("status", {})["isError"])
        finally:
            bridge.close()


class GrepDeadlineTests(LocalIntegration):
    def test_hard_regex_deadline(self):
        self.call("write_text", path="slow.txt", content="a" * 10000 + "!\n")
        result = self.call("grep_files", pattern="(a+)+$", regex=True,
                           path="slow.txt", timeout_s=1)
        self.assertFalse(result["complete"])
        self.assertIn("hard_timeout", result["truncation_reasons"])
        self.assertTrue(self.bridge.list_tools())

    def test_cancel_grep_keeps_server_usable(self):
        import time
        self.call("write_text", path="slow.txt", content="a" * 10000 + "!\n")
        self.bridge.cfg.tool_timeout = 0.3
        with self.assertRaises(TimeoutError):
            self.bridge.call_tool("grep_files", {"pattern": "(a+)+$", "regex": True,
                                  "path": "slow.txt", "timeout_s": 30})
        self.bridge.cfg.tool_timeout = 10
        time.sleep(0.2)
        self.assertEqual(self.call("read_text", path="slow.txt")["content"], "a" * 10000 + "!\n")


class VisionServerTests(unittest.TestCase):
    def test_discovery_without_api_upload(self):
        bridge = MCPBridge(MCPBridgeConfig(command=sys.executable,
            args=[str(SERVER_ROOT / "mcp_openai_vision.py")], env=dict(os.environ)))
        bridge.start()
        try:
            self.assertTrue({"describe_image", "image_info", "compare_regions"} <=
                            {t["name"] for t in bridge.list_tools()})
        finally:
            bridge.close()


class ManagerTests(unittest.TestCase):
    def test_independent_connections_namespaces_and_permissions(self):
        import json
        # Import the unchanged manager with the candidate bridge already loaded.
        sys.path.append(str(ROOT.parent))
        from mcp_manager import MCPManager
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "servers.json"
            config.write_text(json.dumps({"servers": {
                "local": {"command": sys.executable,
                          "args": [str(SERVER_ROOT / "mcp_server.py")],
                          "env": {"MCP_ROOT": tmp}},
                "older": {"command": str(ROOT.parent / ".venv/bin/python"),
                          "args": [str(FIXTURES / "reliability_server.py")],
                          "read_only_tools": ["status"]},
                "disabled": {"enabled": False},
                "missing": {"command": "/nonexistent/mcp-v25", "required": False}
            }}))
            manager = MCPManager(tmp, str(config))
            manager.start()
            try:
                self.assertEqual(manager.resolve("read_text"), "local.read_text")
                self.assertFalse(manager.needs_confirmation("older.status"))
                self.assertTrue(manager.needs_confirmation("older.crash"))
                self.assertTrue(manager.needs_confirmation("write_text"))
                self.assertIsNot(manager.servers["local"], manager.servers["older"])
                self.assertEqual(manager.status()["servers"]["missing"]["state"], "failed")
                self.assertEqual(manager.status()["servers"]["disabled"]["state"], "disabled")
                manager.servers["older"].close()
                self.assertFalse(manager.call_tool("list_files", {"path": tmp})["isError"])
            finally:
                manager.close()
