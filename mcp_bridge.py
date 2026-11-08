#!/usr/bin/env python3
"""Synchronous facade over an independently owned MCP 2.x Client.

The owner task enters AND exits SDK contexts; request tasks never own transport
cancel scopes. No failed call is replayed (it may already have changed CAD).
"""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
import mcp.types as types


@dataclass
class MCPBridgeConfig:
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    tool_timeout: float = 300.0
    startup_timeout: float = 15.0
    transport: str = "stdio"
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    image_dir: Optional[str] = None


class MCPBridge:
    def __init__(self, cfg: MCPBridgeConfig):
        self.cfg = cfg
        self._thread = None
        self._loop = None
        self._client = None
        self._owner = None
        self._stop = None
        self._ready = threading.Event()
        self._startup_error = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._client is not None:
                return
            raise RuntimeError("MCP connection is starting or closing")
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-bridge")
        self._thread.start()
        if not self._ready.wait(self.cfg.startup_timeout):
            self.close()
            raise TimeoutError(f"MCP startup timed out after {self.cfg.startup_timeout:g} seconds")
        if self._startup_error is not None:
            self.close()
            raise RuntimeError(f"MCPBridge failed to start: {self._startup_error}") from self._startup_error
        if self._client is None:
            raise RuntimeError("MCPBridge exited during startup")

    def close(self) -> None:
        loop = self._loop
        if loop and loop.is_running():
            loop.call_soon_threadsafe(self._request_stop)
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=10)
            if thread.is_alive():
                raise TimeoutError("MCP bridge shutdown did not finish within 10 seconds")

    def _request_stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._client is None and self._owner is not None:
            self._owner.cancel()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._owner = self._loop.create_task(self._serve())
            self._loop.run_until_complete(self._owner)
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    async def _serve(self) -> None:
        self._stop = asyncio.Event()
        try:
            async with AsyncExitStack() as stack:
                if self.cfg.transport == "stdio":
                    transport = stdio_client(StdioServerParameters(
                        command=self.cfg.command, args=self.cfg.args, env=self.cfg.env))
                elif self.cfg.transport == "streamable_http":
                    http = await stack.enter_async_context(httpx2.AsyncClient(
                        headers=self.cfg.headers,
                        timeout=httpx2.Timeout(30.0, read=max(300.0, self.cfg.tool_timeout))))
                    transport = streamable_http_client(self.cfg.url, http_client=http)
                else:
                    raise ValueError(f"Unsupported transport: {self.cfg.transport}")
                # Reproduce v1 handshakes, including old FreeCAD MCP servers.
                self._client = await stack.enter_async_context(Client(transport, mode="legacy"))
                self._ready.set()
                await self._stop.wait()
        except BaseException as exc:
            self._startup_error = exc
        finally:
            self._client = None
            self._ready.set()

    def _submit(self, factory: Any, label: str) -> Any:
        loop = self._loop
        if (loop is None or not loop.is_running() or self._client is None
                or self._stop.is_set()):
            raise RuntimeError("MCP connection is not running; restart this MCP server")
        # Construct the coroutine only after checking connectivity.
        fut = asyncio.run_coroutine_threadsafe(factory(), loop)
        try:
            return fut.result(timeout=self.cfg.tool_timeout)
        except FutureTimeoutError as exc:
            fut.cancel()  # SDK cancellation sends notifications/cancelled.
            raise TimeoutError(f"MCP {label} timed out after {self.cfg.tool_timeout:g} seconds") from exc

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._submit(lambda: self._call_tool(name, arguments), f"tool '{name}'")

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._submit(self._list_tools, "tools/list")

    async def _list_tools(self) -> List[Dict[str, Any]]:
        results = []
        cursor = None
        seen = set()
        for _ in range(100):
            page = await self._client.list_tools(cursor=cursor, cache_mode="bypass")
            for tool in page.tools:
                results.append({"name": tool.name,
                                "description": tool.description or "",
                                "inputSchema": tool.input_schema})
            cursor = page.next_cursor
            if cursor is None:
                return results
            if cursor in seen:
                raise RuntimeError("MCP tools/list repeated pagination cursor")
            seen.add(cursor)
        raise RuntimeError("MCP tools/list exceeded 100 pages")

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        res = await self._client.call_tool(name, arguments=arguments)
        return await self._normalize_result(res)

    async def _normalize_result(self, res: Any) -> Dict[str, Any]:
        out = {"structured": res.structured_content, "text": [],
               "images": [], "isError": bool(res.is_error)}
        for block in res.content:
            if isinstance(block, types.TextContent):
                out["text"].append(block.text)
            elif isinstance(block, types.ImageContent):
                try:
                    image = await asyncio.to_thread(self._save_image, block.data, block.mime_type)
                    out["images"].append(image)
                    if "path" in image:
                        out["text"].append(f"[MCP image saved to {image['path']}]")
                except Exception as exc:
                    out["images"].append({"error": str(exc)})
        return out

    def _save_image(self, data: str, mime_type: str) -> Dict[str, Any]:
        suffixes = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
        if mime_type not in suffixes:
            return {"error": f"Unsupported MCP image MIME type: {mime_type}"}
        if len(data) > 24 * 1024 * 1024:
            return {"error": "MCP image exceeds size limit"}
        raw = base64.b64decode(data, validate=True)
        if len(raw) > 16 * 1024 * 1024:
            return {"error": "Decoded MCP image exceeds size limit"}
        directory = Path(self.cfg.image_dir or tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix="mcp-view-", suffix=suffixes[mime_type], dir=str(directory))
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(raw)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return {"path": path, "mime_type": mime_type, "bytes": len(raw)}
