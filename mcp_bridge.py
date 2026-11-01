#!/usr/bin/env python3
"""Synchronous bridge to one MCP server, running its session on a private loop."""
from __future__ import annotations

import asyncio
import base64
import os
import tempfile
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
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
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._session: Optional[ClientSession] = None
        self._streams: Optional[Tuple[Any, Any]] = None
        self._client_cm: Any = None
        self._session_cm: Any = None
        self._http_client: Any = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            if self._loop and self._loop.is_running() and self._session:
                return
            raise RuntimeError("MCPBridge thread exists but the session is not ready")
        self._ready.clear()
        self._closed.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="mcp-bridge")
        self._thread.start()
        if not self._ready.wait(timeout=self.cfg.startup_timeout):
            raise RuntimeError(f"MCPBridge startup timed out after {self.cfg.startup_timeout:g} seconds")
        if self._startup_error:
            raise RuntimeError(f"MCPBridge failed to start: {self._startup_error}") from self._startup_error
        if self._session is None or not self._thread.is_alive():
            raise RuntimeError("MCPBridge exited during startup")

    def close(self) -> None:
        if not self._loop or not self._loop.is_running():
            return
        fut = asyncio.run_coroutine_threadsafe(self._aclose(), self._loop)
        try:
            fut.result(timeout=10)
        except FutureTimeoutError:
            fut.cancel()
        finally:
            self._closed.set()
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

    def _submit(self, coro: Any, label: str, timeout: Optional[float] = None) -> Any:
        if (not self._loop or not self._loop.is_running() or not self._thread
                or not self._thread.is_alive() or self._session is None):
            raise RuntimeError("MCPBridge connection is not running; restart this MCP server")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        deadline = self.cfg.tool_timeout if timeout is None else timeout
        try:
            return fut.result(timeout=deadline)
        except FutureTimeoutError as exc:
            fut.cancel()
            raise TimeoutError(f"MCP {label} timed out after {deadline:g} seconds") from exc

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self._submit(self._call_tool(name, arguments), f"tool '{name}'")

    def list_tools(self) -> List[Dict[str, Any]]:
        return self._submit(self._list_tools(), "tools/list")

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            try:
                self._loop.run_until_complete(self._aconnect())
            except BaseException as exc:
                self._startup_error = exc
                self._ready.set()
                return
            self._ready.set()
            self._loop.run_forever()
        finally:
            try:
                self._loop.run_until_complete(self._aclose())
            except Exception:
                pass
            self._loop.close()

    async def _aconnect(self) -> None:
        if self.cfg.transport == "stdio":
            params = StdioServerParameters(
                command=self.cfg.command, args=self.cfg.args, env=self.cfg.env
            )
            self._client_cm = stdio_client(params)
        elif self.cfg.transport == "streamable_http":
            try:
                from mcp.client.streamable_http import streamable_http_client
            except ImportError as exc:
                raise RuntimeError("MCP HTTP transport needs mcp.client.streamable_http") from exc
            try:
                import httpx2 as httpx  # Newer MCP SDKs use httpx2.
            except ImportError:
                import httpx  # MCP Python SDK v1.x uses httpx.
            self._http_client = httpx.AsyncClient(
                headers=self.cfg.headers,
                timeout=httpx.Timeout(connect=30.0, read=max(300.0, self.cfg.tool_timeout),
                                      write=30.0, pool=30.0),
            )
            self._client_cm = streamable_http_client(self.cfg.url, http_client=self._http_client)
        else:
            raise ValueError(f"Unsupported transport: {self.cfg.transport}")
        connection = await self._client_cm.__aenter__()
        read, write = connection[:2]  # HTTP transport may also return a session-id callback.
        self._streams = (read, write)
        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _aclose(self) -> None:
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            finally:
                self._session_cm = None
                self._session = None
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            finally:
                self._client_cm = None
                self._streams = None
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            finally:
                self._http_client = None

    async def _list_tools(self) -> List[Dict[str, Any]]:
        assert self._session is not None
        results: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        seen_cursors = set()
        for _ in range(100):
            if cursor is None:
                page = await self._session.list_tools()
            else:
                try:
                    page = await self._session.list_tools(
                        params=types.PaginatedRequestParams(cursor=cursor)
                    )
                except TypeError:
                    page = await self._session.list_tools(cursor=cursor)
            for tool in page.tools:
                schema = getattr(tool, "inputSchema", None)
                if schema is None:
                    schema = getattr(tool, "input_schema", {"type": "object"})
                if not isinstance(schema, dict):
                    schema = schema.model_dump(by_alias=True, exclude_none=True)
                results.append({
                    "name": tool.name,
                    "description": getattr(tool, "description", "") or "",
                    "inputSchema": schema,
                })
            cursor = getattr(page, "nextCursor", None) or getattr(page, "next_cursor", None)
            if not cursor:
                return results
            if cursor in seen_cursors:
                raise RuntimeError("MCP tools/list repeated pagination cursor")
            seen_cursors.add(cursor)
        raise RuntimeError("MCP tools/list exceeded 100 pages")

    def _save_image(self, data: str, mime_type: str) -> Dict[str, Any]:
        if mime_type not in ("image/png", "image/jpeg", "image/webp"):
            return {"error": f"Unsupported MCP image MIME type: {mime_type}"}
        if len(data) > 24 * 1024 * 1024:
            return {"error": "MCP image exceeds size limit"}
        raw = base64.b64decode(data, validate=True)
        if len(raw) > 16 * 1024 * 1024:
            return {"error": "Decoded MCP image exceeds size limit"}
        directory = Path(self.cfg.image_dir or tempfile.gettempdir())
        directory.mkdir(parents=True, exist_ok=True)
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[mime_type]
        fd, path = tempfile.mkstemp(prefix="mcp-view-", suffix=suffix, dir=str(directory))
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(raw)
        except Exception:
            Path(path).unlink(missing_ok=True)
            raise
        return {"path": path, "mime_type": mime_type, "bytes": len(raw)}

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        # Never pprint a tool response in the event-loop thread: large text or
        # a full screenshot can block stdout and stall every subsequent call.
        res = await self._session.call_tool(name, arguments=arguments)
        out: Dict[str, Any] = {
            "structured": getattr(res, "structuredContent", None),
            "text": [],
            "images": [],
            "isError": bool(getattr(res, "isError", False)),
        }
        for block in res.content:
            if isinstance(block, types.TextContent):
                out["text"].append(block.text)
            elif isinstance(block, types.ImageContent):
                try:
                    result = await asyncio.to_thread(self._save_image, block.data, block.mimeType)
                    out["images"].append(result)
                    if "path" in result:
                        out["text"].append(f"[MCP image saved to {result['path']}]")
                except Exception as exc:
                    out["images"].append({"error": str(exc)})
        return out
