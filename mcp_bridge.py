#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pprint import pprint
import mcp.types as types


@dataclass
class MCPBridgeConfig:
    command: str
    args: list[str]
    env: dict[str, str]


class MCPBridge:
    def __init__(self, cfg: MCPBridgeConfig):
        self.cfg = cfg
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._closed = threading.Event()

        self._session: Optional[ClientSession] = None
        self._streams: Optional[Tuple[Any, Any]] = None  # (read, write)
        self._client_cm = None
        self._session_cm = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            raise RuntimeError("MCPBridge failed to start (timeout).")

    def close(self) -> None:
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._aclose(), self._loop)
        fut.result(timeout=10)
        self._closed.set()

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self._loop or not self._session:
            raise RuntimeError("MCPBridge not started.")
        fut = asyncio.run_coroutine_threadsafe(self._call_tool(name, arguments), self._loop)
        return fut.result(timeout=60)

    # ----------------- internals -----------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._aconnect())
            self._ready.set()
            self._loop.run_forever()
        finally:
            try:
                self._loop.run_until_complete(self._aclose())
            except Exception:
                pass
            self._loop.close()

    async def _aconnect(self) -> None:
        server_params = StdioServerParameters(
            command=self.cfg.command,
            args=self.cfg.args,
            env=self.cfg.env,
        )

        self._client_cm = stdio_client(server_params)
        read, write = await self._client_cm.__aenter__()
        self._streams = (read, write)

        self._session_cm = ClientSession(read, write)
        self._session = await self._session_cm.__aenter__()
        await self._session.initialize()

    async def _aclose(self) -> None:
        if self._session_cm:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._session = None

        if self._client_cm:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._client_cm = None
            self._streams = None

        if self._loop and self._loop.is_running():
            self._loop.stop()

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert self._session is not None
        # call_tool signature matches the SDK examples. :contentReference[oaicite:3]{index=3}
        res = await self._session.call_tool(name, arguments=arguments)
        print("---- CALL-TOOL resp----\n")
        pprint(res)
        print("------------------------")
        # Most tools return TextContent or structuredContent; normalize.
        out: Dict[str, Any] = {"structured": res.structuredContent, "text": []}
        for block in res.content:
            if isinstance(block, types.TextContent):
                out["text"].append(block.text)
        return out