"""Test-only server; supports both the retained v1 and candidate v2 SDK."""
import asyncio
import os
try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer

server = MCPServer("migration-test")
cancelled = False

@server.tool()
async def slow() -> str:
    global cancelled
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        cancelled = True
        raise
    return "finished"

@server.tool()
def status() -> dict:
    return {"cancelled": cancelled}

@server.tool()
def crash() -> str:
    os._exit(23)

if __name__ == "__main__":
    server.run(transport="stdio")
