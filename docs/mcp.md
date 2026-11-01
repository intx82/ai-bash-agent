# Multi-server MCP integration for the existing agent

This adds `mcp_manager.py` and `mcp_servers.json` while preserving the current
`run_agent_turn()` interface and legacy local MCP calls.

## Files

Install these **four** files next to the existing `mcp_server.py`:

- `agent.py` — updated dispatcher and live tool descriptions
- `mcp_bridge.py` — one connection/session per MCP server; tool discovery; stdio and Streamable HTTP; image result saving
- `mcp_manager.py` — config, lifecycle, aliases, routing, confirmation decisions
- `mcp_servers.json` — enabled MCP servers

Existing `mcp_server.py` and `llm_backends.py` do not need editing.

## Enable FreeCAD

1. Install `uv` (`uvx`) and `FreeCAD`.
2. Install the addon from <https://github.com/neka-nat/freecad-mcp> into FreeCAD's Mod directory.
3. In FreeCAD select the MCP Addon workbench and start its RPC server.
4. FreeCAD is **enabled by default in the corrected example configuration** (`servers.freecad.enabled=true`). If you retain the old config, change `false` to `true`.
5. Restart `agent.py` (or the GTK GUI using it); enter `status` to see connected servers. The manager prints connected/failed/disabled status during CLI startup.

The external FreeCAD MCP server runs using `uvx freecad-mcp --only-text-feedback`.
It connects to the FreeCAD addon on localhost, normally XML-RPC port 9875.
Port 9875 is **not** an MCP endpoint.

The FreeCAD MCP external server requires Python 3.12+, independent of the Python
used by your agent and the Python embedded in FreeCAD. `uvx` handles that
independent environment.

**Preflight** (on the agent machine):

```bash
command -v uvx
uvx --version
```

Do not run `uvx freecad-mcp` manually in a terminal as a connectivity test:
it is a stdio MCP service and waits for protocol messages on stdin.
Start FreeCAD's **RPC Server** inside the FreeCAD GUI, then let the agent
spawn the `uvx freecad-mcp --only-text-feedback` process.

If the config points to a missing executable or the MCP server fails startup,
FreeCAD is marked `failed` (because it is optional), and local tools remain
usable. The error appears in `status`, in `manager.list_servers`, and in the
agent's tool validation feedback. The LLM must not attempt tools belonging to
servers that are disabled or failed.

## LLM tool usage

Legacy local calls remain valid:

```json
{"type":"tool","plan":["Read source"],"tool":"mcp","name":"read_text","arguments":{"path":"src/test.c"}}
```

A namespaced local call also works: `local.read_text`.

External tools always include their server namespace. Example (arguments are
based on discovered tools, not hardcoded in the agent):

```json
{"type":"tool","plan":["Create model"],"tool":"mcp","name":"freecad.create_document","arguments":{"name":"TestModel"}}
```

Tool names, descriptions and input JSON schemas are discovered through MCP
`tools/list` at start. The agent's system message includes the discovered
external tools. For very large catalogs, the LLM can request
`manager.list_servers` to check connection state, `manager.list_tools`
(optional `server`, `offset`, `limit`) and `manager.describe_tool` (`name`)
to inspect available tools and their full schemas without wasting prompt context.
`manager.list_tools({"server":"freecad"})` now reports `ok=false`, `isError=true`
and a specific reason when FreeCAD is disabled or failed, instead of a
misleading successful empty list. These virtual manager tools are
read-only and do not require confirmation. A failed optional server doesn't prevent local tools from
working. `status` lists connected/disabled/unavailable servers in CLI mode.

## Configuration format

`servers` is a mapping from server names to definitions. Each tool is called
with `server_name.tool_name`. The `local` server additionally provides old
unprefixed aliases for compatibility.

```json
{
  "servers": {
    "local": {
      "required": true,
      "transport": "stdio",
      "command": "${PYTHON}",
      "args": ["${AGENT_DIR}/mcp_server.py"],
      "env": {"MCP_ROOT": "${WORKSPACE}"}
    },
    "another_server": {
      "enabled": false,
      "transport": "streamable_http",
      "url": "https://example.com/mcp",
      "headers": {"Authorization": "Bearer ${ENV:EXAMPLE_MCP_TOKEN}"},
      "timeout_s": 60,
      "required": false,
      "read_only_tools": ["search"]
    }
  }
}
```

Supported fields: `enabled`, `required`, `transport` (`stdio` or
`streamable_http`), `command`, `args`, `env`, `url`, `headers`, `timeout_s`,
`startup_timeout_s`, `read_only_tools`, `confirm_tools`. Remote insecure HTTP
is blocked by default; explicit `allow_insecure_http=true` overrides the check.
Only explicit placeholders `${PYTHON}`, `${AGENT_DIR}`, `${WORKSPACE}` and
`${ENV:VARIABLE}` are expanded. Secrets should be provided via environment
variables, not embedded in the JSON file.

For third-party servers, every tool requires user confirmation except those
explicitly named in `read_only_tools`. Do not mark tools read-only without
reviewing their behavior. For the local server, existing confirmation policy
is preserved. The `--no-confirm` option bypasses these prompts as before.

## Images

The bridge saves returned MCP PNG/JPEG/WebP image blocks as files in the
workspace's `mcp_images/` directory and includes their paths in the tool
result. The CLI can therefore refer to the image file; rendering those images
inline inside the GTK GUI is a separate optional GUI enhancement.

## Dependencies and testing

The updated bridge uses the installed Python `mcp` SDK (as did your previous
version). Streamable HTTP additionally needs an SDK exposing
`mcp.client.streamable_http`, and the corresponding HTTP dependency (`httpx`
for older MCP Python SDK versions, or `httpx2` for newer ones).

Run `python3 -m py_compile agent.py mcp_manager.py mcp_bridge.py` after copying.
Restart the agent to pick up configuration changes. No MCP server or daemon is
installed automatically by these files.

## Troubleshooting FreeCAD not appearing

If `freecad.create_document` is rejected:

1. Check `mcp_servers.json` beside **the running** `agent.py` (not only in your download folder): `servers.freecad.enabled` must be `true`.
2. Restart the agent/GTK application after changing config.
3. Run `status` in CLI mode or ask the LLM to call `manager.list_servers` and inspect `freecad.state` and `freecad.error`.
4. If `state=failed`, confirm `uvx` is in the PATH of the agent process and review the error text.
5. If `state=connected`, inspect `manager.list_tools({"server":"freecad"})` to see the actual discovered tool names.
6. If tools are listed but operations fail, verify FreeCAD GUI is open and the addon RPC server was started on the expected host/port. A connected MCP stdio process does **not** establish that the FreeCAD GUI RPC process is online. If discovered, call `freecad.get_rpc_status`.

Only discovered tools are advertised as executable; example tool names in older prompts are not evidence of availability.

## Regression testing

`python3 test_mcp_availability.py` verifies the disabled / failed / connected server cases, discovery errors and agent validation feedback using in-process mock MCP bridges. `python3 test_multi_mcp.py` and `python3 test_mcp_bridge_thread.py` exercise routing, confirmations and bridge timeout isolation. These tests do not verify a real FreeCAD RPC addon.
