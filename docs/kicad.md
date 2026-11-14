---
name: kicad
description: Create, inspect, modify, and validate KiCad schematics and projects with KiCad Copilot MCP and native KiCad CLI.
---

# KiCad schematic and project workflow

## 1. Discover capabilities and read the right documentation

- Call `manager.list_servers` before using external MCP services. If `kicad` is unavailable or disabled, report that; do not call its tools.
- Use the LIVE catalog, `manager.list_tools`, and `manager.describe_tool` for exact tool names and complete input schemas. Examples here are not evidence of availability. Never invent a GUI, wiring, or export tool.
- External calls use `tool="mcp"`, qualified `name="kicad.<tool>"`, and `arguments`. Keep approval prompts unless the administrator explicitly marks an operation read-only.
- Read `/home/intl/wrk/dlab/kicad-copilot/docs/SKILL.md`, then the relevant `circuit-maker/instructions.md`, `schematic/project-and-sheets.md`, `schematic/checkpoints.md`, `kicad-cli.md`, or `analysis.md`. For PCB work, also read `pcb-layout/instructions.md` and the routing documentation before making changes.
- `docs/kicad-mcp.md` describes connection setup; recheck runtime state rather than treating old setup notes as current capability checks.

## 2. Resolve files and protect the user's work

KiCad Copilot edits saved files, not the open editor through IPC. It has no selected/current project state. Ask the user to save intentional editor changes before external edits. Afterwards reload/reopen affected files without saving a stale editor copy over the result.

Call `kicad.get_project_info` with the supplied project/file/directory path. Resolve returned relative file paths against the directory holding the root `.kicad_pro`, not the agent workspace. Preserve user-supplied destinations. Read the exact target `.kicad_sch` with `kicad.get_schematic` before modifying it. That call reads one physical sheet, not the entire hierarchy.

Save a checkpoint before manual edits, layout changes, or risky operations. A checkpoint covers ONE file only: protect each affected sheet/project/board separately. `extract_circuit` automatically checkpoints its target, but that does not protect siblings. Record checkpoint IDs; restore only an explicitly selected ID and revalidate afterwards.

Use local MCP file tools for listing, searching, reading, and editing files, not shell file-management commands. Prefer small targeted edits; do not regenerate an existing schematic wholesale.

## 3. Create projects and hierarchical sheets

Use `kicad.create_doc` with a `doc` object:

- `doc_type="project"`, `path="/destination/name.kicad_pro"`: creates the project and matching root schematic.
- `doc_type="schematic"`, `path="/destination/sheet.kicad_sch"`: standalone schematic.
- `doc_type="schematic_page"`, child `path`, and exact `parent_path`: creates a child and its parent sheet symbol.

Creation does not overwrite existing destinations. Re-read project information after creating sheets. Target each physical child file explicitly; preserve relative Sheetfile references and hierarchical connectivity.

## 4. Plan components, pins, and electrical changes

1. Establish supply voltage, interfaces, operating targets, and component values. Calculate relevant timing/current/power limits; flag assumptions. Do not add speculative circuitry outside scope.
2. Build an intended net table of `(reference, pin number)` memberships. Check actual symbol pin numbers AND the selected package/datasheet pinout. Never infer transistor E/B/C, diode polarity, or connector numbering from appearance alone.
3. Resolve each component with `kicad.component_search`, using an exact MPN or `Library:Symbol`; pass the schematic path for project library lookup. Use the returned real `part_uuid` (including supported `kicad:Library:Symbol` IDs); never fabricate catalog identifiers. Use reusable-block search only if discovered and appropriate.
4. Call `kicad.extract_circuit` with the SAME exact sheet path and a coherent change set using its current full schema. Added components require designator, value, pins, block_name, search_query, and part_uuid. Each pin needs pin_number, name, and signal_name. Shared signals must have identical names.
5. This API adds/removes components rather than editing them in place. When replacement is necessary, remove/add with the same reference; use available external connection operations where sufficient. Preserve unaffected references, UUIDs, footprints, and connections.
6. Read the complete result, including recoverable item errors. A successful transport response may contain partial failures. Re-read the sheet and fix only the failed/missing operations; do not blindly replay a partially successful add batch.

## 5. Layout and visible wiring

Use `kicad.beautify_schematic` for compact functional groups: `path` is the exact sheet; `blocks` maps block names to reference arrays. A reference may appear in only one group; unlisted components remain in place. Re-read and validate connectivity after reassembly.

Logical connectivity and visual wiring are separate requirements. Matching labels may connect a circuit electrically but do not necessarily satisfy a request for a visibly wired schematic. Inspect the generated result instead of claiming that component insertion or beautification guarantees continuous wires.

For manual wiring/layout fallback, checkpoint first and read actual symbol geometry. Maintain valid KiCad S-expressions, unique UUIDs, library IDs, instance paths, and format version. Account for symbol placement, rotation, mirroring, and library-versus-sheet Y-axis conventions when locating pin endpoints. Put wires exactly on pin endpoints and the connection grid. Add junctions for intended branches; avoid accidental joins at crossings. Keep wires off symbol bodies/text and keep labels readable. Use genuine no-connect markers only on intentionally unused pins.

If space is insufficient, consider a checkpointed top-level paper change (A4 to A3, or a supported custom size) rather than compressing everything illegibly. Prefer supported MCP operations; manual S-expression editing is a controlled fallback, not the default creation workflow.

## 6. Validate using native KiCad, not appearances alone

Get the CLI executable/version from `get_project_info.kicad_cli`. If absent, check PATH or ask for configuration. This installation also has `/home/intl/wrk/dlab/ai-bash-agent/kicad-cli`; verify it before relying on it. Never persist temporary AppImage mount paths. Read installed command `--help` for unfamiliar flags.

Run native checks/exports through the command execution tool; do not use shell redirection or shell file editing. In these command templates, substitute the discovered executable for `kicad-cli` and exact absolute paths for the files:

- ERC: `kicad-cli sch erc --format json --severity-all --exit-code-violations --output /project/erc.json /project/design.kicad_sch`
- Netlist: `kicad-cli sch export netlist --format kicadxml --output /project/netlist.xml /project/design.kicad_sch`
- Preview: `kicad-cli sch export svg --exclude-drawing-sheet --output /project/preview/ /project/design.kicad_sch`

Check return codes AND report content. Read JSON reports with MCP file tools. ERC without the violations exit-code flag may return success despite findings. Do not suppress warnings, downgrade severities, or add unjustified PWR_FLAG/no-connect markers merely to reach zero.

Compare exported netlists against the intended pin/net table. For layout-only or library-only edits, compare before/after net membership as sets of `(reference, pin)` pairs rather than generated net IDs or arbitrary names. Equal net counts alone do not prove unchanged connections. Check every affected sheet and root hierarchy as appropriate. Zero ERC findings do not prove functional behavior, correct operating frequency, ratings, or analog performance; those require calculations and, when requested, simulation or measurements.

`kicad.analyze_schematic` can supplement ERC with focused connectivity, power, BOM, or hierarchy reports. Read `analysis.md` and the live schema; distinguish heuristic findings from native ERC results. Lifecycle audits may use network services and need their own approval.

Exporting an SVG/PDF is not visual inspection. Claim visual review only after actually viewing rendered pixels with an available approved vision workflow. A preview path alone is not evidence. Report electrical validation and visual validation separately.

## 7. Repair embedded-library mismatch warnings correctly

A cached symbol differing from its installed library can produce ERC mismatch warnings even when the nets are correct. Do not hide them with exclusions or replace the whole drawing to silence them.

- Confirm which installed/project library the ID resolves to, including sym-lib-table settings and the running KiCad version. Libraries may use a monolithic `.kicad_sym` or split `.kicad_symdir` files; discover actual paths rather than assuming a version/layout.
- Prefer KiCad's supported Update Symbols from Library workflow when available. For a manual fallback, checkpoint and read the authoritative symbol definition; handle inheritance if present.
- Align the embedded definition completely, not just its visible outline: metadata/properties, hide flags, pin-name offsets, graphics, pin names/numbers/types/positions, and relevant simulation fields. Avoid piecemeal guessing. Retain instance values, placements, references, footprints, UUIDs, and project paths unless a change is explicitly intended.
- Re-run ERC and compare before/after exported pin-net memberships. A pin-geometry change may require moving wires; never assume a library update preserved connectivity.

## 8. PCB handoff and completion

A schematic edit does not automatically update the PCB. Only when requested, read the PCB instructions, protect the board, and use the discovered `kicad.import_pcb_changes` with the exact board path and root schematic override if needed. Review synchronization results before placement or routing. Layout/routing use their documented DSLs, separate approvals, and native validation; do not infer a routed board from a successful schematic operation.

Finish with the exact files changed, ERC error/warning counts, netlist comparison outcome, preview/report paths, and any unresolved issues or unperformed checks. State explicitly when validation is incomplete. Remind the user to reload external changes in KiCad. For agent prompt updates, restart the application or begin a fresh session so the revised instructions are loaded.
