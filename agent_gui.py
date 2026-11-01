#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import importlib
import inspect
import builtins
import ctypes
import os
import io
import json
import re
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

GTKUI_VERSION = "R260820_GUI_ONLY_12"


# ---------------------------------------------------------------------------
# GTK agent GUI
# ---------------------------------------------------------------------------
# GUI-only wrapper.  Agent/core logic is expected to live in a separate module
# named "agent" by default, or in AGENT_CORE_MODULE if that environment
# variable is set.  GTK imports stay lazy so importing this module works on
# headless systems.

def _markdown_fence(text: str, lang: str = "text") -> str:
    safe = text.replace("```", "` ` `")
    return f"```{lang}\n{safe.rstrip()}\n```"


_BOX_LINE_RE = re.compile(r"^\s*[-─━_=!]{8,}\s*$")


def _compact_agent_console_output(output_text: str) -> str:
    """Make captured agent stdout/stderr usable in the GUI transcript.

    The core agent may still print terminal-style separators.  The GUI keeps
    the useful content but strips that frame, so live blocks stay compact.
    """
    text = str(output_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    out: List[str] = []

    heading_map = {
        "🧠 REASONING": "REASONING:",
        "🧭 PLAN": "PLAN:",
        "⚠️  BASH TOOL REQUEST ⚠️": "BASH TOOL REQUEST:",
        "⚠️ BASH TOOL REQUEST ⚠️": "BASH TOOL REQUEST:",
        "⚠️  MCP TOOL REQUEST ⚠️": "MCP TOOL REQUEST:",
        "⚠️ MCP TOOL REQUEST ⚠️": "MCP TOOL REQUEST:",
    }

    def append_blank() -> None:
        if out and out[-1] != "":
            out.append("")

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            append_blank()
            continue

        if _BOX_LINE_RE.match(stripped):
            continue

        if stripped in heading_map:
            append_blank()
            out.append(heading_map[stripped])
            continue

        if stripped == "Agent:":
            continue

        if stripped.startswith("Agent: "):
            append_blank()
            out.append("MESSAGE:")
            out.append(stripped[len("Agent: "):])
            continue

        out.append(line)

    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _raise_async_exception(thread_id: Optional[int], exc_type: type[BaseException]) -> bool:
    """Ask CPython to raise an exception in another Python thread.

    This is intentionally used only for the GUI Stop button, which should act
    like Ctrl-C in the CLI: interrupt the current wait/agent turn and return
    control to the UI.  It is still cooperative-ish: if the worker is blocked
    inside a native extension or an OS call, the exception may be delivered
    only when Python bytecode runs again.
    """
    if not thread_id:
        return False
    try:
        setter = ctypes.pythonapi.PyThreadState_SetAsyncExc
        setter.argtypes = (ctypes.c_ulong, ctypes.py_object)
        setter.restype = ctypes.c_int
        result = setter(ctypes.c_ulong(int(thread_id)), ctypes.py_object(exc_type))
        if result == 0:
            return False
        if result > 1:
            # Undo if CPython reports that more than one thread state was hit.
            setter(ctypes.c_ulong(int(thread_id)), None)
            return False
        return True
    except Exception:
        return False


GUI_INPUT_HEIGHT = _env_int("AGENT_GUI_INPUT_HEIGHT", 112)
GUI_VISIBLE_TEXT_LIMIT = _env_int("AGENT_GUI_VISIBLE_TEXT_LIMIT", 24000)
GUI_VISIBLE_LINE_LIMIT = _env_int("AGENT_GUI_VISIBLE_LINE_LIMIT", 350)
GUI_KEEP_TAIL_CHARS = _env_int("AGENT_GUI_KEEP_TAIL_CHARS", 6000)
GUI_KEEP_TAIL_LINES = _env_int("AGENT_GUI_KEEP_TAIL_LINES", 80)

# Fenced code blocks are rendered with GtkSourceView again, but only for
# real code snippets and only while they are reasonably small. Huge tool
# output/logs still use the plain tagged TextBuffer path; otherwise GTK can
# become slow and the main transcript scrollbar becomes unusable.
GUI_SOURCE_PREVIEW = _env_bool("AGENT_GUI_SOURCE_PREVIEW", True)
GUI_SOURCE_PREVIEW_MAX_CHARS = _env_int("AGENT_GUI_SOURCE_PREVIEW_MAX_CHARS", 18000)
GUI_SOURCE_PREVIEW_MAX_LINES = _env_int("AGENT_GUI_SOURCE_PREVIEW_MAX_LINES", 220)
GUI_SOURCE_PREVIEW_MAX_VISUAL_LINES = _env_int("AGENT_GUI_SOURCE_PREVIEW_MAX_VISUAL_LINES", 56)

_SOURCE_LANG_ALIASES = {
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "hh": "cpp",
    "python": "python3",
    "py": "python3",
    "bash": "sh",
    "shell": "sh",
    "sh": "sh",
    "json": "json",
    "xml": "xml",
    "html": "html",
    "css": "css",
    "js": "js",
    "javascript": "js",
    "lua": "lua",
    "make": "makefile",
    "makefile": "makefile",
    "cmake": "cmake",
    "meson": "meson",
    "yaml": "yaml",
    "yml": "yaml",
    "toml": "toml",
    "ini": "ini",
    "sql": "sql",
}

_PLAIN_CODE_LANGS = {"", "text", "txt", "plain", "plaintext", "output", "log", "none"}


def _limit_visible_text(
    text: Any,
    max_chars: int = GUI_VISIBLE_TEXT_LIMIT,
    max_lines: int = GUI_VISIBLE_LINE_LIMIT,
) -> str:
    """Limit very large tool/LLM output for the GTK transcript only.

    The full tool result is still kept in the agent context by agent.py.  This
    function only prevents the visible TextBuffer from becoming several MB tall,
    which makes GTK scrolling and layout unusable.
    """
    s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    original_chars = len(s)
    original_lines = s.count("\n") + (1 if s else 0)
    truncated = False

    if max_lines > 0 and original_lines > max_lines:
        lines = s.split("\n")
        head_count = max(1, max_lines - GUI_KEEP_TAIL_LINES)
        tail_count = max(1, min(GUI_KEEP_TAIL_LINES, max_lines // 3))
        omitted = len(lines) - head_count - tail_count
        s = (
            "\n".join(lines[:head_count])
            + f"\n\n[… GUI preview truncated: {omitted} lines omitted …]\n\n"
            + "\n".join(lines[-tail_count:])
        )
        truncated = True

    if max_chars > 0 and len(s) > max_chars:
        head_count = max(1, max_chars - GUI_KEEP_TAIL_CHARS)
        tail_count = max(1, min(GUI_KEEP_TAIL_CHARS, max_chars // 3))
        omitted = len(s) - head_count - tail_count
        s = (
            s[:head_count]
            + f"\n\n[… GUI preview truncated: {omitted} chars omitted …]\n\n"
            + s[-tail_count:]
        )
        truncated = True

    if truncated:
        s += (
            f"\n\n[Full result is still available to the agent/core context; "
            f"GUI preview was limited from {original_lines} lines / {original_chars} chars.]"
        )

    return s


def _looks_large_for_markdown(text: str) -> bool:
    return len(text) > GUI_VISIBLE_TEXT_LIMIT or text.count("\n") + 1 > GUI_VISIBLE_LINE_LIMIT


def _load_default_core() -> Any:
    module_name = os.environ.get("AGENT_CORE_MODULE", "agent")
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        print(f"Cannot import agent core module {module_name!r}: {exc}", file=sys.stderr)
        print("Set AGENT_CORE_MODULE=<module_name> or place agent.py next to this GUI file.", file=sys.stderr)
        raise SystemExit(1) from exc


def _make_default_args(core: Any) -> Any:
    return SimpleNamespace(
        server=os.environ.get("AGENT_SERVER", "llama.cpp"),
        url=os.environ.get("AGENT_URL", getattr(core, "DEFAULT_SERVER_URL", "http://127.0.0.1:8080/v1/chat/completions")),
        model=os.environ.get("AGENT_MODEL") or None,
        api_key=os.environ.get("AGENT_API_KEY") or None,
        max_tokens=_env_int("AGENT_MAX_TOKENS", 4096),
        temp=_env_float("AGENT_TEMP", 0.7),
        no_confirm=_env_bool("AGENT_NO_CONFIRM", False),
        no_json_schema=_env_bool("AGENT_NO_JSON_SCHEMA", False),
        debug=_env_bool("AGENT_DEBUG", False),
        gui=True,
        open_md=os.environ.get("AGENT_OPEN_MD") or None,
    )


def run_agent_gui(args: Optional[Any] = None, core: Optional[Any] = None) -> None:
    """Start the GTK wrapper for the agent.

    This file is GUI-only.  The backend/MCP/LLM turn implementation is imported
    from a separate core module, normally `agent.py`.  A caller may pass `core`
    and `args` directly; otherwise defaults are read from environment variables.
    """
    if core is None:
        core = _load_default_core()
    if args is None:
        args = _make_default_args(core)
    elif getattr(args, "url", None) is None:
        args.url = getattr(core, "DEFAULT_SERVER_URL", None)

    DEFAULT_SERVER_URL = core.DEFAULT_SERVER_URL
    DEFAULT_MODEL_LLAMA = core.DEFAULT_MODEL_LLAMA
    DEFAULT_MODEL_OPENAI = core.DEFAULT_MODEL_OPENAI
    SYSTEM_PROMPT = core.SYSTEM_PROMPT
    _start_mcp_for_workspace = core._start_mcp_for_workspace
    _request_handoff_summary = core._request_handoff_summary
    _build_project_status = core._build_project_status
    run_agent_turn = core.run_agent_turn
    try:
        import gi

        # Pin every GTK-related namespace before importing gi.repository.
        # Without Gdk pinning, PyGObject may load Gdk 4 first on systems that
        # have GTK4 installed, then Gtk 3 startup fails with:
        # "Requiring namespace 'Gdk' version '3.0', but '4.0' is already loaded".
        gi.require_version("Gdk", "3.0")
        gi.require_version("Gtk", "3.0")

        have_gtksource = False
        if GUI_SOURCE_PREVIEW:
            try:
                gi.require_version("GtkSource", "4")
                have_gtksource = True
            except Exception:
                have_gtksource = False

        from gi.repository import Gtk, Gdk, GLib, Pango
        if have_gtksource:
            from gi.repository import GtkSource
        else:
            GtkSource = None
        import mistune
    except Exception as exc:
        print(f"Cannot start GTK UI: {exc}", file=sys.stderr)
        print(
            "Install GTK UI dependencies, for example: "
            "python3-gi gir1.2-gtk-3.0 python3-mistune "
            "gir1.2-gtksource-4 (optional, for highlighted code preview)",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    class FixedInputScrolledWindow(Gtk.ScrolledWindow):
        """A scroller whose size request never depends on typed text."""

        def __init__(self, fixed_height: int) -> None:
            super().__init__()
            self.fixed_height = max(1, int(fixed_height))

        def do_get_request_mode(self) -> Any:
            return Gtk.SizeRequestMode.CONSTANT_SIZE

        def do_get_preferred_width(self) -> Tuple[int, int]:
            return 1, 1

        def do_get_preferred_width_for_height(self, _height: int) -> Tuple[int, int]:
            return 1, 1

        def do_get_preferred_height(self) -> Tuple[int, int]:
            return self.fixed_height, self.fixed_height

        def do_get_preferred_height_for_width(self, _width: int) -> Tuple[int, int]:
            return self.fixed_height, self.fixed_height

    class AgentGuiWindow(Gtk.Window):
        def _pick_initial_window_size(self) -> Tuple[int, int]:
            try:
                display = Gdk.Display.get_default()
                monitor = display.get_primary_monitor() if display is not None else None
                if monitor is not None:
                    area = monitor.get_workarea()
                    return (
                        min(1000, max(720, int(area.width * 0.82))),
                        min(760, max(520, int(area.height * 0.82))),
                    )
            except Exception:
                pass
            return 1000, 760

        @staticmethod
        def _configure_scrolled_window(scrolled: Any, min_h: int = 1, max_h: Optional[int] = None) -> None:
            # The important part: do not let the child TextView natural size
            # propagate to the toplevel.  Without this, long input or huge
            # output can make the whole window grow beyond the screen.
            # Width must never follow a long child line.  When max_h is set,
            # height propagation must be enabled for GtkScrolledWindow's
            # min/max-content-height limits to control growth and start
            # scrolling at max_h.
            for method_name, propagate in (
                ("set_propagate_natural_width", False),
                ("set_propagate_natural_height", max_h is not None),
            ):
                method = getattr(scrolled, method_name, None)
                if method is not None:
                    try:
                        method(propagate)
                    except Exception:
                        pass
            if hasattr(scrolled, "set_min_content_width"):
                try:
                    scrolled.set_min_content_width(1)
                except Exception:
                    pass
            if hasattr(scrolled, "set_min_content_height"):
                try:
                    scrolled.set_min_content_height(min_h)
                except Exception:
                    pass
            if max_h is not None and hasattr(scrolled, "set_max_content_height"):
                try:
                    scrolled.set_max_content_height(max_h)
                except Exception:
                    pass

        def __init__(self) -> None:
            super().__init__(title="AI Agent")

            self.set_resizable(True)
            try:
                self.set_type_hint(Gdk.WindowTypeHint.NORMAL)
            except Exception:
                pass
            default_w, default_h = self._pick_initial_window_size()
            self.set_default_size(default_w, default_h)
            self.connect("destroy", self.on_destroy)

            self.args = args
            self.model = args.model or (
                DEFAULT_MODEL_OPENAI
                if args.server in ("chatgpt", "openai")
                else DEFAULT_MODEL_LLAMA
            )

            self.backend: Any = None
            self.workspace: Optional[tempfile.TemporaryDirectory[str]] = None
            self.mcp: Any = None
            self.messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            self.transcript_markdown = ""
            self.busy = False
            self._warned_no_event_callback = False
            self.worker_thread: Optional[threading.Thread] = None
            self.worker_ident: Optional[int] = None
            self.stop_requested = False
            self.cancel_event = threading.Event()

            self.md_parser = mistune.create_markdown(
                renderer="ast",
                plugins=["task_lists", "strikethrough", "table"],
            )

            self.text_view = Gtk.TextView()
            self.text_view.set_editable(False)
            self.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.text_view.set_left_margin(10)
            self.text_view.set_right_margin(10)
            self.text_view.set_top_margin(8)
            self.text_view.set_bottom_margin(8)
            self.buffer = self.text_view.get_buffer()
            self.create_tags()

            self.scroll = Gtk.ScrolledWindow()
            # The transcript should wrap inside the window, not create a
            # horizontal scrollbar or request a giant natural size.
            self.scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            self._configure_scrolled_window(self.scroll, min_h=240)
            self.text_view.set_hexpand(True)
            self.text_view.set_vexpand(True)
            self.text_view.set_size_request(1, 1)
            self.scroll.add(self.text_view)

            self.input_buffer = Gtk.TextBuffer()
            self.input_text_view = Gtk.TextView.new_with_buffer(self.input_buffer)
            self.input_text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            self.input_text_view.set_left_margin(8)
            self.input_text_view.set_right_margin(8)
            self.input_text_view.set_top_margin(6)
            self.input_text_view.set_bottom_margin(6)
            self.input_text_view.set_accepts_tab(False)
            self.input_text_view.set_hexpand(True)
            self.input_text_view.set_vexpand(False)
            self.input_text_view.set_size_request(1, 1)
            self.input_text_view.set_tooltip_text("Multiline prompt. Press Ctrl+Enter to send, Enter for a new line.")
            self.input_text_view.connect("key-press-event", self.on_input_key_press)
            self.input_buffer.connect("changed", self.on_input_buffer_changed)

            self.input_scroll = FixedInputScrolledWindow(GUI_INPUT_HEIGHT)
            self.input_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
            self.input_scroll.set_shadow_type(Gtk.ShadowType.IN)
            self.input_scroll.set_hexpand(True)
            self.input_scroll.set_vexpand(False)
            self.input_scroll.set_size_request(1, GUI_INPUT_HEIGHT)
            self._configure_scrolled_window(self.input_scroll, min_h=GUI_INPUT_HEIGHT, max_h=GUI_INPUT_HEIGHT)
            self.input_scroll.add(self.input_text_view)

            self.send_button = Gtk.Button(label="➤ Send")
            self.send_button.connect("clicked", self.on_send_clicked)

            self.open_button = Gtk.Button(label="📄 Open agent.md")
            self.open_button.set_tooltip_text(
                "Open an agent.md/instructions Markdown file, or choose another text file."
            )
            self.open_button.connect("clicked", self.open_agent_md)

            self.load_button = Gtk.Button(label="📂 Load")
            self.load_button.set_tooltip_text("Load LLM context from a JSON file")
            self.load_button.connect("clicked", self.load_context_dialog)

            self.save_button = Gtk.Button(label="💾 Save")
            self.save_button.set_tooltip_text("Save current LLM context to a JSON file")
            self.save_button.connect("clicked", self.save_context_dialog)

            self.clear_button = Gtk.Button(label="🧹 Clear")
            self.clear_button.connect("clicked", lambda _button: self.submit_text("clear"))

            self.status_button = Gtk.Button(label="ℹ️ Status")
            self.status_button.connect("clicked", lambda _button: self.submit_text("status"))

            self.handoff_button = Gtk.Button(label="🗘 Handoff")
            self.handoff_button.connect("clicked", lambda _button: self.submit_text("handoff"))

            self.auto_confirm_check = Gtk.CheckButton(label="⚠️ Auto-confirm tools")
            self.auto_confirm_check.set_active(bool(args.no_confirm))
            self.auto_confirm_check.set_tooltip_text(
                "When disabled, write/run tool requests ask for confirmation in a dialog."
            )

            top_bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            top_bar.pack_start(self.open_button, False, False, 0)
            top_bar.pack_start(self.load_button, False, False, 0)
            top_bar.pack_start(self.save_button, False, False, 0)
            top_bar.pack_start(self.clear_button, False, False, 0)
            top_bar.pack_start(self.status_button, False, False, 0)
            top_bar.pack_start(self.handoff_button, False, False, 0)
            top_bar.pack_end(self.auto_confirm_check, False, False, 0)

            input_hint = Gtk.Label(label="Message (Ctrl+Enter to send):")
            input_hint.set_halign(Gtk.Align.START)

            bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            bottom.set_vexpand(False)
            bottom.set_size_request(1, GUI_INPUT_HEIGHT)
            self.send_button.set_valign(Gtk.Align.FILL)
            bottom.pack_start(self.input_scroll, True, True, 0)
            bottom.pack_start(self.send_button, False, False, 0)

            outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            outer.set_border_width(8)
            outer.pack_start(top_bar, False, False, 0)
            outer.pack_start(self.scroll, True, True, 0)
            outer.pack_start(input_hint, False, False, 0)
            outer.pack_start(bottom, False, False, 0)
            self.add(outer)

            self.append_markdown(
                f"# AI Agent\n\n"
                f"GUI: `{GTKUI_VERSION}`  \n"
                f"Agent: `{getattr(core, 'AGENT_VERSION', 'unknown')}`  \n"
                f"Backend: `{args.server}`  \n"
                f"Model: `{self.model}`  \n"
                f"Code preview: `{'GtkSourceView' if GtkSource is not None else 'plain TextBuffer'}`\n\n"
                "Use the multiline input box at the bottom to talk to the agent. "
                "Press **Ctrl+Enter** to send, or use the **Send** button. While the agent is running, the same button becomes **Stop**. "
                "Use **Open agent.md** to choose an instructions Markdown file; "
                "**Load**/**Save** work with LLM context JSON files. "
                "Special commands still work: `status`, `clear`, `save`, `load`, `handoff`, `exit`."
            )

            if args.open_md:
                try:
                    self.load_markdown_file(args.open_md)
                except Exception as exc:
                    self.append_markdown(
                        "## Could not open Markdown file\n\n" + _markdown_fence(str(exc))
                    )

            self.start_runtime()

        def start_runtime(self) -> None:
            try:
                from llm_backends import BackendConfig, make_backend

                cfg = BackendConfig(
                    server=self.args.server,
                    model=self.model,
                    url=self.args.url,
                    api_key=self.args.api_key,
                    prefer_json_schema=(not self.args.no_json_schema),
                )
                self.backend = make_backend(cfg)

                with contextlib.redirect_stdout(io.StringIO()) as stdout_buf, contextlib.redirect_stderr(io.StringIO()) as stderr_buf:
                    ok = self.backend.check_connection()

                captured = (stdout_buf.getvalue() + stderr_buf.getvalue()).strip()
                if captured:
                    self.append_markdown("## Backend check\n\n" + _markdown_fence(captured))

                if not ok:
                    self.set_busy(True)
                    self.append_markdown("## Backend connection failed\n\nCheck server/model settings and restart the GUI.")
                    return

                self.workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
                self.mcp = _start_mcp_for_workspace(self.workspace.name)
                self.append_markdown(
                    f"## Ready\n\nWorkspace: `{self.workspace.name}`\n"
                )
            except Exception as exc:
                self.set_busy(True)
                self.append_markdown("## Startup error\n\n" + _markdown_fence(str(exc)))

        def on_destroy(self, *_args: Any) -> None:
            self.cleanup()
            Gtk.main_quit()

        def cleanup(self) -> None:
            try:
                if self.mcp is not None:
                    self.mcp.close()
            except Exception:
                pass
            try:
                if self.workspace is not None:
                    self.workspace.cleanup()
            except Exception:
                pass

        def set_busy(self, busy: bool) -> None:
            self.busy = busy
            can_run = self.backend is not None and self.mcp is not None
            input_sensitive = (not busy) and can_run

            self.input_text_view.set_sensitive(input_sensitive)
            self.input_scroll.set_sensitive(input_sensitive)

            # During a running turn, keep the button active and repurpose it as
            # Stop.  This mirrors Ctrl-C in the CLI and avoids needing a second
            # control in the small bottom bar.
            self.send_button.set_sensitive(can_run)
            if busy:
                self.send_button.set_label("⛔ Stop")
                self.send_button.set_tooltip_text("Interrupt the current agent turn, like Ctrl-C in the CLI")
            else:
                self.send_button.set_label("➤ Send")
                self.send_button.set_tooltip_text("Send message to the agent. Ctrl+Enter works too.")

            self.open_button.set_sensitive(not busy)
            self.load_button.set_sensitive(not busy)
            self.save_button.set_sensitive(True)
            self.clear_button.set_sensitive(not busy)
            self.status_button.set_sensitive(not busy)
            self.handoff_button.set_sensitive((not busy) and can_run)

        def append_markdown(self, markdown_text: str) -> None:
            if not markdown_text:
                return
            fragment = str(markdown_text).rstrip() + "\n\n"
            self.transcript_markdown += fragment
            self.render_markdown_fragment(fragment)
            self.scroll_to_bottom()

        def append_markdown_from_worker(self, markdown_text: str) -> None:
            GLib.idle_add(lambda: (self.append_markdown(markdown_text), False)[1])

        def append_agent_output_from_worker(self, output_text: str) -> None:
            GLib.idle_add(lambda: (self.append_agent_output_block(output_text), False)[1])

        def append_agent_event_from_worker(self, title: str, body: str = "") -> None:
            GLib.idle_add(lambda: (self.append_agent_event_block(title, body), False)[1])

        def append_agent_event_block(self, title: str, body: str = "") -> None:
            title_text = str(title or "Agent event").strip() or "Agent event"
            body_text = str(body or "").strip()
            if body_text:
                if _looks_large_for_markdown(body_text):
                    body_text = _markdown_fence(_limit_visible_text(body_text), "text")
                block = f"### {title_text}\n\n{body_text}"
            else:
                block = f"### {title_text}"
            self.append_markdown(block)
            self.buffer.place_cursor(self.buffer.get_end_iter())
            self.scroll_to_bottom()

        def append_agent_output_block(self, output_text: str) -> None:
            """Append captured agent stdout/stderr as one compact block.

            The CLI output contains decorative separator lines intended for a
            terminal.  The GUI strips that frame and keeps one stable dark
            TextBuffer block, so clicking the transcript does not resize it and
            the output is much less noisy.
            """
            text = _compact_agent_console_output(output_text)
            if not text:
                text = "No visible output."
            text = _limit_visible_text(text)

            block = "Agent output\n" + text
            self.transcript_markdown += _markdown_fence(block) + "\n\n"

            self.insert("\n")
            self.insert(block.rstrip() + "\n", "code_block")
            self.insert("\n")
            self.buffer.place_cursor(self.buffer.get_end_iter())
            self.scroll_to_bottom()

        def scroll_to_bottom(self) -> None:
            def do_scroll() -> bool:
                # Layout is asynchronous after re-rendering markdown and after
                # adding child widgets such as code blocks.  Running this more
                # than once avoids the common GTK race where the adjustment
                # limits are still old during the first idle callback.
                end_iter = self.buffer.get_end_iter()
                self.buffer.place_cursor(end_iter)
                self.text_view.scroll_to_iter(end_iter, 0.0, False, 0.0, 1.0)

                adj = self.scroll.get_vadjustment()
                adj.set_value(max(0.0, adj.get_upper() - adj.get_page_size()))
                return False

            GLib.idle_add(do_scroll)
            GLib.timeout_add(50, do_scroll)
            GLib.timeout_add(150, do_scroll)

        def get_input_text(self) -> str:
            start = self.input_buffer.get_start_iter()
            end = self.input_buffer.get_end_iter()
            return self.input_buffer.get_text(start, end, True)

        def set_input_text(self, text: str) -> None:
            self.input_buffer.set_text(text or "")
            self.input_text_view.grab_focus()

        def clear_input_text(self) -> None:
            self.input_buffer.set_text("")

        def on_input_key_press(self, _widget: Any, event: Any) -> bool:
            if event.keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
                if event.state & Gdk.ModifierType.CONTROL_MASK:
                    self.on_send_clicked(None)
                    return True
            return False

        def on_input_buffer_changed(self, *_args: Any) -> None:
            # Keep the prompt editor a fixed-height scrollable area.  This is
            # what prevents long multiline prompts from changing the toplevel
            # window's natural size or decoration/state.
            try:
                self.input_scroll.set_size_request(1, GUI_INPUT_HEIGHT)
                self.input_text_view.scroll_to_mark(
                    self.input_buffer.get_insert(), 0.0, False, 0.0, 1.0
                )
            except Exception:
                pass

        def submit_text(self, text: str) -> None:
            self.set_input_text(text)
            self.on_send_clicked(None)

        def on_send_clicked(self, _button: Any) -> None:
            if self.busy:
                self.request_stop_current_turn()
                return

            user_input = self.get_input_text().strip()
            if not user_input:
                return

            self.clear_input_text()
            self.append_markdown("## You\n\n" + user_input)

            if user_input.lower() == "exit":
                self.cleanup()
                Gtk.main_quit()
                return

            auto_confirm = bool(self.auto_confirm_check.get_active())
            self.stop_requested = False
            self.cancel_event = threading.Event()
            self.set_busy(True)

            worker = threading.Thread(
                target=self.worker_handle_input,
                args=(user_input, auto_confirm),
                daemon=True,
            )
            self.worker_thread = worker
            worker.start()

        def request_stop_current_turn(self) -> None:
            if not self.busy:
                return
            if self.stop_requested:
                return

            self.stop_requested = True
            try:
                self.cancel_event.set()
            except Exception:
                pass

            self.send_button.set_sensitive(False)
            self.send_button.set_label("Stopping…")
            self.append_agent_event_block(
                "⛔ Stop requested",
                "Trying to interrupt the current agent turn. This is similar to Ctrl-C in the CLI. "
                "If the worker is inside a non-interruptible OS/native call, it may stop only after that call returns.",
            )

            if not _raise_async_exception(self.worker_ident, KeyboardInterrupt):
                self.append_agent_event_block(
                    "⚠️ Stop notice",
                    "Could not inject KeyboardInterrupt into the worker thread. "
                    "A cooperative cancel flag was still set, so cancellable core/tool code will stop at the next check.",
                )

        def finish_worker_turn(self) -> None:
            self.worker_thread = None
            self.worker_ident = None
            self.stop_requested = False
            self.cancel_event = threading.Event()
            self.set_busy(False)

        def worker_handle_input(self, user_input: str, auto_confirm: bool) -> None:
            self.worker_ident = threading.get_ident()
            try:
                output = self.handle_input_in_worker(user_input, auto_confirm)
                if isinstance(output, tuple) and len(output) == 2 and output[0] == "agent_output":
                    self.append_agent_output_from_worker(str(output[1]))
                elif output:
                    self.append_markdown_from_worker(str(output))
            except KeyboardInterrupt:
                self.append_markdown_from_worker("## ⛔ Turn interrupted\n\nStopped by user.")
            except Exception as exc:
                if self.stop_requested:
                    self.append_markdown_from_worker("## ⛔ Turn interrupted\n\nStopped by user.")
                else:
                    self.append_markdown_from_worker("## GUI worker error\n\n" + _markdown_fence(str(exc)))
            finally:
                GLib.idle_add(lambda: (self.finish_worker_turn(), False)[1])

        def handle_input_in_worker(self, user_input: str, auto_confirm: bool) -> Any:
            cmd = user_input.lower()

            if cmd == "status":
                workspace_name = self.workspace.name if self.workspace else "<not started>"
                return (
                    "## Status\n\n"
                    f"Server: `{self.args.server}`  \n"
                    f"Model: `{self.model}`  \n"
                    f"Messages: `{len(self.messages)}`  \n"
                    f"Workspace: `{workspace_name}`"
                )

            if cmd == "clear":
                old = self.workspace.name if self.workspace else "<none>"
                try:
                    if self.mcp is not None:
                        self.mcp.close()
                except Exception:
                    pass
                try:
                    if self.workspace is not None:
                        self.workspace.cleanup()
                except Exception:
                    pass

                self.messages = [self.messages[0]]
                self.workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
                self.mcp = _start_mcp_for_workspace(self.workspace.name)
                GLib.idle_add(lambda: (self.reset_transcript(), False)[1])
                return (
                    "## Context cleared\n\n"
                    f"Old workspace: `{old}`  \n"
                    f"New workspace: `{self.workspace.name}`"
                )

            if cmd == "save":
                with open("ctx.json", "wt", encoding="utf-8") as fd:
                    json.dump(self.messages, fd)
                return "## Context saved\n\nSaved to `ctx.json`."

            if cmd == "load":
                with open("ctx.json", "rt", encoding="utf-8") as fd:
                    self.messages = json.load(fd)
                user_input = "Remind me, please, what has been done?"

            if cmd == "handoff":
                with contextlib.redirect_stdout(io.StringIO()) as stdout_buf, contextlib.redirect_stderr(io.StringIO()) as stderr_buf:
                    handoff_summary = _request_handoff_summary(
                        backend=self.backend,
                        messages=self.messages,
                        max_tokens=self.args.max_tokens,
                    )
                    project_status = _build_project_status(self.workspace.name)

                captured = (stdout_buf.getvalue() + stderr_buf.getvalue()).strip()
                self.messages = [self.messages[0]]
                handoff_payload = (
                    "HANDOFF CONTEXT\n"
                    "===============\n\n"
                    "Summary from previous context:\n"
                    f"{handoff_summary}\n\n"
                    "Current project status:\n"
                    f"{project_status}\n\n"
                    "Continue from this state. "
                    "The workspace and current project files were preserved during handoff."
                )
                self.messages.append({"role": "user", "content": handoff_payload})
                result = "## Handoff complete\n\nContext was reset, workspace preserved."
                if captured:
                    result += "\n\n" + _markdown_fence(captured)
                return result

            live_events = {"count": 0}

            def live_event(title: str, body: str = "") -> None:
                live_events["count"] += 1
                self.append_agent_event_from_worker(title, body)

            kwargs = {
                "backend": self.backend,
                "mcp": self.mcp,
                "messages": self.messages,
                "user_input": user_input,
                "max_tokens": self.args.max_tokens,
                "temperature": self.args.temp,
                "no_confirm": auto_confirm,
                "workdir": self.workspace.name,
                "debug": self.args.debug,
            }
            old_input = None
            try:
                params = inspect.signature(run_agent_turn).parameters
            except Exception:
                params = {}

            if "confirm_func" in params:
                kwargs["confirm_func"] = self.confirm_from_worker
            elif not auto_confirm:
                old_input = builtins.input

                def gui_input(prompt: str = "") -> str:
                    return "y" if self.confirm_from_worker(prompt) else "n"

                builtins.input = gui_input

            if "cancel_event" in params:
                kwargs["cancel_event"] = self.cancel_event

            if "event_callback" in params:
                kwargs["event_callback"] = live_event
            elif not self._warned_no_event_callback:
                self._warned_no_event_callback = True
                self.append_agent_event_from_worker(
                    "GUI compatibility",
                    "This agent.py core has no event_callback hook, so live reasoning/tool events are unavailable. "
                    "Captured agent output will appear after the turn finishes.",
                )

            try:
                with contextlib.redirect_stdout(io.StringIO()) as stdout_buf, contextlib.redirect_stderr(io.StringIO()) as stderr_buf:
                    run_agent_turn(**kwargs)
            except KeyboardInterrupt:
                return "## ⛔ Turn interrupted\n\nStopped by user."
            finally:
                if old_input is not None:
                    builtins.input = old_input

            if self.stop_requested:
                return "## ⛔ Turn interrupted\n\nStopped by user."

            if len(self.messages) > 21:
                self.messages = [self.messages[0]] + self.messages[-20:]

            captured = (stdout_buf.getvalue() + stderr_buf.getvalue()).strip()
            if live_events["count"] == 0:
                if captured:
                    return ("agent_output", captured)
                return ("agent_output", "No visible output.")
            return None

        def reset_transcript(self) -> None:
            self.transcript_markdown = (
                f"# AI Agent\n\n"
                f"GUI: `{GTKUI_VERSION}`  \n"
                f"Agent: `{getattr(core, 'AGENT_VERSION', 'unknown')}`  \n"
                f"Backend: `{self.args.server}`  \n"
                f"Model: `{self.model}`  \n"
                f"Code preview: `{'GtkSourceView' if GtkSource is not None else 'plain TextBuffer'}`\n\n"
            )
            self.render_markdown(self.transcript_markdown)

        def confirm_from_worker(self, prompt: str) -> bool:
            done = threading.Event()
            answer = {"value": False}

            def ask() -> bool:
                dialog = Gtk.MessageDialog(
                    parent=self,
                    flags=Gtk.DialogFlags.MODAL,
                    message_type=Gtk.MessageType.WARNING,
                    buttons=Gtk.ButtonsType.YES_NO,
                    text="Confirm agent tool execution",
                )
                dialog.format_secondary_text(prompt)
                response = dialog.run()
                dialog.destroy()
                answer["value"] = response == Gtk.ResponseType.YES
                done.set()
                return False

            GLib.idle_add(ask)
            while not done.wait(0.1):
                if self.stop_requested:
                    return False
            return bool(answer["value"])

        def insert_plain_code_block(self, code: str) -> None:
            # Stable path for plain text, logs, tool output, and very large
            # code blocks. It stays inside the parent Gtk.TextBuffer, so it
            # cannot force the top-level window to resize.
            text = _limit_visible_text(code).rstrip("\n")
            if text:
                self.insert(text + "\n", "code_block")
            else:
                self.insert("\n", "code_block")
            self.insert("\n")

        def _normalize_source_lang(self, lang: Optional[str]) -> str:
            if not lang:
                return ""
            return str(lang).strip().lower().split()[0]

        def _should_use_source_preview(self, code: str, lang: Optional[str]) -> bool:
            normalized_lang = self._normalize_source_lang(lang)
            if GtkSource is None or not GUI_SOURCE_PREVIEW:
                return False
            if normalized_lang in _PLAIN_CODE_LANGS:
                return False
            # Tool output commonly arrives as fenced json/text and can be huge.
            # Keep that on the stable TextBuffer path.
            if len(code) > GUI_SOURCE_PREVIEW_MAX_CHARS:
                return False
            if code.count("\n") + 1 > GUI_SOURCE_PREVIEW_MAX_LINES:
                return False
            return True

        def insert_source_code_block(self, code: str, lang: Optional[str]) -> None:
            normalized_lang = self._normalize_source_lang(lang)

            it = self.buffer.get_end_iter()
            anchor = self.buffer.create_child_anchor(it)

            source_buffer = GtkSource.Buffer()
            source_buffer.set_text(code.rstrip("\n") + "\n")

            lang_manager = GtkSource.LanguageManager.get_default()
            lang_id = _SOURCE_LANG_ALIASES.get(normalized_lang, normalized_lang)
            language = lang_manager.get_language(lang_id) if lang_id else None
            if language is not None:
                source_buffer.set_language(language)
            source_buffer.set_highlight_syntax(True)

            style_manager = GtkSource.StyleSchemeManager.get_default()
            for scheme_name in ("oblivion", "cobalt", "classic"):
                scheme = style_manager.get_scheme(scheme_name)
                if scheme is not None:
                    source_buffer.set_style_scheme(scheme)
                    break

            source_view = GtkSource.View.new_with_buffer(source_buffer)
            source_view.set_editable(False)
            source_view.set_cursor_visible(False)
            source_view.set_can_focus(False)
            try:
                source_view.set_focus_on_click(False)
            except Exception:
                pass
            source_view.set_monospace(True)
            source_view.set_show_line_numbers(True)
            source_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
            source_view.set_left_margin(10)
            source_view.set_right_margin(10)
            source_view.set_top_margin(8)
            source_view.set_bottom_margin(8)
            source_view.set_hexpand(False)
            source_view.set_vexpand(False)

            lines = code.rstrip("\n").split("\n") or [""]
            visual_lines = 0
            for line in lines:
                # Approximate wrapping height without letting the widget ask
                # for an unbounded natural height.
                visual_lines += max(1, min(6, (len(line) // 100) + 1))
            visual_lines = max(3, min(GUI_SOURCE_PREVIEW_MAX_VISUAL_LINES, visual_lines))
            height = 22 * visual_lines + 24

            try:
                available_w = self.text_view.get_allocated_width() - 48
                win_w = self.get_allocated_width() - 80
                width = max(320, min(available_w, win_w, 1200))
            except Exception:
                width = 760
            if width <= 1:
                width = 760

            # Safety difference from the old implementation: no GtkFrame,
            # no huge natural width, no width derived from long source lines.
            source_view.set_size_request(width, height)
            source_view.override_background_color(
                Gtk.StateFlags.NORMAL,
                Gdk.RGBA(0.20, 0.20, 0.20, 1.0),
            )
            source_view.show_all()

            self.text_view.add_child_at_anchor(source_view, anchor)
            self.insert("\n\n")

        def insert_code_block(self, code: str, lang: Optional[str] = None) -> None:
            if self._should_use_source_preview(code, lang):
                self.insert_source_code_block(code, lang)
            else:
                self.insert_plain_code_block(code)

        def insert_checkbox(self, checked: bool) -> None:
            it = self.buffer.get_end_iter()
            anchor = self.buffer.create_child_anchor(it)
            checkbox = Gtk.CheckButton()
            checkbox.set_active(checked)
            checkbox.set_sensitive(False)
            checkbox.show()
            self.text_view.add_child_at_anchor(checkbox, anchor)

        def create_tags(self) -> None:
            self.buffer.create_tag(
                "h1",
                weight=Pango.Weight.BOLD,
                scale=1.8,
                pixels_above_lines=12,
                pixels_below_lines=8,
            )
            self.buffer.create_tag(
                "h2",
                weight=Pango.Weight.BOLD,
                scale=1.5,
                pixels_above_lines=10,
                pixels_below_lines=6,
            )
            self.buffer.create_tag(
                "h3",
                weight=Pango.Weight.BOLD,
                scale=1.25,
                pixels_above_lines=8,
                pixels_below_lines=5,
            )
            self.buffer.create_tag("bold", weight=Pango.Weight.BOLD)
            self.buffer.create_tag("italic", style=Pango.Style.ITALIC)
            self.buffer.create_tag(
                "code",
                family="monospace",
                foreground="#dddddd",
                background="#333333",
            )
            self.buffer.create_tag(
                "code_block",
                family="monospace",
                foreground="#dddddd",
                background="#333333",
                left_margin=16,
                right_margin=16,
                pixels_above_lines=8,
                pixels_below_lines=8,
            )
            self.buffer.create_tag(
                "agent_output_title",
                weight=Pango.Weight.BOLD,
                left_margin=16,
                right_margin=16,
                pixels_above_lines=10,
                pixels_below_lines=4,
            )
            self.buffer.create_tag("quote", left_margin=24, style=Pango.Style.ITALIC)
            self.buffer.create_tag(
                "link",
                underline=Pango.Underline.SINGLE,
                foreground="blue",
            )

        def insert(self, text: str, *tags: str) -> None:
            it = self.buffer.get_end_iter()
            if tags:
                self.buffer.insert_with_tags_by_name(it, text, *tags)
            else:
                self.buffer.insert(it, text)

        def render_inline(self, children: Any, inherited_tags: Tuple[str, ...] = ()) -> None:
            if not children:
                return

            for node in children:
                node_type = node.get("type")
                if node_type == "text":
                    self.insert(node.get("raw", ""), *inherited_tags)
                elif node_type == "strong":
                    self.render_inline(node.get("children", []), inherited_tags + ("bold",))
                elif node_type == "emphasis":
                    self.render_inline(node.get("children", []), inherited_tags + ("italic",))
                elif node_type == "codespan":
                    self.insert(node.get("raw", ""), *(inherited_tags + ("code",)))
                elif node_type == "linebreak":
                    # Markdown hard line break: two trailing spaces before newline.
                    # Used by status text like: f"Server: `...`  \n".
                    self.insert("\n", *inherited_tags)
                elif node_type == "softbreak":
                    # Preserve source newlines instead of flattening everything into
                    # one long line. This is friendlier for GUI transcript output.
                    self.insert("\n", *inherited_tags)
                elif node_type == "link":
                    url = node.get("attrs", {}).get("url", "")
                    self.render_inline(node.get("children", []), inherited_tags + ("link",))
                    if url:
                        self.insert(f" ({url})", *(inherited_tags + ("link",)))
                elif "children" in node:
                    self.render_inline(node["children"], inherited_tags)

        def render_block(self, node: Dict[str, Any]) -> None:
            node_type = node.get("type")

            if node_type == "blank_line":
                return

            if node_type == "heading":
                level = node.get("attrs", {}).get("level", 1)
                tag = f"h{min(level, 3)}"
                self.render_inline(node.get("children", []), (tag,))
                self.insert("\n\n")
                return

            if node_type == "paragraph":
                self.render_inline(node.get("children", []))
                self.insert("\n\n")
                return

            if node_type == "block_code":
                code = node.get("raw", "")
                lang = node.get("attrs", {}).get("info")
                if lang:
                    lang = lang.strip().split()[0]
                self.insert_code_block(code, lang)
                return

            if node_type == "block_quote":
                for child in node.get("children", []):
                    start_offset = self.buffer.get_char_count()
                    self.render_block(child)
                    end_offset = self.buffer.get_char_count()
                    if end_offset > start_offset:
                        self.buffer.apply_tag_by_name(
                            "quote",
                            self.buffer.get_iter_at_offset(start_offset),
                            self.buffer.get_iter_at_offset(end_offset),
                        )
                return

            if node_type == "list":
                ordered = node.get("attrs", {}).get("ordered", False)
                for index, item in enumerate(node.get("children", []), start=1):
                    if item.get("type") == "task_list_item":
                        checked = item.get("attrs", {}).get("checked", False)
                        self.insert_checkbox(checked)
                        self.insert(" ")
                    else:
                        prefix = f"{index}. " if ordered else "• "
                        self.insert(prefix)

                    for child in item.get("children", []):
                        if child.get("type") == "block_text":
                            self.render_inline(child.get("children", []))
                        else:
                            self.render_block(child)
                    self.insert("\n")
                self.insert("\n")
                return

            if node_type == "thematic_break":
                self.insert("────────────\n\n")
                return

            if "children" in node:
                for child in node["children"]:
                    if isinstance(child, dict):
                        self.render_block(child)

        def render_markdown_fragment(self, markdown_text: str) -> None:
            tokens = self.md_parser(markdown_text)
            for node in tokens:
                self.render_block(node)
            self.buffer.place_cursor(self.buffer.get_end_iter())

        def render_markdown(self, markdown_text: str) -> None:
            self.buffer.set_text("")
            self.render_markdown_fragment(markdown_text)

        def load_markdown_file(self, filename: str) -> None:
            path = Path(filename).expanduser().resolve()
            text = path.read_text(encoding="utf-8")
            self.transcript_markdown = text.rstrip() + "\n\n"
            self.render_markdown(self.transcript_markdown)
            self.set_title(f"AI Agent — {path.name}")

        def find_agent_md(self) -> Optional[Path]:
            candidates = [
                Path.cwd() / "agent.md",
                Path(getattr(core, "__file__", "agent.py")).resolve().with_name("agent.md"),
                Path(__file__).resolve().with_name("agent.md"),
            ]

            seen = set()
            for candidate in candidates:
                try:
                    resolved = candidate.expanduser().resolve()
                except Exception:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                if resolved.is_file():
                    return resolved
            return None

        def add_text_file_filters(self, dialog: Any) -> None:
            md_filter = Gtk.FileFilter()
            md_filter.set_name("Markdown / text files")
            for pattern in ("*.md", "*.markdown", "*.txt", "*.json", "*.yaml", "*.yml"):
                md_filter.add_pattern(pattern)
            dialog.add_filter(md_filter)

            all_filter = Gtk.FileFilter()
            all_filter.set_name("All files")
            all_filter.add_pattern("*")
            dialog.add_filter(all_filter)

        def add_context_file_filters(self, dialog: Any) -> None:
            json_filter = Gtk.FileFilter()
            json_filter.set_name("LLM context JSON")
            json_filter.add_pattern("*.json")
            dialog.add_filter(json_filter)

            all_filter = Gtk.FileFilter()
            all_filter.set_name("All files")
            all_filter.add_pattern("*")
            dialog.add_filter(all_filter)

        def open_agent_md(self, _button: Any) -> None:
            dialog = Gtk.FileChooserDialog(
                title="Open agent instructions",
                parent=self,
                action=Gtk.FileChooserAction.OPEN,
            )
            dialog.add_buttons(
                "_Cancel", Gtk.ResponseType.CANCEL,
                "_Open", Gtk.ResponseType.ACCEPT,
            )
            self.add_text_file_filters(dialog)

            default_agent_md = self.find_agent_md()
            if default_agent_md is not None:
                dialog.set_filename(str(default_agent_md))
            else:
                dialog.set_current_name("agent.md")

            try:
                if dialog.run() != Gtk.ResponseType.ACCEPT:
                    return
                filename = dialog.get_filename()
                if not filename:
                    return
                self.load_markdown_file(filename)
            except Exception as exc:
                self.append_markdown(
                    "## Could not open selected file\n\n" + _markdown_fence(str(exc))
                )
            finally:
                dialog.destroy()

        def validate_context_messages(self, obj: Any) -> List[Dict[str, str]]:
            if not isinstance(obj, list):
                raise ValueError("Context file must contain a JSON list of messages.")

            validated: List[Dict[str, str]] = []
            for index, item in enumerate(obj):
                if not isinstance(item, dict):
                    raise ValueError(f"Message #{index + 1} is not an object.")
                role = item.get("role")
                content = item.get("content")
                if not isinstance(role, str) or not isinstance(content, str):
                    raise ValueError(
                        f"Message #{index + 1} must have string 'role' and 'content' fields."
                    )
                if role not in ("system", "user", "assistant", "tool"):
                    raise ValueError(f"Message #{index + 1} has unsupported role: {role!r}.")
                validated.append({"role": role, "content": content})

            if not validated or validated[0].get("role") != "system":
                validated.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            return validated

        def load_context_dialog(self, _button: Any) -> None:
            if self.busy:
                return

            dialog = Gtk.FileChooserDialog(
                title="Load LLM context",
                parent=self,
                action=Gtk.FileChooserAction.OPEN,
            )
            dialog.add_buttons(
                "_Cancel", Gtk.ResponseType.CANCEL,
                "_Load", Gtk.ResponseType.ACCEPT,
            )
            self.add_context_file_filters(dialog)
            dialog.set_current_name("ctx.json")

            try:
                if dialog.run() != Gtk.ResponseType.ACCEPT:
                    return
                filename = dialog.get_filename()
                if not filename:
                    return

                path = Path(filename).expanduser().resolve()
                data = json.loads(path.read_text(encoding="utf-8"))
                self.messages = self.validate_context_messages(data)
                self.append_markdown(
                    "## Context loaded\n\n"
                    f"Loaded `{len(self.messages)}` messages from `{path}`."
                )
            except Exception as exc:
                self.append_markdown(
                    "## Could not load context\n\n" + _markdown_fence(str(exc))
                )
            finally:
                dialog.destroy()

        def save_context_dialog(self, _button: Any) -> None:
            dialog = Gtk.FileChooserDialog(
                title="Save LLM context",
                parent=self,
                action=Gtk.FileChooserAction.SAVE,
            )
            dialog.add_buttons(
                "_Cancel", Gtk.ResponseType.CANCEL,
                "_Save", Gtk.ResponseType.ACCEPT,
            )
            dialog.set_do_overwrite_confirmation(True)
            dialog.set_current_name("ctx.json")
            self.add_context_file_filters(dialog)

            try:
                if dialog.run() != Gtk.ResponseType.ACCEPT:
                    return
                filename = dialog.get_filename()
                if not filename:
                    return

                path = Path(filename).expanduser().resolve()
                messages_snapshot = [dict(message) for message in list(self.messages)]
                path.write_text(
                    json.dumps(messages_snapshot, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self.append_markdown(
                    "## Context saved\n\n"
                    f"Saved `{len(messages_snapshot)}` messages to `{path}`."
                )
            except Exception as exc:
                self.append_markdown(
                    "## Could not save context\n\n" + _markdown_fence(str(exc))
                )
            finally:
                dialog.destroy()

        # Backward-compatible alias for older code that called open_markdown().
        def open_markdown(self, _button: Any) -> None:
            self.open_agent_md(_button)

    win = AgentGuiWindow()
    win.show_all()
    Gtk.main()


# Backward-compatible name kept for older launch scripts.
def run_markdown_viewer(initial_file: Optional[str] = None, core: Optional[Any] = None) -> None:
    if core is None:
        core = _load_default_core()
    args = _make_default_args(core)
    args.open_md = initial_file
    run_agent_gui(args, core=core)


def main() -> None:
    if any(arg in ("--version", "-V") for arg in sys.argv[1:]):
        print(f"agent_gui.py {GTKUI_VERSION}")
        return

    core = _load_default_core()
    args = _make_default_args(core)
    run_agent_gui(args, core=core)


if __name__ == "__main__":
    main()
