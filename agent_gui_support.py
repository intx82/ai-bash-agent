#!/usr/bin/env python3
"""GTK3 UI extension: backend connection controls and collapsible chat history.

This file deliberately imports no GTK modules at import time, so the core
remains importable on headless systems. ``agent_gui.py`` passes Gtk/GLib in.
"""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from agent_chat_store import CHAT_FORMAT, ChatStore, format_timestamp


class ChatUIV24Mixin:
    def _install_chat_layout(self, outer: Any, top_bar: Any) -> None:
        Gtk = self._gtk
        self.connection_pending = False
        self.current_chat_path: Optional[Path] = None
        self.current_chat_signature = None
        self._saved_chat_fingerprint = None
        self.chat_store = ChatStore(Path(os.environ.get("AGENT_CHAT_DIR", "save")))
        self.chat_title_label = Gtk.Label(label="Unsaved chat")
        self.chat_title_label.set_ellipsize(self._pango.EllipsizeMode.END)
        self.chat_title_label.set_max_width_chars(32)
        self.chat_title_label.set_tooltip_text("Current saved chat filename")

        self.sidebar_toggle = Gtk.ToggleButton(label="☰ Chats")
        self.sidebar_toggle.set_active(True)
        self.sidebar_toggle.set_tooltip_text("Show or hide saved chats")
        self.sidebar_toggle.connect("toggled", self._on_sidebar_toggle_v24)

        self.connect_button = Gtk.Button(label="🔌 Connect")
        self.connect_button.set_tooltip_text("Choose backend, URL, model, and API key")
        self.connect_button.connect("clicked", self._on_connect_clicked_v24)
        self.connection_status = Gtk.Label(label="Disconnected")
        self.connection_status.set_ellipsize(self._pango.EllipsizeMode.END)
        self.connection_status.set_max_width_chars(28)
        self.connection_status.set_tooltip_text("LLM connection status")
        top_bar.pack_start(self.sidebar_toggle, False, False, 0)
        top_bar.pack_start(self.chat_title_label, False, False, 0)
        top_bar.pack_start(self.connect_button, False, False, 0)
        top_bar.pack_start(self.connection_status, False, False, 0)

        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        sidebar.set_border_width(6)
        sidebar.set_size_request(235, -1)
        sidebar_title = Gtk.Label(label="save/*.json")
        sidebar_title.set_xalign(0)
        sidebar.pack_start(sidebar_title, False, False, 0)

        actions = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.new_chat_button = Gtk.Button(label="＋ New named chat")
        self.new_chat_button.connect("clicked", self._new_chat_v24)
        self.refresh_chat_button = Gtk.Button(label="↻")
        self.refresh_chat_button.set_tooltip_text("Reload saved chat list")
        self.refresh_chat_button.connect("clicked", self._refresh_chats_v24)
        actions.pack_start(self.new_chat_button, True, True, 0)
        actions.pack_start(self.refresh_chat_button, False, False, 0)
        sidebar.pack_start(actions, False, False, 0)

        self.chat_list = Gtk.ListBox()
        self.chat_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.chat_list.connect("row-activated", self._on_chat_activated_v24)
        self.chat_list_scroll = Gtk.ScrolledWindow()
        self.chat_list_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.chat_list_scroll.add(self.chat_list)
        sidebar.pack_start(self.chat_list_scroll, True, True, 0)

        # Do not use Gtk.Paned here: its saved divider position and non-shrinkable
        # first pane continue to reserve the sidebar width even after the
        # Gtk.Revealer hides its contents. A plain Gtk.Box gives the transcript
        # the entire freed width when the sidebar widget is hidden.
        self.sidebar_panel = sidebar
        # Remember the width without the sidebar. Without explicit window
        # resizing GTK3 can enlarge the toplevel on every reveal but leave its
        # expanded width unchanged when the sidebar is hidden.
        self._sidebar_collapsed_width = None
        self._sidebar_expanded_width = None
        self._sidebar_resize_serial = 0
        self.chat_split = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=0)
        self.chat_split.pack_start(self.sidebar_panel, False, False, 0)
        self.chat_split.pack_start(outer, True, True, 0)
        self.add(self.chat_split)
        self._refresh_chats_v24()

    def _on_sidebar_toggle_v24(self, button: Any) -> None:
        show = bool(button.get_active())
        was_visible = self.sidebar_panel.get_visible()
        if show == was_visible:
            return

        # GTK does NOT automatically shrink a top-level window when children
        # disappear. Repeated reveals otherwise add 235px every time.
        window_width, window_height = self.get_size()
        sidebar_width = 235
        if was_visible:
            sidebar_width = max(sidebar_width,
                                self.sidebar_panel.get_allocated_width())

        if show:
            # A hidden window may have been manually resized; use its actual
            # current width rather than the last requested width.
            if self._sidebar_collapsed_width is None:
                self._sidebar_collapsed_width = window_width
            elif self._sidebar_expanded_width is None:
                # When hidden, its width belongs entirely to the chat pane.
                # Ignore a stale size report from an immediate rapid toggle.
                if abs(window_width - self._sidebar_collapsed_width) < 80:
                    self._sidebar_collapsed_width = window_width
            target_width = self._sidebar_collapsed_width + sidebar_width
            # Opening chats must never push the window wider than the monitor.
            try:
                display = self.get_display()
                monitor = display.get_monitor_at_window(self.get_window())
                target_width = min(target_width, max(1, monitor.get_workarea().width - 16))
            except (AttributeError, TypeError):
                pass
            self._sidebar_expanded_width = target_width
        else:
            if self._sidebar_collapsed_width is None:
                self._sidebar_collapsed_width = max(1, window_width - sidebar_width)
            target_width = self._sidebar_collapsed_width
            self._sidebar_expanded_width = None

        self.sidebar_panel.set_visible(show)
        # GTK's deferred relayout may override the first resize. A second
        # resize after allocation enforces the target; serial guards prevent
        # stale callbacks from an earlier, rapid toggle overriding a new one.
        self._sidebar_resize_serial += 1
        serial = self._sidebar_resize_serial
        self.resize(max(1, target_width), max(1, window_height))

        def settle_width() -> bool:
            if serial == self._sidebar_resize_serial:
                self.resize(max(1, target_width), max(1, window_height))
            return False

        self._glib.idle_add(settle_width)
        self._glib.timeout_add(100, settle_width)

    def _refresh_chats_v24(self, *_args: Any) -> None:
        Gtk = self._gtk
        for row in self.chat_list.get_children():
            self.chat_list.remove(row)
        for entry in self.chat_store.list_chats():
            row = Gtk.ListBoxRow()
            row.chat_path = entry.path
            col = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            col.set_border_width(5)
            title = Gtk.Label(label=entry.title)
            title.set_xalign(0)
            title.set_max_width_chars(28)
            title.set_ellipsize(self._pango.EllipsizeMode.END)
            title.set_tooltip_text(entry.title)
            meta = Gtk.Label(label=format_timestamp(entry.modified) +
                             f" · {entry.size_bytes // 1024} KiB")
            meta.set_xalign(0)
            meta.set_max_width_chars(32)
            meta.set_ellipsize(self._pango.EllipsizeMode.END)
            col.pack_start(title, False, False, 0)
            col.pack_start(meta, False, False, 0)
            row.add(col)
            self.chat_list.add(row)
            if self.current_chat_path == entry.path:
                self.chat_list.select_row(row)
        self.chat_list.show_all()

    def _public_profile_v24(self) -> Dict[str, str]:
        # Connection details remain in memory; traditional chats are bare arrays.
        return {"server": self.args.server, "url": self.args.url,
                "model": self.model}

    def _chat_fingerprint_v24(self) -> str:
        # Discovered MCP metadata changes the system message, but does not
        # constitute a user edit to an existing saved conversation.
        payload = json.dumps(self.messages[1:], ensure_ascii=False,
                             sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _save_current_chat_v24(self, *, silent: bool = False) -> bool:
        path = self.current_chat_path
        if path is None or len(self.messages) < 2:
            return False
        fingerprint = self._chat_fingerprint_v24()
        if fingerprint == self._saved_chat_fingerprint and path.exists():
            return True
        try:
            self.chat_store.save(
                path, list(self.messages), self.transcript_markdown,
                expected_signature=self.current_chat_signature,
                expect_missing=self.current_chat_signature is None)
            self.current_chat_signature = self.chat_store.signature(path)
            self._saved_chat_fingerprint = fingerprint
            self.chat_title_label.set_text(path.name)
            if not silent:
                self._refresh_chats_v24()
            return True
        except Exception as exc:
            # Never silently discard errors even during automatic saving or
            # chat switching: the user must know what was not persisted.
            self.append_markdown("## Chat save failed\n\n" + str(exc))
            return False

    def _save_command_v24(self) -> bool:
        # Called on the GTK main thread in response to typed "save". The
        # worker may still be finishing; defer the chooser until it is idle.
        if self.busy:
            self._glib.timeout_add(100, self._save_command_v24)
            return False
        if self.current_chat_path is None:
            self._save_context_dialog_v24(None)
        else:
            self._save_chat_after_turn_v24()
        return False

    def _save_chat_after_turn_v24(self) -> bool:
        # Called after the agent's live-event callbacks finish.
        self._save_current_chat_v24()
        return False

    def _confirm_leave_chat_v24(self) -> bool:
        if self.current_chat_path is not None:
            if self._chat_fingerprint_v24() != self._saved_chat_fingerprint:
                return self._save_current_chat_v24(silent=True)
            return True
        if len(self.messages) <= 1:
            return True
        Gtk = self._gtk
        dialog = Gtk.MessageDialog(
            parent=self, flags=Gtk.DialogFlags.MODAL,
            message_type=Gtk.MessageType.WARNING,
            buttons=Gtk.ButtonsType.YES_NO,
            text="Discard unsaved conversation?")
        dialog.format_secondary_text(
            "This chat has no saved filename. Click Save first if you need it later.")
        try:
            return dialog.run() == Gtk.ResponseType.YES
        finally:
            dialog.destroy()

    def _ask_new_chat_filename_v24(self) -> Optional[Path]:
        Gtk = self._gtk
        dialog = Gtk.Dialog(title="New chat filename", transient_for=self,
                            flags=Gtk.DialogFlags.MODAL)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                           "_Create", Gtk.ResponseType.ACCEPT)
        box = dialog.get_content_area()
        box.set_spacing(8)
        box.set_border_width(12)
        label = Gtk.Label(label="Save as (for example: device-case.json):")
        label.set_xalign(0)
        entry = Gtk.Entry()
        entry.set_placeholder_text("device-case.json")
        entry.set_activates_default(True)
        dialog.set_default_response(Gtk.ResponseType.ACCEPT)
        box.pack_start(label, False, False, 0)
        box.pack_start(entry, False, False, 0)
        dialog.show_all()
        try:
            if dialog.run() != Gtk.ResponseType.ACCEPT:
                return None
            path = self.chat_store.named_path(entry.get_text())
            if path.exists():
                raise FileExistsError(
                    f"{path.name} already exists. Select it in the chat list, "
                    "or choose a different name.")
            return path
        finally:
            dialog.destroy()

    def _new_chat_v24(self, _button: Any) -> None:
        if self.busy or self.connection_pending:
            return
        try:
            path = self._ask_new_chat_filename_v24()
        except (ValueError, FileExistsError) as exc:
            self.append_markdown("## Cannot create chat\n\n" + str(exc))
            return
        if path is None or not self._confirm_leave_chat_v24():
            return
        self.current_chat_path = path
        self.current_chat_signature = None
        self._saved_chat_fingerprint = None
        self.messages = [{"role": "system", "content": self._system_prompt_v24}]
        self.reset_transcript()
        self.chat_title_label.set_text(path.name)
        self._refresh_chats_v24()

    def _on_chat_activated_v24(self, _listbox: Any, row: Any) -> None:
        if self.busy or self.connection_pending:
            return
        self._load_chat_v24(row.chat_path)

    def _load_chat_v24(self, path: Path) -> None:
        path = Path(path).expanduser().resolve()
        if path == self.current_chat_path:
            return
        try:
            saved = self.chat_store.load(path)
        except (OSError, ValueError, TypeError) as exc:
            self.append_markdown("## Cannot load chat\n\n" + str(exc))
            return
        if not self._confirm_leave_chat_v24():
            return
        messages = saved["messages"]
        if messages and messages[0].get("role") == "system":
            # Refresh the live tool catalog before this conversation continues.
            messages[0]["content"] = self._system_prompt_v24
        else:
            messages.insert(0, {"role": "system", "content": self._system_prompt_v24})
        self.messages = messages
        self.current_chat_path = path
        self.current_chat_signature = saved["signature"]
        self._saved_chat_fingerprint = self._chat_fingerprint_v24()
        self.transcript_markdown = (saved["transcript"]
                                    or self._legacy_chat_header_v24())
        self.render_markdown(self.transcript_markdown)
        self.chat_title_label.set_text(path.name)
        self._refresh_chats_v24()
        profile = saved.get("profile", {})
        if profile.get("model") and profile["model"] != self.model:
            self.connection_status.set_tooltip_text(
                "Saved chat used model " + profile["model"] +
                "; current connection is unchanged")

    def _legacy_chat_header_v24(self) -> str:
        return (f"# AI Agent\n\nBackend: `{self.args.server}`  \n"
                f"Model: `{self.model}`\n\n")

    def _load_context_dialog_v24(self, _button: Any) -> None:
        if self.busy or self.connection_pending:
            return
        Gtk = self._gtk
        dialog = Gtk.FileChooserDialog(
            title="Open saved chat", parent=self, action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                           "_Open", Gtk.ResponseType.ACCEPT)
        if self.chat_store.directory.is_dir():
            dialog.set_current_folder(str(self.chat_store.directory))
        self.add_context_file_filters(dialog)
        try:
            if dialog.run() == Gtk.ResponseType.ACCEPT and dialog.get_filename():
                self._load_chat_v24(Path(dialog.get_filename()))
        finally:
            dialog.destroy()

    def _save_context_dialog_v24(self, _button: Any) -> None:
        if self.busy:
            return
        Gtk = self._gtk
        dialog = Gtk.FileChooserDialog(
            title="Save chat JSON", parent=self, action=Gtk.FileChooserAction.SAVE)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                           "_Save", Gtk.ResponseType.ACCEPT)
        dialog.set_do_overwrite_confirmation(True)
        self.chat_store.directory.mkdir(parents=True, exist_ok=True)
        dialog.set_current_folder(str(self.chat_store.directory))
        dialog.set_current_name(
            self.current_chat_path.name if self.current_chat_path else "my-chat.json")
        self.add_context_file_filters(dialog)
        try:
            if dialog.run() == Gtk.ResponseType.ACCEPT and dialog.get_filename():
                target = Path(dialog.get_filename()).expanduser().resolve()
                if target == self.current_chat_path:
                    expected = self.current_chat_signature
                    expect_missing = expected is None
                else:
                    expected = None
                    expect_missing = False  # Explicit Save As has overwrite confirmation.
                self.chat_store.save(
                    target, list(self.messages), self.transcript_markdown,
                    expected_signature=expected, expect_missing=expect_missing)
                self.current_chat_path = target
                self.current_chat_signature = self.chat_store.signature(target)
                self._saved_chat_fingerprint = self._chat_fingerprint_v24()
                self.chat_title_label.set_text(target.name)
                self._refresh_chats_v24()
                self.append_markdown(f"## Chat saved\n\n`{target}`")
        except Exception as exc:
            self.append_markdown("## Chat save failed\n\n" + str(exc))
        finally:
            dialog.destroy()

    def _start_runtime_v24(self) -> None:
        if self.workspace is None:
            self.workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
        self._system_prompt_v24 = self.messages[0]["content"]
        self._connect_async_v24(self.args.server, self.args.url,
                                self.model, self.args.api_key, initial=True)

    def _sync_v24_controls(self) -> None:
        waiting = bool(self.connection_pending)
        self.connect_button.set_sensitive(not self.busy and not waiting)
        self.sidebar_toggle.set_sensitive(not self.busy)
        self.new_chat_button.set_sensitive(not self.busy and not waiting)
        self.refresh_chat_button.set_sensitive(not self.busy)
        if waiting:
            self.connect_button.set_label("Connecting…")
            self.send_button.set_sensitive(False)
            self.input_text_view.set_sensitive(False)
        else:
            self.connect_button.set_label("🔌 Connect")

    def _on_connect_clicked_v24(self, _button: Any) -> None:
        if self.busy or self.connection_pending:
            return
        Gtk = self._gtk
        dialog = Gtk.Dialog(title="Connect to LLM", transient_for=self, flags=Gtk.DialogFlags.MODAL)
        dialog.add_buttons("_Cancel", Gtk.ResponseType.CANCEL,
                           "Connect", Gtk.ResponseType.ACCEPT)
        dialog.set_default_response(Gtk.ResponseType.ACCEPT)
        grid = Gtk.Grid(row_spacing=9, column_spacing=9)
        grid.set_border_width(13)
        dialog.get_content_area().add(grid)
        server_choices = (
            ("llama.cpp", "llama.cpp"),
            ("openai-compatible", "OpenAI-compatible / remote API"),
            ("openai", "OpenAI SDK"),
        )
        chooser = Gtk.ComboBoxText()
        for value, label in server_choices:
            chooser.append(value, label)
        chooser.set_active_id(self.args.server if self.args.server != "chatgpt" else "openai")
        if chooser.get_active_id() is None:
            chooser.set_active_id("openai-compatible")
        grid.attach(Gtk.Label(label="Backend", xalign=0), 0, 0, 1, 1)
        grid.attach(chooser, 1, 0, 1, 1)
        url = Gtk.Entry()
        url.set_width_chars(52)
        url.set_text(self.args.url or "")
        grid.attach(Gtk.Label(label="URL", xalign=0), 0, 1, 1, 1)
        grid.attach(url, 1, 1, 1, 1)
        model = Gtk.Entry()
        model.set_text(self.model)
        grid.attach(Gtk.Label(label="Model", xalign=0), 0, 2, 1, 1)
        grid.attach(model, 1, 2, 1, 1)
        key = Gtk.Entry()
        key.set_visibility(False)
        key.set_placeholder_text("Leave blank to keep current key (if any)")
        grid.attach(Gtk.Label(label="API key", xalign=0), 0, 3, 1, 1)
        grid.attach(key, 1, 3, 1, 1)
        note = Gtk.Label(label="The API key stays in memory; chats never contain it.")
        note.set_xalign(0)
        grid.attach(note, 1, 4, 1, 1)
        dialog.show_all()
        try:
            if dialog.run() != Gtk.ResponseType.ACCEPT:
                return
            selected = chooser.get_active_id() or "openai-compatible"
            endpoint = url.get_text().strip()
            model_name = model.get_text().strip()
            entered_key = key.get_text().strip()
            api_key = entered_key or self.args.api_key
            parsed = urlsplit(endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                self.append_markdown("## Connection error\n\nEnter a valid HTTP(S) URL.")
                return
            if not model_name:
                self.append_markdown("## Connection error\n\nModel is required.")
                return
            if api_key and parsed.scheme != "https" and parsed.hostname not in (
                    "127.0.0.1", "localhost", "::1"):
                warning = Gtk.MessageDialog(
                    parent=self, flags=Gtk.DialogFlags.MODAL,
                    message_type=Gtk.MessageType.WARNING,
                    buttons=Gtk.ButtonsType.YES_NO,
                    text="Send API key over unencrypted HTTP?")
                warning.format_secondary_text(
                    "This remote connection is not encrypted; the API key could be intercepted.")
                try:
                    if warning.run() != Gtk.ResponseType.YES:
                        return
                finally:
                    warning.destroy()
            self._connect_async_v24(selected, endpoint, model_name, api_key)
        finally:
            # GTK Entry keeps secret only in volatile widget memory.
            key.set_text("")
            dialog.destroy()

    def _connect_async_v24(self, server: str, url: str, model: str,
                           api_key: Optional[str], initial: bool = False) -> None:
        if self.connection_pending or self.busy:
            return
        self.connection_pending = True
        self.connection_status.set_text("Connecting…")
        self._sync_v24_controls()

        def worker() -> None:
            candidate: Any = None
            new_mcp: Any = None
            error: Optional[str] = None
            try:
                from llm_backends import BackendConfig, make_backend
                candidate = make_backend(BackendConfig(
                    server=server, url=url, model=model, api_key=api_key,
                    prefer_json_schema=(not self.args.no_json_schema)))
                if not candidate.check_connection():
                    error = "Connection check failed. Inspect the terminal for HTTP details."
                elif self.mcp is None:
                    new_mcp = self._start_mcp_v24(self.workspace.name)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            def finish() -> bool:
                self.connection_pending = False
                if error is None:
                    prior = self.backend
                    self.backend = candidate
                    if new_mcp is not None:
                        self.mcp = new_mcp
                    self.args.server = server
                    self.args.url = url
                    self.args.model = model
                    self.args.api_key = api_key
                    self.model = model
                    self.connection_status.set_text("Connected")
                    self.connection_status.set_tooltip_text(f"{server}: {model} · {url}")
                    self.append_markdown(f"## LLM connected\n\nBackend: `{server}`  \nModel: `{model}`")
                    if prior is not None and prior is not candidate:
                        try:
                            session = getattr(prior, "session", None)
                            if session is not None:
                                session.close()
                        except Exception:
                            pass
                else:
                    self.connection_status.set_text("Disconnected" if self.backend is None else "Connected (previous)")
                    self.append_markdown("## LLM connection failed\n\n" + error)
                self.set_busy(False)
                self._sync_v24_controls()
                return False
            self._glib.idle_add(finish)

        threading.Thread(target=worker, daemon=True, name="agent-gui-connect").start()
