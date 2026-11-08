#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent_chat_store import ChatStore, CHAT_FORMAT, chat_title
from llm_backends import BackendConfig, OpenAICompatibleBackend, make_backend


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = ChatStore(Path(self.temp.name) / "save")
        self.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Create an RS485 bridge"},
            {"role": "assistant", "content": json.dumps({"type":"message","message":"Done"})},
        ]

    def test_round_trip_and_no_secret(self):
        path = self.store.new_path()
        self.store.save(path, self.messages, "## You\n\nCreate bridge\n", {
            "url": "https://user:pass@example.com/v1/chat/completions?api_key=secret",
            "server": "openai-compatible", "model": "gpt-6.1-sol",
            "api_key": "sk-proj-no-saving", "token": "not-saving",
        })
        raw = path.read_text(encoding="utf-8")
        self.assertNotIn("sk-proj-no-saving", raw)
        self.assertNotIn("secret", raw)
        self.assertNotIn("pass", raw)
        loaded = self.store.load(path)
        self.assertEqual(loaded["source_format"], CHAT_FORMAT)
        self.assertEqual(loaded["messages"], self.messages)
        self.assertIn("Create bridge", loaded["transcript"])
        self.assertEqual(loaded["profile"]["model"], "gpt-6.1-sol")

    def test_legacy_list_and_old_agent_context(self):
        legacy = self.store.directory / "old_chat.json"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text(json.dumps(self.messages))
        loaded = self.store.load(legacy)
        self.assertEqual(loaded["source_format"], "legacy-list")
        self.assertIn("## You", loaded["transcript"])
        old = self.store.directory / "ctx.json"
        old.write_text(json.dumps({"format": "agent-context", "version": 2,
                                   "messages": self.messages,
                                   "llama_cache": {"slot_id": 0}}))
        self.assertEqual(self.store.load(old)["messages"], self.messages)
        self.assertEqual(len(self.store.list_chats()), 2)

    def test_unrelated_json_is_skipped_not_deleted(self):
        self.store.directory.mkdir(parents=True)
        junk = self.store.directory / "unrelated.json"
        junk.write_text('{"unrelated": true}')
        self.assertEqual(self.store.list_chats(), [])
        self.assertTrue(junk.exists())

    def test_new_unique_paths(self):
        self.assertNotEqual(self.store.new_path(), self.store.new_path())


class RemoteBackendTests(unittest.TestCase):
    def setUp(self):
        self.cfg = BackendConfig(server="openai-compatible", model="gpt-6.1-sol",
                                 url="https://api.openai.com/v1/chat/completions",
                                 api_key="sk-testing")

    def test_remote_uses_bearer_and_no_llama_slot(self):
        backend = make_backend(self.cfg)
        self.assertIsInstance(backend, OpenAICompatibleBackend)
        self.assertEqual(backend.session.headers["Authorization"], "Bearer sk-testing")
        result = MagicMock()
        result.ok = True
        result.json.return_value = {"choices": [{"message": {"content": '{"type":"message"}'}}]}
        with patch.object(backend.session, "post", return_value=result) as request:
            answer = backend.chat_completion([{"role": "user", "content": "hello"}], 256, 0.7)
        self.assertIn("message", answer["content"])
        payload = request.call_args.kwargs["json"]
        self.assertNotIn("cache_prompt", payload)
        self.assertNotIn("id_slot", payload)
        self.assertNotIn("temperature", payload)

    def test_default_llama_backend_auto_detects_openai(self):
        cfg = BackendConfig(server="llama.cpp", model="gpt-6.1-sol",
                            url=self.cfg.url, api_key="sk-testing")
        self.assertIsInstance(make_backend(cfg), OpenAICompatibleBackend)

    def test_connection_check_url(self):
        backend = make_backend(self.cfg)
        response = MagicMock(ok=True)
        with patch.object(backend.session, "get", return_value=response) as get:
            self.assertTrue(backend.check_connection())
        self.assertEqual(get.call_args.args[0], "https://api.openai.com/v1/models")

    def test_auth_failure_no_retries(self):
        backend = make_backend(self.cfg)
        response = MagicMock(ok=False, status_code=401, text="unauthorized")
        with patch.object(backend.session, "post", return_value=response) as post:
            with patch("builtins.print"):
                self.assertIsNone(backend.chat_completion([], 128, 0.7))
        self.assertEqual(post.call_count, 1)


SYNTHETIC_GUI = '''
from typing import Any, Dict, List, Optional, Tuple
GTKUI_VERSION = "R260820_GUI_ONLY_12"
class Gtk:
    class Window: pass
    class TextBox: pass
class Chat:
    class Foo: pass

def dummy():
    class AgentGuiWindow(Gtk.Window):
        def __init__(self):
            self.add(outer)
            self.append_markdown("Ready")
            self.start_runtime()

        def start_runtime(self) -> None:
            print("old")

        def on_destroy(self, *_args):
            pass

        def cleanup(self) -> None:
            try:
                pass
            except Exception:
                pass

        def set_busy(self, busy: bool) -> None:
            self.handoff_button.set_sensitive((not busy) and can_run)

        def append_markdown(self, text):
            pass

        def on_send_clicked(self, _button: Any) -> None:
            if self.busy:
                return

        def finish_worker_turn(self) -> None:
            self.worker_thread = None
            self.worker_ident = None
            self.stop_requested = False
            self.cancel_event = threading.Event()
            self.set_busy(False)

        def handle_input_in_worker(self, user_input, auto_confirm):
            if cmd == "clear":
                self.messages = [self.messages[0]]
                self.workspace = tempfile.TemporaryDirectory(prefix="ai-bash-agent-")
                GLib.idle_add(lambda: (self.reset_transcript(), False)[1])

            if cmd == "save":
                with open("ctx.json", "wt", encoding="utf-8") as fd:
                    json.dump(self.messages, fd)
                return "## Context saved\\n\\nSaved to `ctx.json`."

            if cmd == "load":
                with open("ctx.json", "rt", encoding="utf-8") as fd:
                    self.messages = json.load(fd)
                user_input = "Remind me, please, what has been done?"

        def load_context_dialog(self, _button: Any) -> None:
            print("old load")

        def save_context_dialog(self, _button: Any) -> None:
            print("old save")

        def open_markdown(self, _button: Any):
            pass
'''

if __name__ == "__main__":
    unittest.main(verbosity=2)
