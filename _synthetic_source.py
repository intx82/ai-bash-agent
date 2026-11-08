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
