#!/usr/bin/env python3

from pathlib import Path

import gi
gi.require_version("Gtk", "3.0")
gi.require_version("GtkSource", "4")
from gi.repository import Gtk, Pango, GtkSource, Gdk

import mistune


class MarkdownViewer(Gtk.Window):
    def __init__(self):
        super().__init__(title="Lightweight Markdown Viewer")

        self.set_default_size(800, 600)
        self.connect("destroy", Gtk.main_quit)

        self.text_view = Gtk.TextView()
        self.text_view.set_editable(False)
        self.text_view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self.text_view.set_left_margin(10)
        self.text_view.set_right_margin(10)
        self.text_view.set_top_margin(8)
        self.text_view.set_bottom_margin(8)

        self.buffer = self.text_view.get_buffer()
        self.create_tags()

        scroll = Gtk.ScrolledWindow()
        scroll.add(self.text_view)

        open_button = Gtk.Button(label="Open .md File")
        open_button.connect("clicked", self.open_markdown)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.pack_start(scroll, True, True, 0)
        box.pack_start(open_button, False, False, 0)


        self.add(box)

        self.md_parser = mistune.create_markdown(renderer="ast", plugins=["task_lists", "strikethrough", "table"])

    def insert_code_block(self, code: str, lang: str | None = None):
        it = self.buffer.get_end_iter()
        anchor = self.buffer.create_child_anchor(it)

        source_buffer = GtkSource.Buffer()
        source_buffer.set_text(code)

        lang_manager = GtkSource.LanguageManager.get_default()

        lang_aliases = {
            "c": "c",
            "h": "c",
            "cpp": "cpp",
            "c++": "cpp",
            "cc": "cpp",
            "hpp": "cpp",
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
        }

        language = None

        if lang:
            lang = lang.strip().lower().split()[0]
            lang_id = lang_aliases.get(lang, lang)
            language = lang_manager.get_language(lang_id)

        if language:
            source_buffer.set_language(language)
        else:
            print(f"Unknown language for code block: {lang!r}")
            source_buffer.set_language("sh")

        source_buffer.set_highlight_syntax(True)

        style_manager = GtkSource.StyleSchemeManager.get_default()

        # Try different schemes:
        # classic, cobalt, tango, solarized-light, solarized-dark, oblivion
        scheme = style_manager.get_scheme("oblivion")
        if scheme:
            source_buffer.set_style_scheme(scheme)

        source_view = GtkSource.View.new_with_buffer(source_buffer)
        source_view.set_editable(False)
        source_view.set_cursor_visible(False)
        source_view.set_monospace(True)
        source_view.set_show_line_numbers(False)

        source_view.set_left_margin(10)
        source_view.set_right_margin(10)
        source_view.set_top_margin(8)
        source_view.set_bottom_margin(8)

        source_view.set_size_request(-1, 140)

        source_view.override_background_color(
            Gtk.StateFlags.NORMAL,
            Gdk.RGBA(0.2, 0.2, 0.2, 1.0),
        )

        frame = Gtk.Frame()
        frame.set_shadow_type(Gtk.ShadowType.IN)
        frame.add(source_view)
        frame.show_all()

        self.text_view.add_child_at_anchor(frame, anchor)
        self.insert("\n\n")

    def insert_checkbox(self, checked: bool):
        it = self.buffer.get_end_iter()

        anchor = self.buffer.create_child_anchor(it)

        checkbox = Gtk.CheckButton()
        checkbox.set_active(checked)
        checkbox.set_sensitive(False)
        checkbox.show()

        self.text_view.add_child_at_anchor(checkbox, anchor)

    def create_tags(self):
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

        self.buffer.create_tag(
            "bold",
            weight=Pango.Weight.BOLD,
        )

        self.buffer.create_tag(
            "italic",
            style=Pango.Style.ITALIC,
        )

        self.buffer.create_tag(
            "code",
            family="monospace",
            background="#eeeeee",
        )

        self.buffer.create_tag(
            "code_block",
            family="monospace",
            background="#eeeeee",
            left_margin=16,
            right_margin=16,
            pixels_above_lines=8,
            pixels_below_lines=8,
        )

        self.buffer.create_tag(
            "quote",
            left_margin=24,
            style=Pango.Style.ITALIC,
        )

        self.buffer.create_tag(
            "link",
            underline=Pango.Underline.SINGLE,
            foreground="blue",
        )

    def insert(self, text, *tags):
        it = self.buffer.get_end_iter()

        if tags:
            self.buffer.insert_with_tags_by_name(it, text, *tags)
        else:
            self.buffer.insert(it, text)

    def render_inline(self, children, inherited_tags=()):
        if not children:
            return

        for node in children:
            node_type = node.get("type")

            if node_type == "text":
                self.insert(node.get("raw", ""), *inherited_tags)

            elif node_type == "strong":
                self.render_inline(
                    node.get("children", []),
                    inherited_tags + ("bold",),
                )

            elif node_type == "emphasis":
                self.render_inline(
                    node.get("children", []),
                    inherited_tags + ("italic",),
                )

            elif node_type == "codespan":
                self.insert(node.get("raw", ""), *(inherited_tags + ("code",)))

            elif node_type == "link":
                url = node.get("attrs", {}).get("url", "")
                self.render_inline(
                    node.get("children", []),
                    inherited_tags + ("link",),
                )
                if url:
                    self.insert(f" ({url})", *(inherited_tags + ("link",)))

            elif "children" in node:
                self.render_inline(node["children"], inherited_tags)

    def render_block(self, node):
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

            print("CODE LANG:", repr(lang))

            if lang:
                lang = lang.strip().split()[0]

            self.insert_code_block(code, lang)
            return

        if node_type == "block_quote":
            for child in node.get("children", []):
                before = self.buffer.get_end_iter()
                self.render_block(child)
                after = self.buffer.get_end_iter()
                self.buffer.apply_tag_by_name("quote", before, after)
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

        # Fallback for unknown nodes
        if "children" in node:
            for child in node["children"]:
                self.render_block(child)

    def render_markdown(self, markdown_text):
        self.buffer.set_text("")

        tokens = self.md_parser(markdown_text)

        for node in tokens:
            self.render_block(node)

    def open_markdown(self, button):
        dialog = Gtk.FileChooserDialog(
            title="Open Markdown File",
            parent=self,
            action=Gtk.FileChooserAction.OPEN,
        )

        dialog.add_buttons(
            "_Cancel", Gtk.ResponseType.CANCEL,
            "_Open", Gtk.ResponseType.ACCEPT,
        )

        md_filter = Gtk.FileFilter()
        md_filter.set_name("Markdown files")
        md_filter.add_pattern("*.md")
        md_filter.add_pattern("*.markdown")
        md_filter.add_pattern("*.txt")
        dialog.add_filter(md_filter)

        try:
            if dialog.run() != Gtk.ResponseType.ACCEPT:
                return

            filename = dialog.get_filename()
            text = Path(filename).read_text(encoding="utf-8")

            self.render_markdown(text)

        finally:
            dialog.destroy()


def main():
    win = MarkdownViewer()
    win.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()