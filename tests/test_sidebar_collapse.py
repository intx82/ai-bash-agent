#!/usr/bin/env python3
"""Offline GTK-layout contract regression test (no PyGObject required)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_gui_support import ChatUIV24Mixin


class Widget:
    def __init__(self, **kwargs):
        self.children = []
        self.visible = True
        self.active = False
        self.orientation = kwargs.get('orientation')
        self.spacing = kwargs.get('spacing')
        self.resize_policy = None
        self.label = kwargs.get('label', '')
        self.signals = {}
        self.width_request = 0

    def __getattr__(self, name):
        if name.startswith('set_'):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def connect(self, name, callback):
        self.signals[name] = callback

    def set_active(self, active):
        self.active = active

    def get_active(self):
        return self.active

    def set_visible(self, visible):
        self.visible = visible

    def set_size_request(self, width, _height):
        self.width_request = width

    def pack_start(self, widget, expand, fill, padding):
        self.children.append((widget, expand, fill, padding))

    def add(self, widget):
        self.children.append((widget, True, True, 0))

    def get_children(self):
        return [x[0] for x in self.children]

    def remove(self, row):
        self.children = [x for x in self.children if x[0] != row]

    def show_all(self):
        self.visible = True


class FakeList(Widget):
    pass


class FakeGtk:
    Orientation = SimpleNamespace(HORIZONTAL='horizontal', VERTICAL='vertical')
    SelectionMode = SimpleNamespace(SINGLE='single')
    PolicyType = SimpleNamespace(NEVER='never', AUTOMATIC='automatic')
    Box = Widget
    Button = Widget
    Label = Widget
    ToggleButton = Widget
    ListBox = FakeList
    ScrolledWindow = Widget


class FakeAgent(ChatUIV24Mixin):
    def __init__(self):
        self._gtk = FakeGtk
        self._pango = SimpleNamespace(EllipsizeMode=SimpleNamespace(END='end'))
        self.owned = []

    def add(self, widget):
        self.owned.append(widget)


class SidebarLayoutTest(unittest.TestCase):
    def test_collapse_removes_sidebar_from_horizontal_box_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict('os.environ', {'AGENT_CHAT_DIR': tmp}):
                agent = FakeAgent()
                chat_content = Widget()
                toolbar = Widget()
                agent._install_chat_layout(chat_content, toolbar)

                container = agent.owned[0]
                self.assertEqual(container.orientation, FakeGtk.Orientation.HORIZONTAL)
                self.assertEqual(len(container.children), 2)
                sidebar, sidebar_expand, _, _ = container.children[0]
                main, main_expand, _, _ = container.children[1]
                self.assertIs(sidebar, agent.sidebar_panel)
                self.assertIs(main, chat_content)
                self.assertFalse(sidebar_expand)
                self.assertTrue(main_expand)
                self.assertEqual(sidebar.width_request, 235)
                self.assertTrue(sidebar.visible)

                agent.sidebar_toggle.set_active(False)
                agent._on_sidebar_toggle_v24(agent.sidebar_toggle)
                self.assertFalse(sidebar.visible)
                self.assertTrue(main.visible)
                self.assertEqual([w for w, *_ in container.children if w.visible], [main])

                agent.sidebar_toggle.set_active(True)
                agent._on_sidebar_toggle_v24(agent.sidebar_toggle)
                self.assertTrue(sidebar.visible)
                self.assertEqual([w for w, *_ in container.children if w.visible], [sidebar, main])


if __name__ == '__main__':
    unittest.main(verbosity=2)
