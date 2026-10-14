"""Tests for the desktop window of the viewer (viz/app.py).

pywebview itself is not exercised here: a stand-in module records what `show` asks
of it. What is tested is our side: the right page and title go to the window, the
browser is used when pywebview is missing, and a missing page fails loudly.
"""

import sys
import tempfile
import types
import webbrowser
from contextlib import contextmanager
from pathlib import Path

import pytest

from viz.app import show, window_requested


@contextmanager
def viewer_page():
    """A temporary HTML file standing for a viewer page."""
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "viewer.html"
        page.write_text("<title>test</title>", encoding="utf-8")
        yield page


@contextmanager
def module_as(name, module):
    """Make `import name` return `module` (None makes the import fail)."""
    saved = sys.modules.get(name, "missing")
    sys.modules[name] = module
    try:
        yield
    finally:
        if saved == "missing":
            del sys.modules[name]
        else:
            sys.modules[name] = saved


def fake_pywebview():
    """A stand-in for pywebview that records the calls."""
    calls = []
    fake = types.ModuleType("webview")
    fake.create_window = lambda title, **kwargs: calls.append(("window", title, kwargs))
    fake.start = lambda: calls.append(("start",))
    return fake, calls


def test_opens_a_window_with_the_page_and_starts_it():
    fake, calls = fake_pywebview()
    with viewer_page() as page, module_as("webview", fake):
        show(page, title="Jineta test")
    (kind, title, kwargs), start = calls
    assert kind == "window"
    assert title == "Jineta test"
    assert kwargs["url"] == page.resolve().as_uri()  # file:///... of that page
    assert start == ("start",)


def test_falls_back_to_the_browser_without_pywebview():
    opened = []
    original = webbrowser.open
    webbrowser.open = opened.append
    try:
        with viewer_page() as page, module_as("webview", None):
            show(page)
    finally:
        webbrowser.open = original
    assert opened == [page.resolve().as_uri()]


def test_a_missing_page_is_an_error():
    with pytest.raises(FileNotFoundError):
        show(Path("does/not/exist.html"))


def test_no_window_flag():
    assert window_requested([])
    assert window_requested(["--other"])
    assert not window_requested(["--no-window"])
