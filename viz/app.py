"""Show a viewer page in its own desktop window, not in a browser tab.

The viewer is a web page (HTML + three.js), like the screens of many desktop
applications. What makes it look like an application is the window: pywebview opens
a native window (title bar, no address bar or tabs) and draws the page inside it with
the system's web engine (on Windows, the Edge engine that ships with Windows 10/11).

pywebview is optional. Without it, the page opens in the default browser as before.

Open an existing page without running the simulation again:

    python -m viz.app docs/iris_scramble_viewer.html
"""

import argparse
import sys
import webbrowser
from pathlib import Path

TITLE = "Jineta · Engagement Viewer"
BACKGROUND = "#04090f"  # the viewer's own background, shown while the page loads


def show(path: Path, title: str = TITLE, size: tuple[int, int] = (1400, 860)) -> None:
    """Open a viewer page in a desktop window (or in the browser, as a fallback).

    Blocks until the window is closed, like `plt.show()`.

    Args:
        path: the HTML file written by `viz.viewer.write_viewer`.
        title: window title.
        size: initial window size in pixels (width, height).
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"no viewer page at {path}")

    try:
        import webview  # pywebview
    except ImportError:
        print("pywebview is not installed (pip install pywebview): opening the browser")
        webbrowser.open(path.as_uri())
        return

    webview.create_window(
        title,
        url=path.as_uri(),
        width=size[0],
        height=size[1],
        min_size=(900, 600),
        background_color=BACKGROUND,
    )
    webview.start()


def window_requested(argv: list[str] | None = None) -> bool:
    """False when the command line asks for no window (`--no-window`): the examples
    then only write their files, e.g. to regenerate `docs/` before a commit."""
    return "--no-window" not in (sys.argv[1:] if argv is None else argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Open a Jineta viewer page.")
    parser.add_argument("page", type=Path, help="viewer HTML file, e.g. docs/...html")
    show(parser.parse_args(argv).page)


if __name__ == "__main__":
    main()
