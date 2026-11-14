"""Open the viewer on a training's folder, a replay or one recording.

    python -m tribuna runs/first_course              # a training (live while it runs)
    python -m tribuna runs/realism                   # both stages of a training
    python -m tribuna runs/replays/first_course      # examples.record_agent
    python -m tribuna runs/x/laps/000000163840.json  # one recording
    python -m tribuna runs/first_course --browser    # in the web browser

The recordings are played in the training's order, from the first, each with
the laps of the ones before as fading trails; while a training goes on, new
ones are read every second and played when the player reaches them (End jumps
to the newest). Beside them, the training's log: how its episodes ended and
their lap time. The keys are in the window.

The viewer is a web page (tribuna/web, three.js) served from this computer
only (tribuna.server) and shown in a window of its own (pywebview, which uses
the system's web engine: WebView2 on Windows); with --browser, or when there
is no pywebview, in the web browser. Closing the window ends it; with the
browser, Ctrl+C here.
"""

import argparse
import sys
import time
import webbrowser
from pathlib import Path

from tribuna.server import start

TITLE = "Jineta · tribuna"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tribuna", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "path", help="a run folder, a replay folder or a .json recording"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="open it in the web browser, not a window",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="the server's port (default: any free one)"
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    if path.suffix == ".json" and not path.is_file():
        sys.exit(f"No such recording: {path}")
    if not path.exists():
        # A training that has not made its folder yet: the page waits for it.
        print(f"{path} does not exist yet: waiting for its first recording")
    server, url = start(path, args.port)
    print(f"Viewer at {url}", flush=True)
    try:
        if args.browser or not window(url):
            webbrowser.open(url)
            print("Opened in the web browser; Ctrl+C here to stop.", flush=True)
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


def window(url: str) -> bool:
    """Show the page in a desktop window until it is closed; False when that
    cannot be done here (no pywebview, or no web engine for it)."""
    try:
        import webview
    except ImportError:
        print("No pywebview (pip install pywebview): using the web browser.")
        return False
    try:
        webview.create_window(
            TITLE, url, width=1500, height=900, background_color="#0b0f15"
        )
        webview.start()
    except Exception as error:  # the backends raise their own kinds
        print(f"No window ({error}): using the web browser.")
        return False
    return True


if __name__ == "__main__":
    main()
