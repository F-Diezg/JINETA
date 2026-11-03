"""Interactive 3D viewer of the MVP engagements, as a single HTML file.

Flies the two cases of `scenarios/pursuit/` (the same ones as `iris_pursuit.py`),
writes them into the viewer, producing `docs/iris_pursuit_viewer.html`, and opens it
in its own window (`viz/app.py`; the browser if pywebview is not installed). Drag to
orbit, scroll to zoom, right-drag to pan, and use the timeline to play, pause and
scrub.

    python -m examples.iris_pursuit_viewer               # simulate, write, open
    python -m examples.iris_pursuit_viewer --no-window   # only write the page

The simulator still never draws: this script runs it and exports the results.
"""

from examples.run_scenarios import DOCS, SCENARIOS, play_folder, report
from viz.app import show, window_requested
from viz.pages import write_page


def main() -> None:
    page, cases = play_folder(SCENARIOS / "pursuit")
    for case in cases:
        report(case)
    path = write_page(page, cases, DOCS)
    print(f"\nViewer written to {path} ({path.stat().st_size / 1024:.0f} KB)")
    if window_requested():
        show(path)


if __name__ == "__main__":
    main()
