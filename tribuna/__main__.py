"""Open the viewer on a training's folder, a replay or one recording.

    python -m tribuna runs/first_course              # a training (live while it runs)
    python -m tribuna runs/realism                   # both stages of a training
    python -m tribuna runs/replays/first_course      # examples.record_agent
    python -m tribuna runs/x/laps/000000163840.json  # one recording

The recordings are played in order of training steps, from the first; while a
training goes on, new ones are read every second and played when the player
reaches them (End jumps to the newest). The keys are in the window.
"""

import argparse
import sys
import time
from pathlib import Path

from jineta.picadero.recording import read_recording
from tribuna.player import Player
from tribuna.run import Run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tribuna", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "path", help="a run folder, a replay folder or a .json recording"
    )
    parser.add_argument(
        "--newest", action="store_true", help="start at the newest recording"
    )
    args = parser.parse_args(argv)

    try:
        import pyvista  # noqa: F401

        from tribuna.scene import Viewer
    except ImportError:
        sys.exit("The viewer needs PyVista: pip install pyvista")

    path = Path(args.path)
    if path.is_file():
        player, source = Player([read_recording(path)]), None
    else:
        run = Run(path)
        recordings = run.poll()
        while not recordings:  # a training that has not evaluated yet
            print(f"Waiting for the first recording in {path} ...", flush=True)
            time.sleep(2.0)
            recordings = run.poll()
        player, source = Player(recordings), run.poll
        for error in run.errors.values():
            print(f"Skipped: {error}")
    if args.newest:
        player.newest()
    Viewer(player, player.recordings[0].shape, source).show()


if __name__ == "__main__":
    main()
