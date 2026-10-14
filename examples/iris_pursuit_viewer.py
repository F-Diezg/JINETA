"""Interactive 3D viewer of the MVP engagements, as a single HTML file.

Runs the same two engagements as `iris_pursuit.py`, writes them into the viewer
(`viz/viewer.py`), producing `docs/iris_pursuit_viewer.html`, and opens it in its own
window (`viz/app.py`; the browser if pywebview is not installed). Drag to orbit,
scroll to zoom, right-drag to pan, and use the timeline to play, pause and scrub.

    python -m examples.iris_pursuit_viewer               # simulate, write, open
    python -m examples.iris_pursuit_viewer --no-window   # only write the page

The simulator still never draws: this script runs it and exports the results.
"""

from pathlib import Path

import numpy as np

from jineta.engagement import engage
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import StraightLineTarget, TurningTarget
from viz.app import show, window_requested
from viz.viewer import scenario_data, write_viewer

DT = 0.01  # simulation and control step [s]
CAPTURE_RADIUS = 0.5  # [m]
EVERY = 4  # keep one sample in 4: 25 Hz is plenty for the eye, the viewer interpolates
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "iris_pursuit_viewer.html"


def hover_at(position):
    """Iris state hovering level and at rest at a NED position (13 elements)."""
    return np.array([*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


def fly(target, start, speed, t_end, margin=10.0):
    """One engagement: a fresh Iris with its autopilot flies against `target`."""
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    return engage(
        autopilot,
        target,
        hover_at(start),
        speed,
        t_end,
        capture_radius=CAPTURE_RADIUS,
        margin=margin,
    )


# --- The two engagements (same as iris_pursuit.py) -----------------------------------
heading = np.radians(45.0)

slow_target = TurningTarget(
    start=np.array([150.0, 60.0, -120.0]),
    speed=8.0,
    heading=heading,
    turn_rate=np.radians(2.0),
)
slow = fly(slow_target, [0.0, 0.0, -100.0], 16.0, t_end=120.0)

fast_target = StraightLineTarget(
    start=np.array([-300.0, -250.0, -120.0]),
    velocity=25.0 * np.array([np.cos(heading), np.sin(heading), 0.0]),
)
fast = fly(fast_target, [0.0, 0.0, -100.0], 30.0, t_end=60.0, margin=50.0)

scenarios = [
    scenario_data(
        slow,
        slow_target,
        EVERY,
        id="slow",
        name="Slow threat",
        detail="8 m/s, turning 2 deg/s",
        target_speed=8.0,
        commanded_speed=16.0,
    ),
    scenario_data(
        fast,
        fast_target,
        EVERY,
        id="fast",
        name="Fast threat",
        detail="25 m/s, straight",
        target_speed=25.0,
        commanded_speed=30.0,
    ),
]
write_viewer(scenarios, OUTPUT, dt=DT * EVERY, capture_radius=CAPTURE_RADIUS)

for s in scenarios:
    verdict = "captured" if s["captured"] else "missed"
    print(
        f"{s['name']}: {verdict}, closest {s['closest']['distance']:.2f} m "
        f"at t = {s['closest']['time']:.2f} s, {len(s['t'])} frames"
    )
print(f"Viewer written to {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")

if window_requested():
    show(OUTPUT)
