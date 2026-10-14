"""Export engagements to the interactive 3D viewer (`viz/viewer_template.html`).

The viewer only replays data. This module turns `Engagement` results into that data,
in the simulator's own conventions (NED positions and velocities, body -> NED
quaternions [w, x, y, z], SI units), and writes it into the template. The result is
one self-contained HTML file: a page opened from disk (file://) is not allowed to
fetch a separate data file, so the data has to travel inside the page.
"""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jineta.engagement import Engagement
from jineta.jinete.targets import TargetTrack

TEMPLATE = Path(__file__).resolve().parent / "viewer_template.html"
PLACEHOLDER = '"__ENGAGEMENT_DATA__"'

# Minimal page around the template, so the file also opens straight from disk.
PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
</head>
<body>
{body}
</body>
</html>
"""


def rounded(array: npt.ArrayLike, decimals: int) -> list:
    """Nested lists with fewer digits: the file stays small, the eye sees no change."""
    return np.round(np.asarray(array, dtype=float), decimals).tolist()


def scenario_data(
    result: Engagement,
    target: TargetTrack,
    every: int = 4,
    launch: dict | None = None,
    **info,
) -> dict:
    """The data of one engagement that the viewer needs, downsampled.

    Args:
        result: the engagement, on one time axis from t = 0.
        target: the target track, evaluated again for its velocity.
        every: keep one sample in `every` (the viewer interpolates between them).
        launch: optional launch on alert, `{"time": [s], "range": [m],
            "site": NED [m]}`. The viewer then shows the alert range and a
            standby phase before the launch.
        **info: labels shown by the viewer (`id`, `name`, `detail`,
            `target_speed`, `commanded_speed`).
    """
    n = len(result.times)
    keep = np.arange(0, n, every)
    if keep[-1] != n - 1:
        keep = np.append(keep, n - 1)  # always keep the last sample (end of the run)

    times = result.times[keep]
    states = result.states[keep]
    target_velocities = np.array([target(t)[1] for t in times])

    # The closest approach happens between samples: interpolate both positions there.
    t_ca = result.closest.time

    def at_closest(track):
        return [float(np.interp(t_ca, result.times, track[:, i])) for i in range(3)]

    data = {
        **info,
        "captured": bool(result.captured),
        "pursuit_feasible": bool(result.pursuit_feasible),
        "closest": {
            "time": round(t_ca, 4),
            "distance": round(result.closest.distance, 4),
            "interceptor": rounded(at_closest(result.states[:, :3]), 3),
            "target": rounded(at_closest(result.target_positions), 3),
        },
        "t": rounded(times, 3),
        "interceptor": {
            "pos": rounded(states[:, :3], 3),  # NED [m]
            "vel": rounded(states[:, 3:6], 3),  # NED [m/s]
            "quat": rounded(states[:, 6:10], 5),  # body -> NED, [w, x, y, z]
        },
        "target": {
            "pos": rounded(result.target_positions[keep], 3),
            "vel": rounded(target_velocities, 3),
        },
    }
    if launch is not None:
        data["launch"] = {
            "time": round(float(launch["time"]), 4),
            "range": float(launch["range"]),
            "site": rounded(launch["site"], 3),
        }
    return data


def write_viewer(
    scenarios: list[dict], path: Path, dt: float, capture_radius: float
) -> Path:
    """Write the viewer page with these scenarios (from `scenario_data`) to `path`.

    Args:
        scenarios: one dict per engagement; the viewer shows one tab for each.
        path: the HTML file to write; its folder is created if needed.
        dt: time between the samples kept [s] (the viewer's frame-by-frame step).
        capture_radius: drawn on the range plot [m].
    """
    data = {
        "meta": {
            "frame": "NED",
            "quaternion": "body -> NED, [w, x, y, z]",
            "dt": dt,
            "capture_radius": capture_radius,
            "interceptor": "Iris quadrotor (PX4 SITL profile)",
            "guidance": "pure pursuit",
        },
        "scenarios": scenarios,
    }
    # "</" inside a <script> block would end it early, so it is escaped in the JSON.
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    body = TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PAGE.format(body=body), encoding="utf-8")
    return path
