"""End-to-end MVP: one Iris interceptor against one scripted target (pure pursuit).

Every layer of the simulator works together here for the first time: the 6-DOF rigid
body of `core`, the Iris multirotor and its cascaded autopilot in `jinete`, the pure
pursuit guidance law, and the engagement loop that interpolates the closest approach
between samples.

Pure pursuit means "always fly straight at where the target is right now". It is the
simplest guidance law there is, and it has a known weakness: whatever the starting
geometry, the pursuer ends up behind the target, so it only wins when it is clearly
faster. Two engagements show both sides of that:

  1. A slow threat (8 m/s, gently turning) against an interceptor commanded at
     16 m/s: it is caught.
  2. A fast threat (25 m/s straight) against an interceptor commanded at 30 m/s,
     more than the Iris can actually fly: the interceptor nearly meets it head-on,
     misses, and then falls behind for good.

The second case is the honest part of the MVP: `pursuit_feasible` only compares the
COMMANDED speed with the target speed, so it says "feasible" even here. What the
vehicle can really fly is a property of the airframe, not of the guidance law.

The two engagements are the files of `scenarios/pursuit/`. This script flies them,
prints the verdicts and saves the figure of the README, `docs/iris_pursuit.png`;
`iris_pursuit_viewer.py` writes the same two into the 3D viewer.
"""

import matplotlib.pyplot as plt
import numpy as np

from examples.run_scenarios import DOCS, SCENARIOS, play_folder, report
from jineta.arena.engagement import Engagement
from viz.app import window_requested

FIGURE_PATH = DOCS / "iris_pursuit.png"
INTERCEPTOR_COLOR = "#2a78d6"
TARGET_COLOR = "#eb6834"
GREY = "#888780"


def separations(result: Engagement):
    """Distance between interceptor and target at each sample [m]."""
    return np.linalg.norm(result.target_positions - result.states[:, :3], axis=1)


def main() -> None:
    """Fly both engagements, print the verdicts, save and show the figure."""
    _, (slow_case, fast_case) = play_folder(SCENARIOS / "pursuit")
    for case in (slow_case, fast_case):
        report(case)
    slow, fast = slow_case.engagement(), fast_case.engagement()
    slow_speed, fast_speed = slow_case.threat.speed, fast_case.threat.speed
    capture_radius = slow_case.interceptor.capture_radius

    # --- Plots -------------------------------------------------------------------
    # East on the horizontal axis and North on the vertical one, so the ground track
    # reads like a map; the third axis is altitude = -z.
    fig = plt.figure(figsize=(12.5, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    interceptor = slow.states[:, :3]
    ax3d.plot(
        interceptor[:, 1],
        interceptor[:, 0],
        -interceptor[:, 2],
        color=INTERCEPTOR_COLOR,
        lw=2,
        label="Interceptor (Iris)",
    )
    ax3d.plot(
        slow.target_positions[:, 1],
        slow.target_positions[:, 0],
        -slow.target_positions[:, 2],
        color=TARGET_COLOR,
        lw=2,
        label=f"Target ({slow_speed:.0f} m/s, turning)",
    )
    for track, color in (
        (interceptor, INTERCEPTOR_COLOR),
        (slow.target_positions, TARGET_COLOR),
    ):
        ax3d.scatter(
            track[0, 1], track[0, 0], -track[0, 2], color=color, s=35, marker="o"
        )
    hit = interceptor[slow.closest.step]
    ax3d.scatter(
        hit[1],
        hit[0],
        -hit[2],
        color="k",
        s=70,
        marker="x",
        zorder=5,
        label=f"Closest approach ({slow.closest.distance:.2f} m)",
    )
    ax3d.set_xlabel("East [m]")
    ax3d.set_ylabel("North [m]")
    ax3d.set_zlabel("Altitude [m]")
    ax3d.set_title("Intercept: circles are the starting points")
    ax3d.legend(loc="upper left", fontsize="small")

    axd = fig.add_subplot(1, 2, 2)
    for result, color, label in (
        (slow, INTERCEPTOR_COLOR, f"vs {slow_speed:.0f} m/s: captured"),
        (fast, TARGET_COLOR, f"vs {fast_speed:.0f} m/s: missed"),
    ):
        axd.plot(result.times, separations(result), color=color, lw=2, label=label)
        axd.plot(
            result.closest.time, result.closest.distance, color=color, marker="o", ms=6
        )
    axd.axhline(capture_radius, color=GREY, lw=1, ls="--")
    axd.text(
        0.5,
        capture_radius * 1.15,
        f"capture radius {capture_radius} m",
        color=GREY,
        fontsize="small",
    )
    axd.set_yscale("log")  # the separation spans from hundreds of metres down to 0.5 m
    axd.set_xlabel("Time [s]")
    axd.set_ylabel("Separation [m]")
    axd.set_title("Separation vs time (dots: closest approach)")
    axd.grid(True, which="both", alpha=0.3)
    axd.legend()

    fig.suptitle("Jineta MVP: Iris interceptor under pure pursuit")
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150)
    print(f"\nFigure saved to {FIGURE_PATH}")
    if window_requested():  # --no-window: only save the figure
        plt.show()


if __name__ == "__main__":
    main()
