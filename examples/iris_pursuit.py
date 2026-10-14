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
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jineta.engagement import Engagement, engage
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import StraightLineTarget, TurningTarget
from viz.app import window_requested

DT = 0.01  # simulation and control step [s], 100 Hz
CAPTURE_RADIUS = 0.5  # separation that counts as an interception [m]
FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "iris_pursuit.png"

INTERCEPTOR_COLOR = "#2a78d6"
TARGET_COLOR = "#eb6834"
GREY = "#888780"


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


def separations(result: Engagement):
    """Distance between interceptor and target at each sample [m]."""
    return np.linalg.norm(result.target_positions - result.states[:, :3], axis=1)


def report(name: str, result: Engagement, commanded_speed: float) -> None:
    """Print the verdict of one engagement."""
    verdict = "PASS" if result.captured else "FAIL"
    top_speed = np.linalg.norm(result.states[:, 3:6], axis=1).max()
    print(f"\n{name}")
    print(
        f"  {verdict}: closest approach {result.closest.distance:.2f} m "
        f"at t = {result.closest.time:.2f} s (capture radius {CAPTURE_RADIUS} m)"
    )
    print(
        f"  run ended at t = {result.times[-1]:.2f} s, "
        f"{len(result.times)} samples at dt = {DT} s"
    )
    print(
        f"  commanded pursuit speed {commanded_speed:.0f} m/s, "
        f"fastest the interceptor actually flew {top_speed:.1f} m/s"
    )
    print(
        f"  pursuit_feasible = {result.pursuit_feasible} "
        "(commanded speed vs target speed only)"
    )


# --- The two engagements -------------------------------------------------------------
# NED: x North, y East, z Down, so an altitude of 100 m is z = -100 m.
SLOW_SPEED, SLOW_COMMAND = 8.0, 16.0
slow_target = TurningTarget(
    start=np.array([150.0, 60.0, -120.0]),
    speed=SLOW_SPEED,
    heading=np.radians(45.0),  # North-East
    turn_rate=np.radians(2.0),  # gentle right turn
)
slow = fly(slow_target, [0.0, 0.0, -100.0], SLOW_COMMAND, t_end=120.0)

FAST_SPEED, FAST_COMMAND = 25.0, 30.0
fast_heading = np.radians(45.0)
fast_target = StraightLineTarget(
    start=np.array([-300.0, -250.0, -120.0]),
    velocity=FAST_SPEED * np.array([np.cos(fast_heading), np.sin(fast_heading), 0.0]),
)
# A larger margin than the default lets the run continue past the miss, so the plot
# shows the interceptor being left behind instead of stopping at the closest approach.
fast = fly(fast_target, [0.0, 0.0, -100.0], FAST_COMMAND, t_end=60.0, margin=50.0)

report(f"Slow threat: {SLOW_SPEED:.0f} m/s turning", slow, SLOW_COMMAND)
report(f"Fast threat: {FAST_SPEED:.0f} m/s straight", fast, FAST_COMMAND)

# --- Plots ---------------------------------------------------------------------------
# East on the horizontal axis and North on the vertical one, so the ground track reads
# like a map; the third axis is altitude = -z.
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
    label=f"Target ({SLOW_SPEED:.0f} m/s, turning)",
)
for track, color in (
    (interceptor, INTERCEPTOR_COLOR),
    (slow.target_positions, TARGET_COLOR),
):
    ax3d.scatter(track[0, 1], track[0, 0], -track[0, 2], color=color, s=35, marker="o")
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
    (slow, INTERCEPTOR_COLOR, f"vs {SLOW_SPEED:.0f} m/s: captured"),
    (fast, TARGET_COLOR, f"vs {FAST_SPEED:.0f} m/s: missed"),
):
    axd.plot(result.times, separations(result), color=color, lw=2, label=label)
    axd.plot(
        result.closest.time, result.closest.distance, color=color, marker="o", ms=6
    )
axd.axhline(CAPTURE_RADIUS, color=GREY, lw=1, ls="--")
axd.text(
    0.5,
    CAPTURE_RADIUS * 1.15,
    f"capture radius {CAPTURE_RADIUS} m",
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
