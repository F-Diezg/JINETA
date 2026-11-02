"""Scramble: the Iris waits on the ground and takes off when a threat comes near.

Same vehicle and guidance as the MVP (`iris_pursuit.py`), but a more realistic start.
The Iris sits on its launch pad at the origin, at rest, rotors stopped, altitude 0.
A threat flies a simple path towards the area. When it comes within ALERT_RANGE of
the pad (straight-line distance, as a radar measures it), the Iris launches and
flies pure pursuit at its commanded speed.

  A. A slow threat (8 m/s, gentle left turn, 110 m up): the Iris climbs, catches it.
  B. A fast threat (25 m/s straight, 120 m up): it is past before the Iris gets up
     to it; the Iris misses, keeps chasing and falls behind.

Each run ends at the capture or when the threat leaves the alert dome, whichever
comes first: past that point the pad no longer cares about it.

How the launch is simulated (`jineta.arena`), with no new physics:
  - The Iris and the threat fly in one world. The Iris is a waiting entity: until
    its launch its derivative is zero, so it stays exactly at rest on the pad, which
    is what a vehicle with its motors off on the ground does (there is no ground
    model).
  - A launch rule (`LaunchOnAlert`) checks the alert once per control step, before
    the controls: the Iris launches at the first t = k * DT at which the threat is
    inside the dome, and its pursuit control runs from that same step.
  - The pair (Iris -> threat) defends the dome, with an infinite margin so that a
    pass never ends it: it ends at the capture or when the threat leaves the dome.

Writes `docs/iris_scramble_viewer.html` (the same viewer as the MVP, which adds the
alert dome and a standby phase) and the start menu `docs/index.html`, shows range and
altitude against time and then opens the viewer app on its menu, in its own window.
`--no-window` only writes the pages. Its functions (`scramble`, `report`, ...) are
also used by `fixed_wing_scramble.py`, so the scenario logic is written once.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jineta.arena.engagement import ESCAPED, Engagement, Pairing, fight, pursuit
from jineta.arena.launch import Dome, LaunchOnAlert
from jineta.arena.world import TrackEntity, VehicleEntity, World
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import StraightLineTarget, TurningTarget
from viz.app import show, window_requested
from viz.menu import write_menu
from viz.viewer import scenario_data, write_viewer

DT = 0.01  # simulation and control step [s]
CAPTURE_RADIUS = 0.5  # [m]
ALERT_RANGE = 400.0  # distance from the pad that triggers the launch [m]
EVERY = 4  # viewer keeps one sample in 4 (25 Hz)
SITE = np.zeros(3)  # launch pad: the origin, on the ground (NED, altitude 0)
ON_THE_PAD = np.array([*SITE, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)  # at rest
DOME = Dome(SITE, ALERT_RANGE)  # the alert dome of the pad
ROOT = Path(__file__).resolve().parents[1]
VIEWER_PATH = ROOT / "docs" / "iris_scramble_viewer.html"

COLORS = {"A": "#2a78d6", "B": "#eb6834"}
GREY = "#888780"


def scramble(threat, speed, t_max):
    """Wait on the pad until the alert, then pursue until the capture or until the
    threat leaves the dome.

    `threat` is a scripted track or a simulated vehicle (a `VehicleEntity` named
    "threat"); either way it flies in the same world as the Iris. Returns the whole
    run on one time axis from t = 0 (an `Engagement`), the launch time, the time the
    run ended (the threat left the dome, or `t_max`) and the `Flight`.
    """
    if not isinstance(threat, VehicleEntity):
        threat = TrackEntity("threat", threat)
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    iris = VehicleEntity(
        "iris",
        autopilot.vehicle.derivative,
        ON_THE_PAD,
        control=pursuit(autopilot, "threat", speed),
        waiting=True,
    )
    world = World([iris, threat], rules=[LaunchOnAlert("iris", DOME, ["threat"])])
    target_speed = np.linalg.norm(
        world.snapshot(0.0, world.initial_state()).velocity("threat")
    )
    # margin = inf: never stop on a pass, only a capture or leaving the dome
    pair = Pairing("iris", "threat", CAPTURE_RADIUS, margin=math.inf, dome=DOME)
    battle = fight(world, [pair], DT, t_max)
    flight, result = battle.flight, battle.results[0]

    t_launch = flight.launched_at["iris"]
    if t_launch is None:
        raise ValueError(f"the threat never comes within {ALERT_RANGE} m")
    whole = Engagement(
        times=flight.times,
        states=flight.state("iris"),
        target_positions=flight.positions("threat"),
        closest=result.closest,
        captured=result.captured,
        pursuit_feasible=speed > target_speed,
    )
    t_exit = result.end_time if result.outcome == ESCAPED else t_max
    return whole, t_launch, t_exit, flight


def report(name, result, t_launch, t_exit, commanded_speed):
    """Print the verdict of one scramble."""
    flying = result.times >= t_launch
    altitude = -result.states[flying, 2]
    speed = np.linalg.norm(result.states[:, 3:6], axis=1)
    top = int(np.argmax(speed))
    climb, horizontal = -result.states[top, 5], np.linalg.norm(result.states[top, 3:5])
    verdict = "PASS" if result.captured else "FAIL"
    print(f"\n{name}")
    print(f"  alert: threat within {ALERT_RANGE:.0f} m at T+{t_launch:.2f} s, launch")
    print(
        f"  {verdict}: closest approach {result.closest.distance:.2f} m "
        f"at T+{result.closest.time:.2f} s "
        f"({result.closest.time - t_launch:.2f} s after the launch)"
    )
    print(
        f"  ordered {commanded_speed:.0f} m/s; fastest {speed[top]:.1f} m/s "
        f"({horizontal:.1f} horizontal, {climb:.1f} climbing)"
    )
    # + 0.0 turns -0.0 (the pad, z = 0) into 0.0 for printing
    print(f"  lowest altitude after the launch {altitude.min() + 0.0:.3f} m (pad: 0 m)")
    end = "capture" if result.captured else f"threat left the dome (T+{t_exit:.2f} s)"
    print(f"  run ended at T+{result.times[-1]:.2f} s: {end}")


def main() -> None:
    """Run both scrambles, print the verdicts, write the viewer, plot, open."""
    # --- The two scrambles ------------------------------------------------------------
    # A: slow threat from the North-West, heading South-South-East, gentle left turn.
    slow_target = TurningTarget(
        start=np.array([420.0, -250.0, -110.0]),
        speed=8.0,
        heading=np.radians(155.0),
        turn_rate=np.radians(-1.5),
    )
    slow, slow_launch, slow_exit, _ = scramble(slow_target, 16.0, t_max=120.0)

    # B: fast threat from the South-West, straight towards the North-East.
    fast_heading = np.radians(40.0)
    fast_target = StraightLineTarget(
        start=np.array([-600.0, -350.0, -120.0]),
        velocity=25.0 * np.array([np.cos(fast_heading), np.sin(fast_heading), 0.0]),
    )
    fast, fast_launch, fast_exit, _ = scramble(fast_target, 30.0, t_max=120.0)

    report("A. Slow threat: 8 m/s, turning", slow, slow_launch, slow_exit, 16.0)
    report("B. Fast threat: 25 m/s, straight", fast, fast_launch, fast_exit, 30.0)

    # --- Viewer -----------------------------------------------------------------------
    launch = {"range": ALERT_RANGE, "site": SITE}
    write_viewer(
        [
            scenario_data(
                slow,
                slow_target,
                EVERY,
                launch={**launch, "time": slow_launch},
                id="scramble-slow",
                name="A · slow threat",
                detail="8 m/s, turning 1.5 deg/s",
                target_speed=8.0,
                commanded_speed=16.0,
            ),
            scenario_data(
                fast,
                fast_target,
                EVERY,
                launch={**launch, "time": fast_launch},
                id="scramble-fast",
                name="B · fast threat",
                detail="25 m/s, straight",
                target_speed=25.0,
                commanded_speed=30.0,
            ),
        ],
        VIEWER_PATH,
        dt=DT * EVERY,
        capture_radius=CAPTURE_RADIUS,
        subtitle="generic target · scramble",
        menu="index.html",
    )
    menu = write_menu(VIEWER_PATH.parent)
    print(f"\nViewer written to {VIEWER_PATH}; menu at {menu}")

    # --- Plots: range and altitude against time ---------------------------------------
    fig, (ax_range, ax_alt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for key, result, t_launch, label in (
        ("A", slow, slow_launch, "A: slow threat, captured"),
        ("B", fast, fast_launch, "B: fast threat, missed"),
    ):
        color = COLORS[key]
        separation = np.linalg.norm(
            result.target_positions - result.states[:, :3], axis=1
        )
        ax_range.plot(result.times, separation, color=color, lw=2, label=label)
        ax_range.plot(result.closest.time, result.closest.distance, "o", color=color)
        ax_alt.plot(result.times, -result.states[:, 2], color=color, lw=2)
        ax_alt.plot(
            result.times, -result.target_positions[:, 2], color=color, lw=1, ls=":"
        )
        for ax in (ax_range, ax_alt):
            ax.axvline(t_launch, color=color, lw=1, ls="--")

    ax_range.axhline(ALERT_RANGE, color=GREY, lw=1, ls="--")
    ax_range.axhline(CAPTURE_RADIUS, color=GREY, lw=1, ls="--")
    ax_range.text(
        0.5, ALERT_RANGE * 1.1, f"alert range {ALERT_RANGE:.0f} m", color=GREY
    )
    ax_range.text(0.5, CAPTURE_RADIUS * 1.2, f"capture {CAPTURE_RADIUS} m", color=GREY)
    ax_range.set_yscale("log")
    ax_range.set_ylabel("Separation [m]")
    ax_range.set_title("Dashed vertical lines: launch. Dots: closest approach")
    ax_range.legend()
    ax_alt.set_ylabel("Altitude [m]")
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_title("Solid: Iris (from the pad at 0 m). Dotted: threat")
    for ax in (ax_range, ax_alt):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Scramble: launch on alert, then pure pursuit")
    fig.tight_layout()
    if window_requested():
        plt.show()  # close the plot to open the viewer app on its menu
        show(menu)


if __name__ == "__main__":
    main()
