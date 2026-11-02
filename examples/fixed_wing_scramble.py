"""Scramble against a fixed-wing drone: the phase 7 scramble with a simulated threat.

The same launch on alert as `iris_scramble.py` (the Iris waits on its pad at the
origin, rotors stopped, launches when the threat comes within ALERT_RANGE and flies
pure pursuit), but the threat is no longer a scripted point: it is the generic small
fixed-wing UAV, simulated in 6-DOF and flown by its own autopilot.

  A. A loitering drone (18 m/s): it flies in from the North-West and circles over
     the area, 150 m around a point near the pad, 100 m up. The Iris is faster
     (~20.7 m/s in level flight) and catches it.
  B. A fast pass (30 m/s, near the fixed wing's top level speed of ~32.5 m/s),
     straight from the South-West at 120 m: the Iris cannot catch it.

How the two bodies are simulated: both fly in the same world (`jineta.arena`), each
with its own autopilot, and the Iris's guidance reads where the fixed wing is at the
start of every step. The pursuit is exactly the `scramble` of `iris_scramble.py`,
given the fixed wing as a vehicle instead of a scripted track. (Until phase 8 the
fixed wing was flown first and replayed; the result is the same to rounding error,
because it does not react to the Iris.)

Writes `docs/fixed_wing_scramble_viewer.html` (the viewer draws the threat as an
aircraft with its real attitude and shows its bank) and the start menu
`docs/index.html`, shows range and altitude against time and then opens the viewer
app on its menu. `--no-window` only writes the pages.
"""

import matplotlib.pyplot as plt
import numpy as np

from examples.iris_scramble import (
    ALERT_RANGE,
    CAPTURE_RADIUS,
    COLORS,
    DT,
    EVERY,
    GREY,
    ROOT,
    SITE,
    report,
    scramble,
)
from jineta.arena.world import VehicleEntity
from jineta.campo.atmosphere import isa_density
from jineta.jinete.fixed_wing import FixedWing
from jineta.jinete.fixed_wing_control import FixedWingAutopilot, orbit_course
from jineta.jinete.profiles import (
    SMALL_FIXED_WING,
    SMALL_FIXED_WING_AUTOPILOT,
    SMALL_FIXED_WING_PROPULSION,
)
from jineta.jinete.targets import RecordedTrack
from jineta.jinete.trim import trim_level_flight
from viz.app import show, window_requested
from viz.menu import write_menu
from viz.viewer import scenario_data, write_viewer

VIEWER_PATH = ROOT / "docs" / "fixed_wing_scramble_viewer.html"
IRIS_ORDERED_SPEED = 25.0  # [m/s]: more than it can fly, so it flies flat out


def fixed_wing_threat(speed, start, course, altitude, orbit=None):
    """The fixed wing as an entity of the world, named "threat".

    It starts trimmed for level flight at `speed`, at `start` (North, East [m]) and
    `altitude`, on `course` [rad]. Its autopilot holds the speed and the altitude;
    the course is fixed, or, with `orbit = (center NED, radius, clockwise)`, comes
    from the orbit guidance at every step.
    """
    aircraft = FixedWing(
        SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION, density=isa_density
    )
    trim = trim_level_flight(aircraft, speed, altitude=altitude, course=course)
    autopilot = FixedWingAutopilot(aircraft, trim, SMALL_FIXED_WING_AUTOPILOT, DT)
    state = np.array(trim.state)
    state[0:2] = start

    def control(t, y, snapshot):
        if orbit is not None:
            center, radius, clockwise = orbit
            autopilot.course_command = orbit_course(y[:3], center, radius, clockwise)
        autopilot(t, y)

    return VehicleEntity("threat", aircraft.derivative, state, control=control)


def replay(flight):
    """The threat's flight as a track, for the viewer (it evaluates the target's
    velocity at each sample it keeps)."""
    threat = flight.state("threat")
    return RecordedTrack(flight.times, threat[:, 0:3], threat[:, 3:6])


def main() -> None:
    """Fly both threats, run both scrambles, write the pages, plot, open the app."""
    # --- A: loitering drone ----------------------------------------------------------
    orbit = (np.array([60.0, 40.0, 0.0]), 150.0, True)
    loiter_threat = fixed_wing_threat(
        18.0, [650.0, -350.0], np.radians(140.0), 100.0, orbit=orbit
    )
    loiter, loiter_launch, loiter_exit, loiter_flight = scramble(
        loiter_threat, IRIS_ORDERED_SPEED, t_max=150.0
    )

    # --- B: fast pass ----------------------------------------------------------------
    pass_threat = fixed_wing_threat(30.0, [-600.0, -350.0], np.radians(40.0), 120.0)
    fast, fast_launch, fast_exit, fast_flight = scramble(
        pass_threat, IRIS_ORDERED_SPEED, t_max=90.0
    )

    report(
        "A. Loitering fixed wing: 18 m/s, orbit of 150 m",
        loiter,
        loiter_launch,
        loiter_exit,
        IRIS_ORDERED_SPEED,
    )
    report(
        "B. Fast fixed wing: 30 m/s, straight",
        fast,
        fast_launch,
        fast_exit,
        IRIS_ORDERED_SPEED,
    )

    # --- Viewer and menu -------------------------------------------------------------
    launch = {"range": ALERT_RANGE, "site": SITE}
    write_viewer(
        [
            scenario_data(
                loiter,
                replay(loiter_flight),
                EVERY,
                launch={**launch, "time": loiter_launch},
                target_kind="fixed_wing",
                target_attitudes=loiter_flight.state("threat")[:, 6:10],
                id="fixed-wing-loiter",
                name="A · loitering",
                detail="18 m/s, orbit of 150 m",
                target_speed=18.0,
                commanded_speed=IRIS_ORDERED_SPEED,
                threat_label="fixed-wing drone",
            ),
            scenario_data(
                fast,
                replay(fast_flight),
                EVERY,
                launch={**launch, "time": fast_launch},
                target_kind="fixed_wing",
                target_attitudes=fast_flight.state("threat")[:, 6:10],
                id="fixed-wing-pass",
                name="B · fast pass",
                detail="30 m/s, straight",
                target_speed=30.0,
                commanded_speed=IRIS_ORDERED_SPEED,
                threat_label="fixed-wing drone",
            ),
        ],
        VIEWER_PATH,
        dt=DT * EVERY,
        capture_radius=CAPTURE_RADIUS,
        subtitle="fixed wing · scramble",
        menu="index.html",
    )
    menu = write_menu(VIEWER_PATH.parent)
    print(f"\nViewer written to {VIEWER_PATH}; menu at {menu}")

    # --- Plots: range and altitude against time --------------------------------------
    fig, (ax_range, ax_alt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for key, result, t_launch, label in (
        ("A", loiter, loiter_launch, "A: loitering fixed wing, captured"),
        ("B", fast, fast_launch, "B: fast fixed wing, missed"),
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
    ax_range.set_yscale("log")
    ax_range.set_ylabel("Separation [m]")
    ax_range.set_title("Dashed vertical lines: launch. Dots: closest approach")
    ax_range.legend()
    ax_alt.set_ylabel("Altitude [m]")
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_title("Solid: Iris (from the pad at 0 m). Dotted: fixed wing")
    for ax in (ax_range, ax_alt):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Scramble against a fixed-wing drone: launch on alert, pure pursuit")
    fig.tight_layout()
    if window_requested():
        plt.show()  # close the plot to open the viewer app on its menu
        show(menu)


if __name__ == "__main__":
    main()
