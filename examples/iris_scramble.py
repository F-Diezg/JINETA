"""Scramble: the Iris waits on the ground and takes off when a threat comes near.

Same vehicle and guidance as the MVP (`iris_pursuit.py`), but a more realistic start.
The Iris sits on its launch pad at the origin, at rest, rotors stopped, altitude 0.
A threat flies a simple path towards the area. When it comes within the alert range
of the pad (straight-line distance, as a radar measures it), the Iris launches and
flies pure pursuit at its commanded speed.

  A. A slow threat (8 m/s, gentle left turn, 110 m up): the Iris climbs, catches it.
  B. A fast threat (25 m/s straight, 120 m up): it is past before the Iris gets up
     to it; the Iris misses, keeps chasing and falls behind.

Each run ends at the capture or when the threat leaves the alert dome, whichever
comes first: past that point the pad no longer cares about it.

The cases are the files of `scenarios/generic-scramble/`; this script flies them,
prints the verdicts, plots range and altitude against time, writes the viewer page
`docs/iris_scramble_viewer.html` and the start menu, and opens the viewer app on
the menu. `--no-window` only writes the pages. `fixed_wing_scramble.py` reuses
`plot_scramble`.

How the launch is simulated (`jineta.arena`), with no new physics:
  - The Iris and the threat fly in one world. The Iris is a waiting entity: until
    its launch its derivative is zero, so it stays exactly at rest on the pad, which
    is what a vehicle with its motors off on the ground does (there is no ground
    model).
  - A launch rule (`LaunchOnAlert`) checks the alert once per control step, before
    the controls: the Iris launches at the first t = k * dt at which the threat is
    inside the dome, and its pursuit control runs from that same step.
  - The pair (Iris -> threat) defends the dome and has no pass margin: it ends at
    the capture or when the threat leaves the dome.
"""

import matplotlib.pyplot as plt
import numpy as np

from examples.run_scenarios import SCENARIOS, play_folder, report, write_folder
from viz.app import show, window_requested

COLORS = {"A": "#2a78d6", "B": "#eb6834"}
GREY = "#888780"


def plot_scramble(cases, title, threat_word):
    """Range and altitude against time for the cases of a scramble page."""
    fig, (ax_range, ax_alt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))
    for key, case in zip(COLORS, cases, strict=False):
        color = COLORS[key]
        result = case.engagement()
        verdict = "captured" if result.captured else "missed"
        separation = np.linalg.norm(
            result.target_positions - result.states[:, :3], axis=1
        )
        label = f"{case.spec.name}: {verdict}"
        ax_range.plot(result.times, separation, color=color, lw=2, label=label)
        ax_range.plot(result.closest.time, result.closest.distance, "o", color=color)
        ax_alt.plot(result.times, -result.states[:, 2], color=color, lw=2)
        ax_alt.plot(
            result.times, -result.target_positions[:, 2], color=color, lw=1, ls=":"
        )
        for ax in (ax_range, ax_alt):
            ax.axvline(case.launch_time, color=color, lw=1, ls="--")

    alert_range = cases[0].dome.radius
    capture_radius = cases[0].interceptor.capture_radius
    ax_range.axhline(alert_range, color=GREY, lw=1, ls="--")
    ax_range.axhline(capture_radius, color=GREY, lw=1, ls="--")
    ax_range.text(
        0.5, alert_range * 1.1, f"alert range {alert_range:.0f} m", color=GREY
    )
    ax_range.text(0.5, capture_radius * 1.2, f"capture {capture_radius} m", color=GREY)
    ax_range.set_yscale("log")
    ax_range.set_ylabel("Separation [m]")
    ax_range.set_title("Dashed vertical lines: launch. Dots: closest approach")
    ax_range.legend()
    ax_alt.set_ylabel("Altitude [m]")
    ax_alt.set_xlabel("Time [s]")
    ax_alt.set_title(f"Solid: Iris (from the pad at 0 m). Dotted: {threat_word}")
    for ax in (ax_range, ax_alt):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main() -> None:
    """Fly both cases, print the verdicts, write the pages, plot, open the app."""
    page, cases = play_folder(SCENARIOS / "generic-scramble")
    for case in cases:
        report(case)
    menu = write_folder(page, cases)
    print(f"Menu at {menu}")
    plot_scramble(cases, "Scramble: launch on alert, then pure pursuit", "threat")
    if window_requested():
        plt.show()  # close the plot to open the viewer app on its menu
        show(menu)


if __name__ == "__main__":
    main()
