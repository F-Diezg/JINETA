"""Scramble against a fixed-wing drone: the phase 7 scramble with a simulated threat.

The same launch on alert as `iris_scramble.py` (the Iris waits on its pad at the
origin, rotors stopped, launches when the threat comes within the alert range and
flies pure pursuit), but the threat is no longer a scripted point: it is the generic
small fixed-wing UAV, simulated in 6-DOF and flown by its own autopilot.

  A. A loitering drone (18 m/s): it flies in from the North-West and circles over
     the area, 150 m around a point near the pad, 100 m up. The Iris is faster
     (~20.7 m/s in level flight) and catches it.
  B. A fast pass (30 m/s, near the fixed wing's top level speed of ~32.5 m/s),
     straight from the South-West at 120 m: the Iris cannot catch it.

Both fly in the same world (`jineta.arena`), each with its own autopilot, in ISA
air; the Iris's guidance reads where the fixed wing is at the start of every step.
The cases are the files of `scenarios/fixed-wing-scramble/`. This script flies
them, prints the verdicts, plots range and altitude against time, writes
`docs/fixed_wing_scramble_viewer.html` (the viewer draws the threat as an aircraft
with its real attitude and shows its bank) and the start menu, and opens the viewer
app on the menu. `--no-window` only writes the pages.
"""

import matplotlib.pyplot as plt

from examples.iris_scramble import plot_scramble
from examples.run_scenarios import SCENARIOS, play_folder, report, write_folder
from viz.app import show, window_requested


def main() -> None:
    """Fly both cases, print the verdicts, write the pages, plot, open the app."""
    page, cases = play_folder(SCENARIOS / "fixed-wing-scramble")
    for case in cases:
        report(case)
    menu = write_folder(page, cases)
    print(f"Menu at {menu}")
    plot_scramble(
        cases,
        "Scramble against a fixed-wing drone: launch on alert, pure pursuit",
        "fixed wing",
    )
    if window_requested():
        plt.show()  # close the plot to open the viewer app on its menu
        show(menu)


if __name__ == "__main__":
    main()
