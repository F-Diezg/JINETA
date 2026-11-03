"""Fly the scenario files and write their viewer pages and the start menu.

    python -m examples.run_scenarios                      # every folder, open the menu
    python -m examples.run_scenarios scenarios/pursuit    # only these folders
    python -m examples.run_scenarios --no-window          # only write the pages

Each folder of `scenarios/` with a `page.yaml` is one viewer page in `docs/`: every
case file of the folder is flown (`jineta.arena.scenario.play`), its verdict is
printed and it becomes one tab of the page. The start menu shows the folders listed
in `scenarios/menu.yaml`, with the title and text of their `page.yaml`. The examples
(`iris_pursuit.py`, `iris_scramble.py`, `fixed_wing_scramble.py`) use the same
functions and add their own plots.
"""

import sys
from pathlib import Path

import numpy as np

from jineta.arena.engagement import CAPTURED, ESCAPED, PASSED
from jineta.arena.scenario import play
from viz.app import show, window_requested
from viz.menu import write_menu
from viz.pages import (
    Case,
    PageSpec,
    load_cases,
    load_page,
    menu_entries,
    menu_folders,
    page_folders,
    write_page,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"
DOCS = ROOT / "docs"


def play_folder(folder: Path | str) -> tuple[PageSpec, list[Case]]:
    """Fly every case of one folder."""
    page = load_page(folder)
    cases = []
    for path, spec in zip(page.cases, load_cases(page), strict=True):
        setup, battle = play(spec)
        cases.append(Case(path, setup, battle))
    return page, cases


def report(case: Case) -> None:
    """Print the verdict of one case."""
    result = case.engagement()
    t_launch = case.launch_time
    flying = result.times >= t_launch
    speed = np.linalg.norm(result.states[:, 3:6], axis=1)
    top = int(np.argmax(speed))
    climb = -result.states[top, 5]
    horizontal = np.linalg.norm(result.states[top, 3:5])
    verdict = "PASS" if result.captured else "FAIL"
    closest = result.closest

    print(f"\n{case.spec.name} ({case.spec.detail})")
    if case.dome is not None and case.launched:
        print(
            f"  alert: threat within {case.dome.radius:.0f} m at T+{t_launch:.2f} s, "
            "launch"
        )
    elif case.dome is not None:
        print(f"  the threat never came within {case.dome.radius:.0f} m: no launch")
    after = f" ({closest.time - t_launch:.2f} s after the launch)" if t_launch else ""
    print(
        f"  {verdict}: closest approach {closest.distance:.2f} m "
        f"at T+{closest.time:.2f} s{after}"
    )
    print(
        f"  ordered {case.interceptor.speed:.0f} m/s; fastest {speed[top]:.1f} m/s "
        f"({horizontal:.1f} horizontal, {climb:.1f} climbing)"
    )
    if case.dome is not None:
        lowest = -result.states[flying, 2].max() + case.dome.site[2]
        # + 0.0 turns -0.0 into 0.0 for printing
        print(f"  lowest height above the pad after the launch {lowest + 0.0:.3f} m")
    end = {
        CAPTURED: "capture",
        ESCAPED: "the threat left the dome",
        PASSED: "a pass (it opened up past the margin)",
    }.get(case.result.outcome, "the time limit")
    print(f"  run ended at T+{result.times[-1]:.2f} s: {end}")
    if len(case.battle.results) > 1:
        print("  every pair (the lines above follow the first one):")
        launched = case.battle.flight.launched_at
        for pair in case.battle.results:
            p = pair.pairing
            launch = launched[p.interceptor]
            when = f", launched T+{launch:.2f} s" if launch else ""
            print(
                f"    {p.interceptor} -> {p.target}: {pair.outcome}, closest "
                f"{pair.closest.distance:.2f} m at T+{pair.closest.time:.2f} s{when}"
            )


def write_folder(page: PageSpec, cases: list[Case]) -> Path:
    """Write the page of one folder and the start menu; return the menu."""
    listed = [folder.resolve() for folder in menu_folders(SCENARIOS)]
    in_menu = page.folder.resolve() in listed
    path = write_page(page, cases, DOCS, menu=in_menu)
    print(f"\nViewer written to {path}")
    return write_menu(DOCS, menu_entries(SCENARIOS))


def main(argv: list[str] | None = None) -> None:
    args = [a for a in (sys.argv[1:] if argv is None else argv) if a != "--no-window"]
    folders = [Path(a) for a in args] or page_folders(SCENARIOS)
    menu = None
    for folder in folders:
        page, cases = play_folder(folder)
        print(f"\n=== {page.title}: {folder}")
        for case in cases:
            report(case)
        menu = write_folder(page, cases)
    if menu is not None and window_requested(argv):
        show(menu)


if __name__ == "__main__":
    main()
