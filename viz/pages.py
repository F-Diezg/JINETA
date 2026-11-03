"""Scenario pages: a folder of scenario files becomes one viewer page.

A folder of `scenarios/` holds one viewer page: a `page.yaml` and one scenario file
per case (a tab of the viewer).

    title: Generic target                 # the menu card's title (type of threat)
    description: >-                       # the menu card's text
      A scripted threat: ...
    page: iris_scramble_viewer.html       # the page written into docs/
    subtitle: "generic target · scramble" # the line above the viewer's title
    cases: [a-slow.yaml, b-fast.yaml]     # the tabs, in this order

`scenarios/menu.yaml` lists the folders the start menu shows, in order; their pages
link back to it.

    pages: [pursuit, generic-scramble, fixed-wing-scramble]

This module reads those files, makes the menu cards and turns a flown case into the
data the viewer replays. Like the rest of `viz`, it never runs the physics: whoever
flies the scenarios (`examples/run_scenarios.py`) hands it the results.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from jineta.arena.engagement import Battle, Engagement
from jineta.arena.scenario import Setup
from jineta.arena.spec import (
    Reader,
    ScenarioError,
    ScenarioSpec,
    load_scenario,
    suggest,
)
from jineta.arena.world import TrackEntity
from jineta.jinete.targets import RecordedTrack, TargetTrack
from viz.menu import MenuEntry
from viz.viewer import scenario_data, write_viewer

PAGE_FILE = "page.yaml"
MENU_LIST = "menu.yaml"
EVERY = 4  # the viewer keeps one sample in 4 (25 Hz at the 100 Hz step)
THREAT_LABELS = {"fixed_wing": "fixed-wing drone"}  # what the viewer's text calls it


@dataclass(frozen=True)
class PageSpec:
    """One `page.yaml`."""

    folder: Path
    title: str
    description: str
    page: str  # file name of the viewer page, in docs/
    subtitle: str
    cases: tuple[Path, ...]  # scenario files, in tab order


def load_page(folder: Path | str) -> PageSpec:
    """Read and check the `page.yaml` of a folder (its case files must exist)."""
    folder = Path(folder)
    path = folder / PAGE_FILE
    if not path.is_file():
        raise ScenarioError(f"{folder}: no {PAGE_FILE}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ScenarioError(f"{path}: not valid YAML: {error}") from error
    r = Reader(data, "", str(path))
    names = r.get("cases", kind="list")
    if not names or not all(isinstance(n, str) for n in names):
        r.fail("must be a list of scenario file names", "cases")
    for name in names:
        if not (folder / name).is_file():
            r.fail(f"no file '{name}' in {folder}", "cases")
    spec = PageSpec(
        folder=folder,
        title=r.text("title"),
        description=r.text("description"),
        page=r.text("page"),
        subtitle=r.text("subtitle"),
        cases=tuple(folder / name for name in names),
    )
    r.done()
    return spec


def page_folders(root: Path | str) -> list[Path]:
    """Every folder of `root` with a `page.yaml`: the ones in the menu first, in its
    order, then the others in name order."""
    listed = menu_folders(root)
    others = sorted(p.parent for p in Path(root).glob(f"*/{PAGE_FILE}"))
    return listed + [folder for folder in others if folder not in listed]


def menu_folders(root: Path | str) -> list[Path]:
    """The folders `menu.yaml` lists, in its order (none if there is no list)."""
    root = Path(root)
    path = root / MENU_LIST
    if not path.is_file():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ScenarioError(f"{path}: not valid YAML: {error}") from error
    r = Reader(data, "", str(path))
    names = r.get("pages", kind="list")
    r.done()
    available = sorted(p.parent.name for p in root.glob(f"*/{PAGE_FILE}"))
    for name in names:
        if name not in available:
            r.fail(
                f"no scenario folder '{name}'{suggest(str(name), available)}", "pages"
            )
    if len(set(names)) != len(names):
        r.fail("a folder is listed twice", "pages")
    return [root / name for name in names]


def menu_entries(root: Path | str) -> tuple[MenuEntry, ...]:
    """The cards of the start menu, from the `page.yaml` of each listed folder."""
    entries = []
    for folder in menu_folders(root):
        page = load_page(folder)
        entries.append(
            MenuEntry(
                kind=page.title,
                description=page.description,
                page=page.page,
                command=f"python -m examples.run_scenarios scenarios/{folder.name}",
            )
        )
    return tuple(entries)


def load_cases(page: PageSpec) -> list[ScenarioSpec]:
    return [load_scenario(path) for path in page.cases]


@dataclass(frozen=True)
class Case:
    """One flown scenario of a page, seen through its first pair (every current
    scenario has one interceptor and one threat; phase 8 step 7 shows them all)."""

    file: Path
    setup: Setup
    battle: Battle

    @property
    def spec(self) -> ScenarioSpec:
        return self.setup.spec

    @property
    def interceptor(self):
        return self.spec.interceptors[0]

    @property
    def threat(self):
        target = self.interceptor.target
        return next(t for t in self.spec.threats if t.id == target)

    @property
    def result(self):
        return self.battle.results[0]

    @property
    def launch_time(self) -> float:
        return float(self.battle.flight.launched_at[self.interceptor.id])

    @property
    def dome(self):
        return self.result.pairing.dome

    def target_track(self) -> TargetTrack:
        """The threat as a track: its script, or its simulated flight replayed."""
        entity = self.setup.world.entity(self.threat.id)
        if isinstance(entity, TrackEntity):
            return entity.track
        flight = self.battle.flight
        state = flight.state(self.threat.id)
        return RecordedTrack(flight.times, state[:, 0:3], state[:, 3:6])

    def engagement(self) -> Engagement:
        """The first pair as an `Engagement` (one interceptor, one target)."""
        flight = self.battle.flight
        target_speed = np.linalg.norm(self.target_track()(0.0)[1])
        return Engagement(
            times=flight.times,
            states=flight.state(self.interceptor.id),
            target_positions=flight.positions(self.threat.id),
            closest=self.result.closest,
            captured=self.result.captured,
            pursuit_feasible=self.interceptor.speed > target_speed,
        )


def case_data(case: Case, every: int = EVERY) -> dict:
    """The viewer data of one case (one tab)."""
    info = {
        "id": case.file.stem,
        "name": case.spec.name,
        "detail": case.spec.detail,
        "target_speed": case.threat.speed,
        "commanded_speed": case.interceptor.speed,
    }
    if case.threat.kind in THREAT_LABELS:
        info["threat_label"] = THREAT_LABELS[case.threat.kind]
    launch = None
    if case.dome is not None:
        launch = {
            "time": case.launch_time,
            "range": case.dome.radius,
            "site": case.dome.site,
        }
    attitudes = None
    target_kind = "generic"
    if case.threat.kind == "fixed_wing":
        target_kind = "fixed_wing"
        attitudes = case.battle.flight.state(case.threat.id)[:, 6:10]
    return scenario_data(
        case.engagement(),
        case.target_track(),
        every,
        launch=launch,
        target_kind=target_kind,
        target_attitudes=attitudes,
        **info,
    )


def write_page(
    page: PageSpec, cases: list[Case], folder: Path | str, menu: bool = False
) -> Path:
    """Write the viewer page of a folder of scenarios into `folder` (docs/); with
    `menu`, the page links back to the start menu."""
    if not cases:
        raise ValueError("a page needs at least one case")
    steps = {case.spec.step for case in cases}
    radii = {case.interceptor.capture_radius for case in cases}
    if len(steps) != 1 or len(radii) != 1:
        raise ScenarioError(
            f"{page.folder}: the cases of one page must share the step and the "
            "capture radius (the viewer has one of each)"
        )
    return write_viewer(
        [case_data(case) for case in cases],
        Path(folder) / page.page,
        dt=steps.pop() * EVERY,
        capture_radius=radii.pop(),
        subtitle=page.subtitle,
        menu="index.html" if menu else None,
    )
