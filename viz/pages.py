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

from jineta.arena.engagement import CAPTURED, Battle, DuelResult, Engagement
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
from viz.viewer import keep_indices, rounded, scenario_data, write_viewer

PAGE_FILE = "page.yaml"
MENU_LIST = "menu.yaml"
EVERY = 4  # the viewer keeps one sample in 4 (25 Hz at the 100 Hz step)
THREAT_LABELS = {"fixed_wing": "fixed-wing drone"}  # what the viewer's text calls it
TRACK_KEYS = ("pos", "vel", "quat")  # the arrays of an entity; the rest is about it


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


def launch_time(flight, name: str) -> float:
    """When an entity launched; the end of the run if it never did."""
    launched = flight.launched_at[name]
    return float(flight.times[-1] if launched is None else launched)


@dataclass(frozen=True)
class Case:
    """One flown scenario of a page. The properties without an argument look at its
    first pair, the one the viewer shows first; `case_data` exports every entity and
    every pair."""

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
    def launched(self) -> bool:
        """False when the interceptor never left its pad (its target never came
        within the alert range)."""
        return self.battle.flight.launched_at[self.interceptor.id] is not None

    @property
    def launch_time(self) -> float:
        """When the first pair's interceptor launched (the end of the run if it
        never did: it waited the whole time)."""
        return launch_time(self.battle.flight, self.interceptor.id)

    @property
    def dome(self):
        return self.result.pairing.dome

    def target_track(self, name: str | None = None) -> TargetTrack:
        """A threat as a track (the first pair's by default): its script, or its
        simulated flight replayed."""
        name = self.threat.id if name is None else name
        entity = self.setup.world.entity(name)
        if isinstance(entity, TrackEntity):
            return entity.track
        flight = self.battle.flight
        state = flight.state(name)
        return RecordedTrack(flight.times, state[:, 0:3], state[:, 3:6])

    def destroyed_at(self) -> dict[str, float]:
        """When each destroyed entity was hit (the capture of its pair) [s]."""
        times: dict[str, float] = {}
        for result in self.battle.results:
            if result.outcome == CAPTURED:
                for name in (result.pairing.interceptor, result.pairing.target):
                    times[name] = min(times.get(name, np.inf), result.end_time)
        return times

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


def entity_data(case: Case, name: str, keep: np.ndarray) -> dict:
    """One entity of the world, downsampled, for the viewer to draw: its track,
    velocity and attitude (a vehicle), when it was launched and destroyed."""
    spec = case.spec
    flight = case.battle.flight
    entity = case.setup.world.entity(name)
    interceptors = {i.id for i in spec.interceptors}
    threat_kinds = {t.id: t.kind for t in spec.threats}
    if name in interceptors:
        role, kind = "interceptor", "multirotor"
    else:
        role = "threat"
        kind = "fixed_wing" if threat_kinds[name] == "fixed_wing" else "generic"
    times = flight.times[keep]
    data = {"id": name, "role": role, "kind": kind}
    data["pos"] = rounded(flight.positions(name)[keep], 3)
    if isinstance(entity, TrackEntity):  # a stopped (destroyed) track has no speed
        empty = np.empty(0)
        data["vel"] = rounded([entity.velocity(t, empty) for t in times], 3)
    else:
        state = flight.state(name)[keep]
        data["vel"] = rounded(state[:, 3:6], 3)
        data["quat"] = rounded(state[:, 6:10], 5)
    launched = flight.launched_at[name]
    if launched is not None and launched > 0.0:
        data["launch_time"] = round(float(launched), 4)
    destroyed = case.destroyed_at().get(name)
    if destroyed is not None:
        data["destroyed_at"] = round(float(destroyed), 4)
    return data


def pair_data(case: Case, result: DuelResult) -> dict:
    """One pair (interceptor -> target) and how it ended, for the viewer."""
    pair = result.pairing
    flight = case.battle.flight
    interceptor = next(i for i in case.spec.interceptors if i.id == pair.interceptor)
    threat = next(t for t in case.spec.threats if t.id == pair.target)
    t_ca = result.closest.time

    def at_closest(name):
        track = flight.positions(name)
        return [float(np.interp(t_ca, flight.times, track[:, i])) for i in range(3)]

    target_speed = float(np.linalg.norm(case.target_track(threat.id)(0.0)[1]))
    data = {
        "interceptor": pair.interceptor,
        "target": pair.target,
        "outcome": result.outcome,
        "captured": result.captured,
        "end_time": round(result.end_time, 4),
        "closest": {
            "time": round(t_ca, 4),
            "distance": round(result.closest.distance, 4),
            "interceptor": rounded(at_closest(pair.interceptor), 3),
            "target": rounded(at_closest(pair.target), 3),
        },
        "commanded_speed": interceptor.speed,
        "target_speed": threat.speed,
        "target_kind": "fixed_wing" if threat.kind == "fixed_wing" else "generic",
        "pursuit_feasible": interceptor.speed > target_speed,
    }
    if threat.kind in THREAT_LABELS:
        data["threat_label"] = THREAT_LABELS[threat.kind]
    if pair.dome is not None:
        data["launch"] = {
            "time": round(launch_time(flight, pair.interceptor), 4),
            "range": pair.dome.radius,
            "site": rounded(pair.dome.site, 3),
        }
    return data


def case_data(case: Case, every: int = EVERY) -> dict:
    """The viewer data of one case (one tab).

    The first pair is exported as the viewer has always read it (`interceptor`,
    `target`, `closest`, ...). A case with more entities adds `others` (every entity
    outside that pair), and one with several pairs adds `pairs` (each pair and its
    end, so the viewer can follow any of them) and `domes` (every alert dome).
    """
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
    data = scenario_data(
        case.engagement(),
        case.target_track(),
        every,
        launch=launch,
        target_kind=target_kind,
        target_attitudes=attitudes,
        **info,
    )
    world = case.setup.world
    results = case.battle.results
    first = {case.interceptor.id, case.threat.id}
    if len(world.entities) == len(first) and len(results) == 1:
        return data  # one interceptor, one threat: the page reads as it always did

    keep = keep_indices(len(case.battle.flight.times), every)
    for key, name in (("interceptor", case.interceptor.id), ("target", case.threat.id)):
        meta = entity_data(case, name, keep)  # id, role, kind, launch and hit times
        data[key].update({k: v for k, v in meta.items() if k not in TRACK_KEYS})
    data["others"] = [
        entity_data(case, entity.name, keep)
        for entity in world.entities
        if entity.name not in first
    ]
    data["pairs"] = [pair_data(case, result) for result in results]
    domes = {}
    for result in results:
        dome = result.pairing.dome
        if dome is not None:
            domes[(tuple(dome.site), dome.radius)] = {
                "site": rounded(dome.site, 3),
                "range": dome.radius,
            }
    data["domes"] = list(domes.values())
    return data


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
