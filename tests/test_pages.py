"""Tests for the scenario pages (viz/pages.py) and the scenario files of the repo.

The files in `scenarios/` are checked here (they all load, every case exists), so a
broken file fails the tests rather than the next run of the examples. The cases
flown here are short made-up ones: the real ones are flown by the examples.
"""

import copy
from pathlib import Path

import pytest
import yaml

from examples.run_scenarios import play_folder
from jineta.arena.spec import ScenarioError
from viz.menu import MenuEntry, page_cases, write_menu
from viz.pages import (
    load_cases,
    load_page,
    menu_entries,
    menu_folders,
    page_folders,
    write_page,
)
from viz.viewer import read_viewer

ROOT = Path(__file__).resolve().parents[1]


def test_every_scenario_file_of_the_repo_loads():
    folders = page_folders(ROOT / "scenarios")
    order = [
        "pursuit",
        "generic-scramble",
        "fixed-wing-scramble",
        "two-pads",
        "radar-scramble",
    ]
    assert [f.name for f in folders] == order
    for folder in folders:
        page = load_page(folder)
        assert len(load_cases(page)) == len(page.cases) >= 2
    # the start menu: one card per listed folder, from its page.yaml
    entries = menu_entries(ROOT / "scenarios")
    assert [e.kind for e in entries] == [
        "Generic target · in the air",
        "Generic target · scramble",
        "Fixed wing · scramble",
        "Two pads · several threats",
        "Radar · scramble",
    ]
    assert entries[0].page == "iris_pursuit_viewer.html"
    assert entries[2].command == (
        "python -m examples.run_scenarios scenarios/fixed-wing-scramble"
    )


def write(folder, name, data):
    (folder / name).write_text(yaml.safe_dump(data), encoding="utf-8")


PAGE = {
    "title": "Test",
    "description": "made-up cases",
    "page": "test_viewer.html",
    "subtitle": "test",
    "cases": ["scramble.yaml", "drone.yaml"],
}
SCRAMBLE = {
    "name": "A · scramble",
    "detail": "straight in",
    "duration": 3,
    "environment": {"atmosphere": "sea_level"},
    "sites": [{"id": "pad", "position": [0, 0], "alert_range": 100}],
    "interceptors": [
        {"id": "iris", "vehicle": "iris", "site": "pad", "target": "t", "speed": 10}
    ],
    "threats": [
        {
            "id": "t",
            "kind": "straight",
            "position": [110, 0],
            "altitude": 30,
            "speed": 20,
            "course_deg": 180,
        }
    ],
}
DRONE = {
    "name": "B · drone",
    "duration": 1,
    "interceptors": [
        {
            "id": "iris",
            "vehicle": "iris",
            "start": {"position": [0, 0], "altitude": 100},
            "target": "d",
            "speed": 10,
        }
    ],
    "threats": [
        {
            "id": "d",
            "kind": "fixed_wing",
            "vehicle": "small_fixed_wing",
            "position": [200, 0],
            "altitude": 100,
            "speed": 22,
            "course_deg": 90,
        }
    ],
}


def test_a_folder_becomes_a_page_with_one_tab_per_case(tmp_path):
    write(tmp_path, "page.yaml", PAGE)
    write(tmp_path, "scramble.yaml", SCRAMBLE)
    write(tmp_path, "drone.yaml", DRONE)
    page, cases = play_folder(tmp_path)
    path = write_page(page, cases, tmp_path / "docs", menu=True)
    data = read_viewer(path)

    assert data["meta"]["subtitle"] == "test"
    assert data["meta"]["menu"] == "index.html"  # links back to the menu
    assert data["meta"]["dt"] == pytest.approx(0.04)
    scramble, drone = data["scenarios"]

    assert (scramble["id"], scramble["name"]) == ("scramble", "A · scramble")
    assert scramble["target_kind"] == "generic" and "threat_label" not in scramble
    # inside the 100 m dome once 110 - 20 t <= sqrt(100^2 - 30^2) = 95.39 m, i.e.
    # from t = 0.7303 s: the first step after that is t = 0.74 s
    assert scramble["launch"]["time"] == pytest.approx(0.74, abs=1e-9)
    assert scramble["launch"]["range"] == 100.0
    assert (scramble["target_speed"], scramble["commanded_speed"]) == (20.0, 10.0)

    assert drone["target_kind"] == "fixed_wing"
    assert drone["threat_label"] == "fixed-wing drone"
    assert len(drone["target"]["quat"]) == len(drone["t"])  # its attitude
    assert "launch" not in drone  # flying from the start


def test_page_mistakes_say_where(tmp_path):
    with pytest.raises(ScenarioError, match=r"no page\.yaml"):
        load_page(tmp_path)
    write(tmp_path, "page.yaml", {**PAGE, "cases": ["missing.yaml"]})
    with pytest.raises(ScenarioError, match=r"no file 'missing\.yaml'"):
        load_page(tmp_path)
    write(tmp_path, "scramble.yaml", SCRAMBLE)
    write(tmp_path, "drone.yaml", DRONE)
    write(tmp_path, "page.yaml", {**PAGE, "titel": "x"})
    with pytest.raises(ScenarioError, match="did you mean 'title'"):
        load_page(tmp_path)


def test_the_cases_of_a_page_share_one_step(tmp_path):
    write(tmp_path, "page.yaml", PAGE)
    write(tmp_path, "scramble.yaml", {**SCRAMBLE, "step": 0.02})
    write(tmp_path, "drone.yaml", DRONE)
    page, cases = play_folder(tmp_path)
    with pytest.raises(ScenarioError, match="share the step"):
        write_page(page, cases, tmp_path / "docs")


def test_the_menu_list_says_which_folders_and_in_what_order(tmp_path):
    for name in ("b-page", "a-page", "hidden"):
        (tmp_path / name).mkdir()
        write(tmp_path / name, "page.yaml", PAGE)
        write(tmp_path / name, "scramble.yaml", SCRAMBLE)
        write(tmp_path / name, "drone.yaml", DRONE)
    assert menu_folders(tmp_path) == []  # no menu.yaml: no cards
    write(tmp_path, "menu.yaml", {"pages": ["b-page", "a-page"]})
    assert [f.name for f in menu_folders(tmp_path)] == ["b-page", "a-page"]
    # the folders of the menu first, in its order; the others still get a page
    assert [f.name for f in page_folders(tmp_path)] == ["b-page", "a-page", "hidden"]
    assert len(menu_entries(tmp_path)) == 2

    write(tmp_path, "menu.yaml", {"pages": ["b-pag"]})
    with pytest.raises(ScenarioError, match="did you mean 'b-page'"):
        menu_folders(tmp_path)
    write(tmp_path, "menu.yaml", {"pages": ["a-page", "a-page"]})
    with pytest.raises(ScenarioError, match="listed twice"):
        menu_folders(tmp_path)


TWO_PADS = {
    "name": "C · two pads",
    "duration": 2,
    "environment": {"atmosphere": "sea_level"},
    "sites": [
        {"id": "north", "position": [100, 0], "alert_range": 400},
        {"id": "south", "position": [-100, 0]},
    ],
    "interceptors": [
        {"id": "n", "vehicle": "iris", "site": "north", "target": "a", "speed": 10},
        {"id": "s", "vehicle": "iris", "site": "south", "target": "b", "speed": 10},
    ],
    "threats": [
        {
            "id": "a",
            "kind": "straight",
            "position": [600, 0],
            "altitude": 50,
            "speed": 20,
            "course_deg": 180,
        },
        {
            "id": "b",
            "kind": "turning",
            "position": [-300, 200],
            "altitude": 50,
            "speed": 5,
            "course_deg": 0,
            "turn_rate_deg_s": 3,
        },
        {
            "id": "lone",
            "kind": "straight",
            "position": [0, 500],
            "altitude": 80,
            "speed": 10,
            "course_deg": 270,
        },
    ],
}


def test_a_case_with_several_entities_exports_them_all(tmp_path):
    """Two pads, two pairs and an unassigned threat: the first pair as always, every
    other entity in `others`, every pair in `pairs`, the one dome in `domes`."""
    write(tmp_path, "page.yaml", {**PAGE, "cases": ["two.yaml"]})
    write(tmp_path, "two.yaml", TWO_PADS)
    page, cases = play_folder(tmp_path)
    data = read_viewer(write_page(page, cases, tmp_path / "docs"))
    (case,) = data["scenarios"]

    assert (case["interceptor"]["id"], case["target"]["id"]) == ("n", "a")
    assert case["interceptor"]["role"] == "interceptor"
    assert case["target"]["kind"] == "generic"
    others = {o["id"]: o for o in case["others"]}
    assert set(others) == {"s", "b", "lone"}  # the unassigned threat is drawn too
    assert (others["s"]["role"], others["s"]["kind"]) == ("interceptor", "multirotor")
    assert "quat" in others["s"] and "quat" not in others["lone"]
    assert "launch_time" not in others["s"]  # no dome: it flies from t = 0
    for other in case["others"]:
        assert len(other["pos"]) == len(other["vel"]) == len(case["t"])

    pairs = case["pairs"]
    assert [(p["interceptor"], p["target"]) for p in pairs] == [("n", "a"), ("s", "b")]
    assert all(p["outcome"] == "open" for p in pairs)  # 2 s: nobody got there
    assert pairs[0]["launch"]["range"] == 400.0 and "launch" not in pairs[1]
    # "a" is still 460 m from the north pad at the end: "n" never launched, and the
    # viewer shows it waiting until the end of the run
    assert pairs[0]["launch"]["time"] == case["t"][-1] == pytest.approx(2.0)
    assert case["launch"]["time"] == pytest.approx(2.0)
    assert case["domes"] == [{"site": [100.0, 0.0, 0.0], "range": 400.0}]


def test_the_menu_shows_how_every_pair_ended(tmp_path):
    """A case with several pairs lists one row per pair, coloured by its outcome,
    and the page's strip has one segment per pair."""
    write(tmp_path, "page.yaml", {**PAGE, "cases": ["two.yaml"]})
    write(tmp_path, "two.yaml", TWO_PADS)
    page, cases = play_folder(tmp_path)
    docs = tmp_path / "docs"
    write_page(page, cases, docs)
    entry = MenuEntry("Two", "test", page.page, "python -m make")
    (case,) = page_cases(docs / page.page)
    assert [(p.label, p.outcome) for p in case.pairs] == [
        ("n → a", "open"),
        ("s → b", "open"),
    ]
    text = write_menu(docs, (entry,)).read_text(encoding="utf-8")
    assert text.count('<tr class="pair">') == 2
    assert text.count('<span class="o-open"></span>') == 2  # the strip
    assert "0 of 2 pairs captured, 1 case" in text
    assert text.count('class="result o-open">still open') == 2


def test_a_case_says_how_it_ended_and_what_guided_its_interceptor(tmp_path):
    """Every case exports its first pair's outcome and end time, and whether the
    interceptor flew on the truth or on a radar's track (the viewer tells it)."""
    radar = copy.deepcopy(SCRAMBLE)
    radar["sites"][0]["radar"] = "short_range"
    radar["interceptors"][0]["knowledge"] = "radar"
    write(tmp_path, "page.yaml", PAGE)
    write(tmp_path, "scramble.yaml", radar)
    write(tmp_path, "drone.yaml", DRONE)
    page, cases = play_folder(tmp_path)
    data = read_viewer(write_page(page, cases, tmp_path / "docs"))
    scramble, drone = data["scenarios"]
    assert (scramble["knowledge"], drone["knowledge"]) == ("radar", "perfect")
    assert drone["outcome"] == "open"  # 1 s from hover: the run's time limit
    assert drone["end_time"] == pytest.approx(1.0)
    assert scramble["outcome"] == cases[0].result.outcome
