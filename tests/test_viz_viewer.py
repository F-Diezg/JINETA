"""Tests for the viewer export (viz/viewer.py) and the start menu (viz/menu.py).

Only our side is tested: what data goes into the pages and what the menu shows
(links, cases, results). The pages themselves (three.js) are not run here.
"""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jineta.arena.engagement import Engagement
from jineta.core.events import ClosestApproach
from jineta.jinete.targets import StraightLineTarget, track_positions
from viz.menu import MENU_FILE, MenuEntry, page_cases, write_menu
from viz.viewer import read_viewer, scenario_data, write_viewer

TARGET = StraightLineTarget(start=[100.0, 0.0, -50.0], velocity=[-5.0, 0.0, 0.0])
ENTRIES = (
    MenuEntry("Kind one", "first page", "one.html", "python -m make one"),
    MenuEntry("Kind two", "second page", "two.html", "python -m make two"),
)
N = 11


def engagement():
    """A small made-up engagement: 11 samples, 0.1 s apart."""
    times = 0.1 * np.arange(N)
    states = np.zeros((N, 13))
    states[:, 6] = 1.0  # level attitude
    states[:, 0] = 8.0 * times
    return Engagement(
        times=times,
        states=states,
        target_positions=track_positions(TARGET, times),
        closest=ClosestApproach(time=0.95, distance=3.0, step=9),
        captured=False,
        pursuit_feasible=True,
    )


def write_page(path, *results):
    """A viewer page with one case per (name, captured) pair."""
    scenarios = [
        scenario_data(
            dataclasses.replace(engagement(), captured=captured),
            TARGET,
            id=f"case-{i}",
            name=name,
            detail=f"detail {i}",
        )
        for i, (name, captured) in enumerate(results)
    ]
    return write_viewer(scenarios, path, dt=0.4, capture_radius=0.5)


def test_a_scripted_target_has_no_attitude():
    data = scenario_data(engagement(), TARGET, every=4)
    assert data["target_kind"] == "generic"
    assert "quat" not in data["target"]
    assert len(data["t"]) == 4  # samples 0, 4, 8 and the last one, 10


def test_a_fixed_wing_target_carries_its_attitude_downsampled():
    attitudes = np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))
    attitudes[:, 1] = np.arange(N)  # mark each sample (not a unit quaternion)
    data = scenario_data(
        engagement(),
        TARGET,
        every=4,
        target_kind="fixed_wing",
        target_attitudes=attitudes,
    )
    assert data["target_kind"] == "fixed_wing"
    assert [q[1] for q in data["target"]["quat"]] == [0.0, 4.0, 8.0, 10.0]


def test_rejects_an_unknown_kind_or_wrong_attitudes():
    with pytest.raises(ValueError):
        scenario_data(engagement(), TARGET, target_kind="balloon")
    with pytest.raises(ValueError):
        scenario_data(
            engagement(),
            TARGET,
            target_kind="fixed_wing",
            target_attitudes=np.ones((3, 4)),
        )


def test_page_carries_the_subtitle_and_the_menu_link():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "viewer.html"
        scenario = scenario_data(engagement(), TARGET)
        write_viewer([scenario], path, dt=0.4, capture_radius=0.5)
        meta = read_viewer(path)["meta"]
        assert meta["menu"] is None  # no link by default (the MVP page)
        write_viewer(
            [scenario],
            path,
            dt=0.4,
            capture_radius=0.5,
            subtitle="fixed wing · scramble",
            menu="index.html",
        )
        meta = read_viewer(path)["meta"]
        assert meta["subtitle"] == "fixed wing · scramble"
        assert meta["menu"] == "index.html"


def test_reading_a_page_gives_back_what_was_written():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "viewer.html"
        scenario = scenario_data(engagement(), TARGET, name="A · test")
        write_viewer([scenario], path, dt=0.4, capture_radius=0.5)
        data = read_viewer(path)
        assert data["scenarios"][0]["name"] == "A · test"
        assert data["scenarios"][0]["t"] == scenario["t"]
        assert data["meta"]["dt"] == 0.4
        other = Path(folder) / "other.html"
        other.write_text("<title>not a viewer</title>", encoding="utf-8")
        with pytest.raises(ValueError):
            read_viewer(other)


def test_menu_cases_and_results_come_from_the_page():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        entry = ENTRIES[0]
        write_page(folder / entry.page, ("A · caught", True), ("B · escaped", False))
        cases = page_cases(folder / entry.page)
        assert [(c.name, c.captured) for c in cases] == [
            ("A · caught", 1),
            ("B · escaped", 0),
        ]
        assert cases[1].pairs[0].distance == 3.0  # the closest approach of the page
        text = write_menu(folder, ENTRIES).read_text(encoding="utf-8")
        caught = text.index("<b>A · caught</b><small>detail 0</small>")
        escaped = text.index("<b>B · escaped</b><small>detail 1</small>")
        assert caught < text.index('class="result o-captured">captured') < escaped
        assert escaped < text.index('class="result o-missed">missed')
        assert "1 of 2 pairs captured, 2 cases" in text
        assert "<b>1</b> scenario page, <b>2</b> cases: <b>1</b> of 2" in text
        assert "1 more page is not generated" in text  # ENTRIES[1]


def test_menu_links_the_pages_that_exist_and_names_the_example_for_the_others():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        present, missing = ENTRIES
        write_page(folder / present.page, ("A · caught", True))
        path = write_menu(folder, ENTRIES)
        assert path == folder / MENU_FILE
        text = path.read_text(encoding="utf-8")
        assert f'href="{present.page}"' in text
        assert f'href="{missing.page}"' not in text
        assert missing.command in text
        assert "__MENU_CARDS__" not in text
