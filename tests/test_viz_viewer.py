"""Tests for the viewer export (viz/viewer.py) and the start menu (viz/menu.py).

Only our side is tested: what data goes into the pages and which links the menu
shows. The pages themselves (three.js) are not run here.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from jineta.core.events import ClosestApproach
from jineta.engagement import Engagement
from jineta.jinete.targets import StraightLineTarget, track_positions
from viz.menu import ENTRIES, MENU_FILE, write_menu
from viz.viewer import scenario_data, write_viewer

TARGET = StraightLineTarget(start=[100.0, 0.0, -50.0], velocity=[-5.0, 0.0, 0.0])
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


def page_data(path):
    text = Path(path).read_text(encoding="utf-8")
    start = text.index('id="engagement-data">') + len('id="engagement-data">')
    return json.loads(text[start : text.index("</script>", start)])


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
        meta = page_data(path)["meta"]
        assert meta["menu"] is None  # no link by default (the MVP page)
        write_viewer(
            [scenario],
            path,
            dt=0.4,
            capture_radius=0.5,
            subtitle="fixed wing · scramble",
            menu="index.html",
        )
        meta = page_data(path)["meta"]
        assert meta["subtitle"] == "fixed wing · scramble"
        assert meta["menu"] == "index.html"


def test_menu_names_the_scenarios_by_threat_type():
    assert [entry.kind for entry in ENTRIES] == ["Generic target", "Fixed wing"]


def test_menu_links_the_pages_that_exist_and_names_the_example_for_the_others():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        present, missing = ENTRIES
        (folder / present.page).write_text("<title>viewer</title>", encoding="utf-8")
        path = write_menu(folder)
        assert path == folder / MENU_FILE
        text = path.read_text(encoding="utf-8")
        assert f'href="{present.page}"' in text
        assert f'href="{missing.page}"' not in text
        assert missing.command in text
        assert "__MENU_CARDS__" not in text
