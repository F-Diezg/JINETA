"""Tests for course files: the first course, the documented format, and every kind
of mistake a file can have (each one an error that says where)."""

import copy
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest

from jineta.campo import course as course_module
from jineta.campo.course import CourseError, course_from_dict, load_course
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.campo.indoor import Box, Cylinder
from jineta.core import quaternion
from jineta.core.simulation import run
from jineta.jinete.contact import resting_height
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR

FIRST_COURSE = Path(__file__).resolve().parent.parent / "courses" / "first_course.yaml"

BASE = {
    "name": "Test",
    "time_limit": 20,
    "room": {"size": [12.0, 8.0, 4.0]},
    "start": {"position": [1.0, 4.0], "yaw_deg": 0},
    "gates": [
        {
            "shape": "rectangle",
            "center": [5.0, 4.0, -1.5],
            "yaw_deg": 0,
            "width": 1.6,
            "height": 1.4,
            "thickness": 0.1,
        },
        {
            "shape": "circle",
            "center": [9.0, 4.0, -1.5],
            "yaw_deg": 0,
            "radius": 0.8,
            "thickness": 0.1,
        },
    ],
    "obstacles": [
        {"shape": "box", "center": [7.0, 1.0, -1.0], "size": [1.0, 1.0, 2.0]},
        {"shape": "cylinder", "base": [7.0, 7.0, 0.0], "radius": 0.3, "height": 4.0},
    ],
}


def changed(edit):
    data = copy.deepcopy(BASE)
    edit(data)
    return data


def test_the_first_course_loads():
    course = load_course(FIRST_COURSE)
    assert course.name == "First course"
    assert course.time_limit == 30.0
    np.testing.assert_array_equal(course.room.size, [18.0, 10.0, 4.0])
    assert [type(g) for g in course.gates] == [
        RectangularGate,
        CircularGate,
        RectangularGate,
        CircularGate,
    ]
    assert course.gates[2].yaw == pytest.approx(np.radians(30.0))
    assert [type(o) for o in course.obstacles] == [Cylinder, Box, Box]
    assert course.obstacles[1].yaw == pytest.approx(np.radians(20.0))
    np.testing.assert_array_equal(course.start, [1.5, 5.0])


def test_the_format_in_the_docs_is_a_valid_course():
    """The example in jineta/campo/course.py is read as a course file: the docs and
    the reader cannot drift apart."""
    doc = inspect.getdoc(course_module)
    example = textwrap.dedent(doc[doc.index("    name:") :])
    import yaml

    course = course_from_dict(yaml.safe_load(example))
    assert course.name == "First course" and len(course.gates) == 2


def test_optional_parts_can_be_left_out():
    no_obstacles = changed(lambda d: d.pop("obstacles"))
    assert course_from_dict(no_obstacles).obstacles == ()
    empty = changed(lambda d: d.update(obstacles=None))
    assert course_from_dict(empty).obstacles == ()
    # A box without yaw_deg is not turned.
    assert course_from_dict(BASE).obstacles[0].yaw == 0.0


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda d: d.update(gate=[]), "course: unknown key 'gate'"),
        (lambda d: d.pop("time_limit"), "course: missing key 'time_limit'"),
        (lambda d: d.update(name="  "), "name: must be a non-empty text"),
        (lambda d: d.update(time_limit=0), "time_limit: must be > 0"),
        (lambda d: d["gates"][0].update(widht=2), "gates[0]: unknown key 'widht'"),
        (lambda d: d["gates"][1].pop("radius"), "gates[1]: missing key 'radius'"),
        (lambda d: d["gates"][0].update(width="1.6"), "gates[0].width: must be a numb"),
        (lambda d: d["gates"][0].update(width=True), "gates[0].width: must be a numb"),
        (lambda d: d["gates"][0].update(width=-1), "gates[0].width: must be > 0"),
        (
            lambda d: d["gates"][0].update(center=[5, 4]),
            "gates[0].center: must be a li",
        ),
        (
            lambda d: d["gates"][0].update(shape="triangle"),
            "gates[0].shape: 'triangle'",
        ),
        (lambda d: d["gates"][1].pop("shape"), "gates[1]: missing key 'shape'"),
        (lambda d: d.update(gates=[]), "gates: must be a list of at least one gate"),
        (lambda d: d["room"].update(size=[12, -8, 4]), "room: every side of a room"),
        (lambda d: d["start"].update(position=[1, 4, 0]), "start.position: must be a"),
        (lambda d: d["obstacles"][0].update(shape="pyramid"), "obstacles[0].shape"),
        (lambda d: d["obstacles"][1].update(radius=0), "obstacles[1].radius: must be"),
        (lambda d: d.update(obstacles={"a": 1}), "obstacles: must be a list"),
        (lambda d: d.update(room=[12, 8, 4]), "room: must be a mapping"),
    ],
)
def test_mistakes_say_where_they_are(edit, message):
    with pytest.raises(CourseError, match=None) as error:
        course_from_dict(changed(edit))
    assert message in str(error.value)


@pytest.mark.parametrize(
    "edit, message",
    [
        # z positive is below the floor: the usual sign mistake, with a hint.
        (lambda d: d["gates"][0].update(center=[5.0, 4.0, 1.5]), "z points down"),
        (
            lambda d: d["gates"][0].update(center=[5.0, 0.5, -1.5]),
            "gates[0]: the frame",
        ),
        (
            lambda d: d["gates"][1].update(center=[9.0, 4.0, -3.5]),
            "gates[1]: the frame",
        ),
        (lambda d: d["obstacles"][0].update(center=[13, 1, -1]), "obstacles[0]: outsi"),
        (lambda d: d["start"].update(position=[13.0, 4.0]), "start.position: outside"),
        (lambda d: d["start"].update(position=[7.0, 1.2]), "inside obstacles[0]"),
        (lambda d: d["start"].update(position=[7.1, 6.9]), "inside obstacles[1]"),
    ],
)
def test_layout_mistakes(edit, message):
    with pytest.raises(CourseError) as error:
        course_from_dict(changed(edit))
    assert message in str(error.value)


def test_load_course_names_the_file(tmp_path):
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(CourseError, match=r"broken\.yaml: not valid YAML"):
        load_course(broken)
    typo = tmp_path / "typo.yaml"
    typo.write_text(FIRST_COURSE.read_text().replace("width:", "widht:", 1), "utf-8")
    with pytest.raises(CourseError, match=r"typo\.yaml: gates\[0\]: unknown key"):
        load_course(typo)


def test_resting_height_of_the_iris():
    """Its gear points are 0.055 m below the centre of mass and sink 1 mm."""
    assert resting_height(IRIS_LANDING_GEAR, IRIS.mass) == pytest.approx(0.054)


def test_start_state_stands_on_the_floor_facing_the_start_heading():
    course = course_from_dict(changed(lambda d: d["start"].update(yaw_deg=90)))
    y = course.start_state(0.054)
    np.testing.assert_allclose(y[0:3], [1.0, 4.0, -0.054])
    np.testing.assert_allclose(y[6:10], quaternion.from_euler(0.0, 0.0, np.pi / 2))
    np.testing.assert_array_equal(y[3:6], 0.0)
    np.testing.assert_array_equal(y[10:13], 0.0)


def test_the_course_builds_its_monitor_and_referee():
    course = course_from_dict(BASE)
    referee = course.referee([[0.0, 0.0, 0.0]], [0.2], [[0.0, 0.0, 0.1]])
    assert referee.gates == course.gates
    assert referee.monitor.gates == course.gates
    assert referee.monitor.obstacles == course.obstacles
    assert referee.time_limit == 20.0


def test_a_straight_lap_of_a_course_from_a_file():
    """A point with a 0.2 m sphere flying North at 2 m/s, 1.5 m up, through both
    gates: the lap ends at the second gate, at the exact moment."""
    course = course_from_dict(BASE)
    referee = course.referee([[0.0, 0.0, 0.0]], [0.2], [[0.0, 0.0, 0.1]])
    y0 = course.start_state(1.5)
    y0[3] = 2.0

    def straight_line(t, y):
        dy = np.zeros(13)
        dy[0:3] = y[3:6]
        return dy

    run(straight_line, y0, 0.01, 10.0, stop=referee)
    assert referee.status == "finished"
    assert referee.lap_time == pytest.approx((9.0 - 1.0) / 2.0, abs=1e-9)
