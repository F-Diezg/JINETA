"""The Iris flies the first course with the example's guidance (examples/iris_course):
everything of phase 8 together. Take-off from the floor with the autopilot landed,
the 13 spheres against walls, obstacles and gate frames, the gear against the floor
after take-off, the gates in order and the lap time.

The lap time is a regression value (measured when the example was written), not
an analytic one: it guards the whole chain against an unnoticed change.
"""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE, fly_course
from jineta.campo.course import load_course


@pytest.fixture(scope="module")
def lap():
    return fly_course(load_course(COURSE_FILE), speed=3.0)


def test_the_iris_finishes_the_first_course(lap):
    times, _, referee = lap
    assert referee.status == "finished"
    assert [p.gate for p in referee.passes] == [0, 1, 2, 3]
    assert referee.monitor.collision is None
    assert referee.lap_time == pytest.approx(8.81, abs=0.02)
    assert times[-2] < referee.lap_time <= times[-1]


def test_it_takes_off_from_the_floor_and_stays_inside_the_room(lap):
    _, states, referee = lap
    assert states[0, 2] == pytest.approx(-0.054)  # standing on its gear
    assert referee.monitor.airborne
    altitude = -states[:, 2]
    assert altitude.max() < 4.0 - 0.151  # never near the ceiling
    assert np.all((states[:, 0] > 0.0) & (states[:, 0] < 18.0))
    assert np.all((states[:, 1] > 0.0) & (states[:, 1] < 10.0))


def test_each_gate_is_crossed_through_its_opening(lap):
    _, _, referee = lap
    course = load_course(COURSE_FILE)
    for p in referee.passes:
        gate = course.gates[p.gate]
        n, u, v = gate.to_local(p.point)
        assert n == pytest.approx(0.0, abs=1e-12)
        assert gate.in_opening(u, v)
