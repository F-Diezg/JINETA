"""Tests for the referee: gates in order, lap time, and how a lap ends.

The drone here is a point that moves in straight lines (known moments) with a
sphere of 0.2 m for collisions; the gates are 1 m square openings 2 m up, crossed
flying North.
"""

import numpy as np
import pytest

from jineta.campo.collision import CollisionMonitor
from jineta.campo.gates import RectangularGate
from jineta.campo.indoor import Box, Room
from jineta.campo.referee import Referee
from jineta.core.simulation import run

DT = 0.01
R = 0.2  # sphere radius
HALL = Room([20.0, 8.0, 4.0])


def gate_at(north, east=4.0):
    return RectangularGate([north, east, -2.0], 0.0, 1.0, 1.0, 0.1)


def referee(gates, room=HALL, obstacles=(), time_limit=20.0):
    monitor = CollisionMonitor(
        room, obstacles, [[0.0, 0.0, 0.0]], [R], [[0.0, 0.0, 0.1]], gates=gates
    )
    return Referee(gates, monitor, time_limit)


def state(north, east=4.0, v_north=2.0):
    y = np.zeros(13)
    y[0:3] = [north, east, -2.0]
    y[3] = v_north
    y[6] = 1.0
    return y


def straight_line(t, y):
    dy = np.zeros(13)
    dy[0:3] = y[3:6]
    return dy


def fly(judge, start, t_end=30.0, derivative=straight_line):
    return run(derivative, start, DT, t_end, stop=judge)


def test_three_gates_in_order_finish_the_lap():
    judge = referee([gate_at(5.0), gate_at(10.0), gate_at(15.0)])
    times, _ = fly(judge, state(1.003))
    assert judge.status == "finished"
    assert [p.gate for p in judge.passes] == [0, 1, 2]
    expected = [(g - 1.003) / 2.0 for g in (5.0, 10.0, 15.0)]
    np.testing.assert_allclose([p.time for p in judge.passes], expected, atol=1e-9)
    np.testing.assert_allclose(judge.passes[1].point, [10.0, 4.0, -2.0], atol=1e-9)
    assert judge.lap_time == pytest.approx(expected[-1], abs=1e-9)
    assert times[-2] < judge.lap_time <= times[-1]  # the run ends with the lap
    assert judge.next_gate is None


def test_a_later_gate_crossed_first_does_not_count():
    """Gate 0 is to the side of the path: gate 1, crossed first, does not count,
    and the lap ends at the time limit."""
    judge = referee([gate_at(5.0, east=6.0), gate_at(10.0)], time_limit=8.0)
    fly(judge, state(1.003))
    assert judge.passes == [] and judge.next_gate == 0
    assert judge.status == "timeout"
    assert judge.end_time == pytest.approx(8.0)
    assert judge.lap_time is None


def test_a_gate_crossed_backwards_does_not_count_but_can_be_flown_again():
    """Start just past gate 0 flying South (back through it), turn at t = 1 s and
    fly North through it: only the forward crossing counts."""

    def there_and_back(t, y):
        dy = np.zeros(13)
        dy[0] = -2.0 if t < 1.0 - 1e-9 else 2.0
        return dy

    judge = referee([gate_at(5.0), gate_at(10.0)])
    fly(judge, state(5.5, v_north=0.0), derivative=there_and_back)
    assert judge.status == "finished"
    assert [p.gate for p in judge.passes] == [0, 1]
    # 1.75 s by hand; RK4 blends the turn into the step that ends at 1 s (its last
    # stage already flies North), so the drone is a few millimetres off.
    assert judge.passes[0].time == pytest.approx(1.75, abs=DT)


def test_a_crash_ends_the_lap():
    crate = Box([7.5, 4.0, -2.0], [0.5, 1.0, 1.0])
    judge = referee([gate_at(5.0), gate_at(10.0)], obstacles=[crate])
    fly(judge, state(1.003))
    assert judge.status == "crashed"
    assert [p.gate for p in judge.passes] == [0]
    hit = judge.monitor.collision
    assert (hit.kind, hit.index) == ("obstacle", 0)
    # The crate's south face is at 7.25: touched when the centre is R short of it.
    assert judge.end_time == hit.time == pytest.approx((7.25 - R - 1.003) / 2.0)
    assert judge.lap_time is None


def test_finishing_a_few_milliseconds_before_a_crash_is_a_finish():
    """The last gate at 10 m and the north wall 4 mm (2 ms at 2 m/s) past the point
    where the sphere touches it: both inside the step from 2.00 to 2.01 s. The gate
    is crossed first (2.004 s), so the lap is finished."""
    start = 10.0 - 2.0 * 2.004
    room = Room([10.0 + R + 0.004, 8.0, 4.0])
    judge = referee([gate_at(10.0)], room=room)
    fly(judge, state(start))
    assert judge.status == "finished"
    assert judge.lap_time == pytest.approx(2.004, abs=1e-9)
    assert judge.monitor.collision.kind == "room"
    assert judge.monitor.collision.time == pytest.approx(2.006, abs=1e-9)


def test_crashing_a_few_milliseconds_before_a_gate_is_a_crash():
    """The north wall touched 2 ms before the centre would cross the last gate, in
    the same step: the crash came first, and the gate does not count."""
    start = 10.0 - 2.0 * 2.006
    room = Room([10.0 + R - 0.004, 8.0, 4.0])
    judge = referee([gate_at(10.0)], room=room)
    fly(judge, state(start))
    assert judge.status == "crashed"
    assert judge.passes == []
    assert judge.end_time == pytest.approx(2.004, abs=1e-9)


def test_reset_repeats_the_lap():
    """A lap that ends in a crash, run twice: reset clears the referee and its
    monitor, so the second lap is the first again."""
    crate = Box([7.5, 4.0, -2.0], [0.5, 1.0, 1.0])
    judge = referee([gate_at(5.0), gate_at(10.0)], obstacles=[crate])
    fly(judge, state(1.003))
    first = (judge.status, judge.end_time, [p.time for p in judge.passes])
    judge.reset()
    assert judge.status == "flying" and judge.passes == []
    assert judge.monitor.collision is None
    fly(judge, state(1.003))
    assert (judge.status, judge.end_time, [p.time for p in judge.passes]) == first


def test_referee_rejects_invalid_input():
    monitor = CollisionMonitor(Room([5, 5, 3]), (), [[0, 0, 0]], [R], [[0, 0, 0.1]])
    with pytest.raises(ValueError, match="gate"):
        Referee([], monitor, 10.0)
    with pytest.raises(ValueError, match="time_limit"):
        Referee([gate_at(2.0)], monitor, 0.0)
