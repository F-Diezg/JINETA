"""Tests for the batch collision monitor and the batch referee: every drone against
the reference (`campo.collision`, `campo.referee`) fed the same samples, laps that
crash, finish, time out or are still flying mixed in one batch, and each drone's
memory its own."""

import numpy as np
import pytest

import examples.iris_course as iris_course
from jineta.campo.collision import CollisionMonitor
from jineta.campo.course import load_course
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.campo.indoor import Box, Room
from jineta.campo.referee import Referee
from jineta.core import quaternion
from jineta.escuadron.collision import BatchCollisionMonitor
from jineta.escuadron.referee import (
    CRASHED,
    FINISHED,
    FLYING,
    STATUS_NAMES,
    BatchReferee,
    batch_referee,
    forward_crossings,
)
from jineta.jinete.contact import resting_height
from jineta.jinete.profiles import IRIS, IRIS_HITBOX, IRIS_LANDING_GEAR

DT = 0.01
COURSE = load_course(iris_course.COURSE_FILE)
REST = resting_height(IRIS_LANDING_GEAR, IRIS.mass)
HITBOX = (IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points)


def flights(rng: np.random.Generator, n: int, samples: int) -> np.ndarray:
    """Sampled flights (samples, n, 13) in the first course's room: from random
    places at random speeds up to ~15 m/s with a random acceleration, any heading,
    tilted up to ~25 degrees, turning; a quarter of them start standing on the
    floor and climb, then come back down. Many end against a wall, an obstacle, a
    gate frame or the floor, some pass close to them, some fly through gates."""
    t = np.arange(samples) * DT
    states = np.zeros((samples, n, 13))
    length, width, height = COURSE.room.size
    start = rng.uniform(
        [1.0, 1.0, -height + 0.6], [length - 1.0, width - 1.0, -0.6], (n, 3)
    )
    velocity = rng.normal(size=(n, 3)) * np.array([6.0, 6.0, 1.0])
    acceleration = rng.normal(size=(n, 3)) * 2.0
    standing = np.arange(n) % 4 == 0
    start[standing, 2] = -REST
    velocity[standing] = [0.0, 0.0, -1.5]
    acceleration[standing] = [0.5, 0.0, 1.5]  # up, then back down to the floor
    states[:, :, 0:3] = (
        start + velocity * t[:, None, None] + 0.5 * acceleration * t[:, None, None] ** 2
    )
    states[:, :, 3:6] = velocity + acceleration * t[:, None, None]
    roll_pitch = rng.normal(size=(n, 2)) * 0.25
    roll_pitch[standing] = 0.0
    yaw0 = rng.uniform(-np.pi, np.pi, n)
    yaw_rate = rng.normal(size=n) * 2.0
    for k in range(samples):
        for i in range(n):
            states[k, i, 6:10] = quaternion.from_euler(
                *roll_pitch[i], yaw0[i] + yaw_rate[i] * t[k]
            )
    return states


def grazes(samples: int) -> np.ndarray:
    """Flights (samples, 12, 13) North at 20 m/s past the pillar of the first
    course (0.3 m radius at North 11.5, East 6.6), its right side grazing it from
    5 micrometres to 2 cm deep, at several phases of the samples: most of these
    contacts fall between two samples that both touch nothing."""
    side = IRIS_HITBOX.extent([0, 1, 0])
    depths = np.repeat([5e-6, 1e-3, 2e-2], 4)
    phases = np.tile([0.0, 0.25, 0.5, 0.75], 3)
    states = np.zeros((samples, 12, 13))
    t = np.arange(samples) * DT
    states[:, :, 0] = 8.0 + 0.2 * phases + 20.0 * t[:, None]
    states[:, :, 1] = 6.6 - 0.3 - side + depths
    states[:, :, 2] = -1.5
    states[:, :, 3] = 20.0
    states[:, :, 6] = 1.0
    return states


def reference_monitor() -> CollisionMonitor:
    return COURSE.monitor(*HITBOX)


def batch_monitor(n: int) -> BatchCollisionMonitor:
    return BatchCollisionMonitor(
        COURSE.room, COURSE.obstacles, *HITBOX, n, gates=COURSE.gates
    )


# ---- collisions ------------------------------------------------------------------


def test_collisions_match_reference_for_every_drone():
    rng = np.random.default_rng(0)
    samples = 160
    standing = np.zeros((samples, 3, 13))  # three drones on their gear all along
    standing[:, :, 0:2] = [[2.0, 2.0], [3.0, 8.0], [16.0, 2.0]]
    standing[:, :, 2] = -REST
    standing[:, :, 6] = 1.0
    history = np.concatenate(
        [flights(rng, 80, samples), standing, grazes(samples)], axis=1
    )
    n = history.shape[1]
    batch = batch_monitor(n)
    for k in range(1, samples):  # run never shows `stop` the starting state
        batch(k * DT, history[k])
    kinds = set()
    for i in range(n):
        reference = reference_monitor()
        for k in range(1, samples):
            if reference(k * DT, history[k, i]):
                break
        expected = reference.collision
        got = batch.collision(i)
        assert batch.airborne[i] == reference.airborne
        if expected is None:
            assert got is None
            continue
        kinds.add(expected.kind)
        assert (got.kind, got.index) == (expected.kind, expected.index)
        assert got.time == pytest.approx(expected.time, abs=1e-9)
        assert np.allclose(got.position, expected.position, atol=1e-9)
    assert kinds == {"room", "obstacle", "gate", "floor"}
    assert 0 < np.sum(~batch.collided) < n  # some fly on to the end
    assert not np.any(batch.collided[80:83])  # standing on the floor is no crash
    assert np.all(batch.check[-12:] == 1)  # every graze found, on the pillar
    # and most of them only between samples: the sample after the contact is free
    after = np.ceil(batch.time[-12:] / DT - 1e-9).astype(int)
    free_after = [
        np.all(batch.margins(history[k, 83 + j : 84 + j])[0, :-1] >= 0.0)
        for j, k in enumerate(after)
    ]
    assert sum(free_after) >= 6


def test_margins_match_reference():
    rng = np.random.default_rng(1)
    history = flights(rng, 50, 2)[1]
    margins = batch_monitor(50).margins(history)
    reference = reference_monitor()
    for state, row in zip(history, margins, strict=True):
        assert np.allclose(row, reference.margins(state), rtol=0, atol=1e-12)


def test_a_drone_gives_the_same_collision_alone_or_in_a_batch():
    rng = np.random.default_rng(0)
    n, samples = 80, 160
    history = flights(rng, n, samples)
    everyone = batch_monitor(n)
    for k in range(1, samples):
        everyone(k * DT, history[k])
    alone = batch_monitor(1)
    for i in range(0, n, 9):
        alone.reset()
        for k in range(1, samples):
            alone(k * DT, history[k, i : i + 1])
        assert alone.collided[0] == everyone.collided[i]
        assert alone.airborne[0] == everyone.airborne[i]
        if alone.collided[0]:
            assert alone.check[0] == everyone.check[i]
            assert alone.time[0] == everyone.time[i]
            assert np.array_equal(alone.position[0], everyone.position[i])


def test_only_the_chosen_drones_are_checked_and_reset():
    rng = np.random.default_rng(2)
    history = flights(rng, 4, 3)
    monitor = batch_monitor(4)
    chosen = np.array([True, False, True, False])
    monitor(DT, history[1], chosen)
    assert np.array_equal(monitor._has_previous, chosen)
    monitor.collided[:] = True
    monitor.reset([1, 2])
    assert np.array_equal(monitor.collided, [True, False, False, True])
    assert np.array_equal(monitor._has_previous, [True, False, False, False])


# ---- laps ------------------------------------------------------------------------


def reference_lap() -> np.ndarray:
    _, states, referee = iris_course.fly_course(COURSE)
    assert referee.status == "finished"
    return states


def laps(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Sampled laps (samples, n, 13): the reference lap of the first course, the
    same lap moved sideways by a few centimetres to decimetres (some clip a frame or
    miss a gate), flown backwards, and random flights; long enough for the
    reference lap to finish."""
    lap = reference_lap()
    samples = len(lap) + 20
    random = flights(rng, 30, samples)
    padded = np.concatenate([lap, np.repeat(lap[-1:], 20, axis=0)])
    shifted = []
    for offset in (0.05, -0.2, 0.4, -0.6):
        moved = padded.copy()
        moved[:, 1] += offset
        moved[:, 2] -= abs(offset) / 2.0
        shifted.append(moved)
    backwards = padded[::-1].copy()
    all_laps = np.concatenate(
        [padded[:, None], np.stack(shifted, axis=1), backwards[:, None], random], axis=1
    )
    return all_laps, samples


def judge_reference(history: np.ndarray, i: int, time_limit: float) -> Referee:
    referee = Referee(COURSE.gates, reference_monitor(), time_limit)
    for k in range(1, len(history)):
        if referee(k * DT, history[k, i]):
            break
    return referee


def assert_same_lap(batch: BatchReferee, i: int, reference: Referee) -> None:
    assert STATUS_NAMES[batch.status[i]] == reference.status
    if reference.end_time is None:
        assert np.isnan(batch.end_time[i])
    else:
        assert batch.end_time[i] == pytest.approx(reference.end_time, abs=1e-9)
    passes = batch.passes(i)
    assert [p.gate for p in passes] == [p.gate for p in reference.passes]
    for got, expected in zip(passes, reference.passes, strict=True):
        assert got.time == pytest.approx(expected.time, abs=1e-9)
        assert np.allclose(got.point, expected.point, atol=1e-9)


@pytest.mark.parametrize("time_limit", [COURSE.time_limit, 3.0])
def test_laps_match_reference_for_every_drone(time_limit):
    """The reference lap finishes at its lap time; the moved and backward laps and
    the random flights crash, time out or are still flying: every drone's status,
    end, passes (gate, moment, point) equal the reference referee's."""
    rng = np.random.default_rng(3)
    history, samples = laps(rng)
    n = history.shape[1]
    monitor = batch_monitor(n)
    batch = BatchReferee(COURSE.gates, monitor, time_limit)
    for k in range(1, samples):
        batch(k * DT, history[k])
    for i in range(n):
        assert_same_lap(batch, i, judge_reference(history, i, time_limit))
    statuses = {STATUS_NAMES[s] for s in batch.status}
    if time_limit == COURSE.time_limit:
        assert batch.status[0] == FINISHED
        assert batch.lap_time[0] == pytest.approx(8.8145, abs=1e-4)
        assert {"finished", "crashed", "flying"} <= statuses
    else:
        assert {"crashed", "timeout"} <= statuses
    assert np.any(batch.passed[1:] > 0)  # gates passed by other drones too


def test_each_drone_on_its_own_clock_and_reset():
    """Drone 1 waits 2 s and then starts its lap (reset, its clock from zero):
    both drones finish the same lap at the same lap time, drone 1 later."""
    lap = reference_lap()
    referee = batch_referee(COURSE, *HITBOX, 2)
    wait = 200
    for k in range(1, len(lap) + wait):
        states = np.stack([lap[min(k, len(lap) - 1)], lap[max(k - wait, 0)]])
        clocks = np.array([k * DT, (k - wait) * DT])
        if k == wait:
            referee.reset([1])
            continue
        referee(clocks, states)
    assert np.array_equal(referee.status, [FINISHED, FINISHED])
    assert referee.lap_time[1] == pytest.approx(referee.lap_time[0], abs=1e-12)
    assert np.array_equal(referee.passed, [4, 4])


def test_ended_laps_are_not_judged_again():
    lap = reference_lap()
    referee = batch_referee(COURSE, *HITBOX, 1)
    for k in range(1, len(lap)):
        referee(k * DT, lap[k : k + 1])
    assert referee.status[0] == FINISHED
    end = referee.end_time[0]
    # flying on into a wall changes nothing
    wall = lap[-1:].copy()
    wall[0, 0] = COURSE.room.size[0] + 1.0
    assert referee(len(lap) * DT, wall)[0]
    assert referee.status[0] == FINISHED and referee.end_time[0] == end
    assert not referee.monitor.collided[0]
    referee.reset()
    assert referee.status[0] == FLYING and referee.passed[0] == 0
    assert np.all(np.isnan(referee.pass_time))


@pytest.mark.parametrize(
    "gate",
    [
        RectangularGate([5.0, 4.0, -2.0], np.radians(30.0), 1.6, 1.0, 0.1),
        CircularGate([5.0, 4.0, -2.0], np.radians(-40.0), 0.8, 0.1),
    ],
    ids=["rectangle", "circle"],
)
def test_forward_crossings_match_the_gates_own(gate):
    """3000 random moves around a gate, many through it, some past its edges, some
    backwards: crossed, fraction and point as the gate's own forward crossings."""
    rng = np.random.default_rng(4)
    heading = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0.0])
    right = np.array([-np.sin(gate.yaw), np.cos(gate.yaw), 0.0])
    down = np.array([0.0, 0.0, 1.0])

    def around(n_low, n_high):
        n = rng.uniform(n_low, n_high, size=(3000, 1))
        u, v = rng.normal(size=(2, 3000, 1)) * 0.7
        return gate.center + n * heading + u * right + v * down

    start, end = around(-0.5, 0.1), around(-0.1, 0.5)
    crossed, fraction, point = forward_crossings(gate, start, end)
    for i in range(3000):
        expected = gate.crossing(start[i], end[i])
        if expected is None or not expected.forward:
            assert not crossed[i]
            continue
        assert crossed[i]
        assert fraction[i] == pytest.approx(expected.fraction, rel=1e-14, abs=1e-15)
        assert np.allclose(point[i], expected.point, rtol=0, atol=1e-14)
    assert 300 < np.sum(crossed) < 2700


SMALL = 0.2  # the point drone's sphere [m]


def straight_flights(starts: np.ndarray, v_north: float, samples: int) -> np.ndarray:
    """Point drones flying North at v_north from `starts` (n, 3), level: (samples,
    n, 13)."""
    states = np.zeros((samples, len(starts), 13))
    t = np.arange(samples) * DT
    states[:, :, 0:3] = starts + np.outer(t, [v_north, 0.0, 0.0])[:, None, :]
    states[:, :, 3] = v_north
    states[:, :, 6] = 1.0
    return states


def test_events_in_the_same_step_and_a_crashed_lap_reset():
    """Point drones (one 0.2 m sphere), gate 1 m square at North 10, flying North at
    2 m/s, in one batch: A crosses the gate 2 ms before touching the north wall
    (finished); B touches a box 2 ms before it would cross the gate (crashed, the
    gate does not count); C hits a crate before the gate (crashed), then is reset
    and flies a free lane (finished). Each as the reference referee says."""
    gate = RectangularGate([10.0, 4.0, -2.0], 0.0, 1.0, 1.0, 0.1)
    room = Room([10.0 + SMALL + 0.004, 8.0, 4.0])
    # B touches the south face of a box (North 10.196, East 4.1 to 4.95) when its
    # centre is R short of it: 4 mm before the gate. A (East 3.85) passes beside.
    box = Box([10.0 - 0.004 + SMALL + 0.25, 4.525, -2.0], [0.5, 0.85, 1.0])
    crate = Box([7.0, 1.0, -2.0], [0.5, 1.0, 1.0])
    obstacles = [box, crate]
    starts = np.array(
        [
            [10.0 - 2.0 * 2.004, 3.85, -2.0],  # A
            [10.0 - 2.0 * 2.006, 4.25, -2.0],  # B
            [5.0, 1.0, -2.0],  # C, at the crate
        ]
    )
    history = straight_flights(starts, 2.0, 260)

    def reference_referee():
        monitor = CollisionMonitor(
            room, obstacles, [[0.0, 0.0, 0.0]], [SMALL], [[0.0, 0.0, 0.1]], gates=[gate]
        )
        return Referee([gate], monitor, 20.0)

    monitor = BatchCollisionMonitor(
        room, obstacles, [[0.0, 0.0, 0.0]], [SMALL], [[0.0, 0.0, 0.1]], 3, gates=[gate]
    )
    batch = BatchReferee([gate], monitor, 20.0)
    for k in range(1, 260):
        batch(k * DT, history[k])
    for i in range(3):
        reference = reference_referee()
        for k in range(1, 260):
            if reference(k * DT, history[k, i]):
                break
        assert_same_lap(batch, i, reference)
    assert np.array_equal(batch.status, [FINISHED, CRASHED, CRASHED])
    assert batch.lap_time[0] == pytest.approx(2.004, abs=1e-9)
    assert batch.end_time[1] == pytest.approx(2.004, abs=1e-9)
    assert batch.passed[1] == 0

    # C again, from a lane through the gate: reset forgets the crash.
    batch.reset([2])
    lane = straight_flights(np.array([[6.0, 4.0, -2.0]]), 2.0, 260)
    for k in range(1, 260):
        states = history[k].copy()
        states[2] = lane[k, 0]
        batch(k * DT, states)
    assert batch.status[2] == FINISHED
    assert batch.lap_time[2] == pytest.approx(2.0, abs=1e-9)
    assert np.array_equal(batch.status[:2], [FINISHED, CRASHED])  # untouched


def test_rejects_invalid_input():
    monitor = batch_monitor(2)
    with pytest.raises(ValueError, match="gate"):
        BatchReferee((), monitor, 10.0)
    with pytest.raises(ValueError, match="time_limit"):
        BatchReferee(COURSE.gates, monitor, 0.0)
    with pytest.raises(ValueError, match="n"):
        BatchCollisionMonitor(COURSE.room, (), *HITBOX, 0)
