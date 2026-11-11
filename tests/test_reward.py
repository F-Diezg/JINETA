"""Tests for the reward and the end of an episode: sampled flights judged by the
batch referee, with the reward of every step checked against distances by hand."""

import itertools

import numpy as np
import pytest

from examples.batch_course import course_flight
from examples.iris_course import COURSE_FILE
from jineta.campo.course import course_from_dict, load_course
from jineta.escuadron.referee import (
    CRASHED,
    FINISHED,
    FLYING,
    TIMEOUT,
    batch_referee,
)
from jineta.jinete.profiles import IRIS_HITBOX, IRIS_LANDING_GEAR
from jineta.picadero.reward import Reward, RewardTerms, RewardWeights, episode_end

DT = 0.01

# Three gates in a row along y = 5 m, 1.5 m up, every 5 m North.
STRAIGHT = course_from_dict(
    {
        "name": "Straight",
        "time_limit": 10,
        "room": {"size": [20.0, 10.0, 4.0]},
        "start": {"position": [1.0, 5.0], "yaw_deg": 0},
        "gates": [
            {
                "shape": "rectangle",
                "center": [x, 5.0, -1.5],
                "yaw_deg": 0,
                "width": 1.6,
                "height": 1.4,
                "thickness": 0.1,
            }
            for x in (5.0, 10.0, 15.0)
        ],
    }
)


def level(positions: np.ndarray) -> np.ndarray:
    """States (n, 13) at these positions, level, nose North, still."""
    states = np.zeros((len(positions), 13))
    states[:, 0:3] = positions
    states[:, 6] = 1.0
    return states


def judge(course, paths, weights=None):
    """Sampled paths (steps + 1, n, 3), one sample per step, judged by the batch
    referee; the reward terms of every step (steps, n) and the final statuses."""
    n = paths.shape[1]
    referee = batch_referee(
        course, IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points, n
    )
    reward = Reward(course.gates, weights or RewardWeights())
    referee(0.0, level(paths[0]))
    progress, gate, crash, total = [], [], [], []
    for k in range(1, len(paths)):
        active = ~referee.ended
        target, passed = referee.next_gate.copy(), referee.passed.copy()
        referee(k * DT, level(paths[k]))
        crashed = active & (referee.status == CRASHED)
        args = (
            paths[k - 1],
            paths[k],
            target,
            referee.passed - passed,
            crashed,
            active,
        )
        terms = reward.terms(*args)
        progress.append(terms.progress)
        gate.append(terms.gate)
        crash.append(terms.crash)
        total.append(reward(*args))
    as_array = [np.array(x) for x in (progress, gate, crash, total)]
    return RewardTerms(*as_array[:3]), as_array[3], referee


def straight_path(start: float, end: float, step: float) -> np.ndarray:
    """Samples along the gate line y = 5, z = -1.5, from x = start, `step` apart."""
    x = start + step * np.arange(round((end - start) / step) + 1)
    return np.column_stack([x, np.full_like(x, 5.0), np.full_like(x, -1.5)])[:, None]


def test_a_straight_lap_by_hand():
    """From x = 1 to the last gate at x = 15 in steps of 0.1 m: 4 m to gate 0, then
    5 m to each of the others, every step 0.1 m of progress, crossing steps
    included (no jump), and one bonus per gate."""
    terms, total, referee = judge(STRAIGHT, straight_path(1.0, 15.0, 0.1))
    assert referee.status[0] == FINISHED
    np.testing.assert_allclose(terms.progress[:, 0], 0.1, atol=1e-9)
    assert terms.progress.sum() == pytest.approx(4.0 + 5.0 + 5.0, abs=1e-9)
    assert terms.gate.sum() == 3
    np.testing.assert_array_equal(np.flatnonzero(terms.gate[:, 0]), [39, 89, 139])
    assert terms.crash.sum() == 0
    assert total.sum() == pytest.approx(14.0 + 3 * 1.0, abs=1e-9)


def test_moving_away_is_negative_progress_and_a_backward_crossing_pays_nothing():
    """Through gate 0 backwards (from x = 6 to 4): no bonus, and the distance to
    gate 0's centre first shrinks (1 m), then grows again (1 m): progress 0."""
    terms, _, referee = judge(STRAIGHT, straight_path(6.0, 4.0, -0.1))
    assert referee.status[0] == FLYING
    assert terms.gate.sum() == 0
    np.testing.assert_allclose(terms.progress[:10, 0], 0.1, atol=1e-9)
    np.testing.assert_allclose(terms.progress[10:, 0], -0.1, atol=1e-9)
    assert terms.progress.sum() == pytest.approx(0.0, abs=1e-9)


def test_a_crash_pays_the_penalty_once_and_then_nothing():
    """Straight at the east wall (y = 10) from y = 8: the Iris's spheres touch it
    and the lap ends; later samples are not rewarded."""
    y = 8.0 + 0.05 * np.arange(60)
    path = np.column_stack([np.full_like(y, 3.0), y, np.full_like(y, -1.5)])[:, None]
    terms, total, referee = judge(STRAIGHT, path)
    assert referee.status[0] == CRASHED
    assert terms.crash.sum() == 1
    k = int(np.flatnonzero(terms.crash[:, 0])[0])
    assert np.all(terms.progress[k + 1 :] == 0.0) and np.all(total[k + 1 :] == 0.0)
    assert total[k, 0] == pytest.approx(terms.progress[k, 0] - 5.0)


def test_weights_scale_the_terms():
    weights = RewardWeights(progress=2.0, gate=10.0, crash=0.0)
    _, total, _ = judge(STRAIGHT, straight_path(1.0, 15.0, 0.1), weights)
    assert total.sum() == pytest.approx(2.0 * 14.0 + 10.0 * 3, abs=1e-9)


def test_the_progress_of_a_real_lap_adds_up_gate_by_gate():
    """The first course flown by the gate guidance (batch engine): the progress
    adds up, gate by gate, to the distance from where the drone started flying to
    that gate to where it was just after crossing it."""
    course = load_course(COURSE_FILE)
    flight, guidance = course_flight(course, np.array([3.0]))
    reward = Reward(course.gates)
    positions, targets, progress, bonus = [flight.states[0, 0:3].copy()], [], [], 0.0
    while not flight.referee.ended[0]:
        target, passed = flight.referee.next_gate.copy(), flight.referee.passed.copy()
        before = flight.states[:, 0:3].copy()
        flight.step(guidance)
        terms = reward.terms(
            before,
            flight.states[:, 0:3],
            target,
            flight.referee.passed - passed,
            flight.referee.status == CRASHED,
            np.ones(1, dtype=bool),
        )
        positions.append(flight.states[0, 0:3].copy())
        targets.append(target[0])
        progress.append(terms.progress[0])
        bonus += terms.gate[0]
    assert flight.referee.status[0] == FINISHED and bonus == 4
    targets = np.array(targets)
    expected = 0.0
    for gate in range(4):
        steps = np.flatnonzero(targets == gate)
        center = course.gates[gate].center
        first, last = positions[steps[0]], positions[steps[-1] + 1]
        expected += np.linalg.norm(first - center) - np.linalg.norm(last - center)
    assert sum(progress) == pytest.approx(expected, abs=1e-9)
    # About the length of the course: from the start to the last gate's centre,
    # through every gate centre.
    points = [positions[0], *(gate.center for gate in course.gates)]
    length = sum(np.linalg.norm(b - a) for a, b in itertools.pairwise(points))
    assert sum(progress) == pytest.approx(length, rel=0.05)


def test_many_drones_give_the_same_as_one_by_one():
    rng = np.random.default_rng(4)
    n = 6
    paths = np.cumsum(rng.normal(scale=0.08, size=(300, n, 3)), axis=0)
    paths += np.array([2.0, 5.0, -1.5])
    paths[:, :, 0] += np.linspace(0.0, 12.0, 300)[:, None]  # heading North
    together, total, _ = judge(STRAIGHT, paths)
    for i in range(n):
        alone, alone_total, _ = judge(STRAIGHT, paths[:, i : i + 1])
        for a, b in zip(alone, together, strict=True):
            np.testing.assert_array_equal(a[:, 0], b[:, i])
        np.testing.assert_array_equal(alone_total[:, 0], total[:, i])


def test_episode_end():
    terminated, truncated = episode_end(np.array([FLYING, FINISHED, CRASHED, TIMEOUT]))
    np.testing.assert_array_equal(terminated, [False, True, True, False])
    np.testing.assert_array_equal(truncated, [False, False, False, True])


@pytest.mark.parametrize(
    "kwargs", [dict(progress=-1.0), dict(gate=np.inf), dict(crash=np.nan)]
)
def test_weights_reject_bad_values(kwargs):
    with pytest.raises(ValueError):
        RewardWeights(**kwargs)


def test_reward_needs_a_gate():
    with pytest.raises(ValueError, match="gate"):
        Reward(())


def test_drones_not_flying_at_the_start_of_the_step_get_nothing():
    """A drone whose lap had already ended (still marked crashed, crossing a gate,
    moving towards its target) is not rewarded; its target may even be -1."""
    reward = Reward(STRAIGHT.gates)
    before = np.array([[1.0, 5.0, -1.5], [1.0, 5.0, -1.5]])
    after = before + np.array([0.5, 0.0, 0.0])
    terms = reward.terms(before, after, [0, -1], [1, 1], [True, True], [True, False])
    np.testing.assert_allclose(terms.progress, [0.5, 0.0], atol=1e-12)
    np.testing.assert_array_equal(terms.gate, [1.0, 0.0])
    np.testing.assert_array_equal(terms.crash, [1.0, 0.0])
    np.testing.assert_allclose(
        reward(before, after, [0, -1], [1, 1], [True, True], [True, False]),
        [0.5 + 1.0 - 5.0, 0.0],
    )
