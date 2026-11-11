"""Tests for the action and the observation of the training environment, against
geometry worked out by hand on the first course."""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.core import quaternion
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.picadero.spaces import (
    ActionLimits,
    BoxSpace,
    Observer,
    autopilot_envelope,
    opening_points,
)

COURSE = load_course(COURSE_FILE)
LIMITS = ActionLimits(2.0, 22.0, [3.5, 3.5, 1.5])


def state(position, velocity=(0.0, 0.0, 0.0), yaw_deg=0.0, rates=(0.0, 0.0, 0.0)):
    y = np.zeros(13)
    y[0:3] = position
    y[3:6] = velocity
    y[6:10] = quaternion.from_euler(0.0, 0.0, np.radians(yaw_deg))
    y[10:13] = rates
    return y


# ---- spaces ------------------------------------------------------------------------


def test_box_space_shape_limits_and_contains():
    space = BoxSpace([-1.0, 0.0], [1.0, 2.0])
    assert space.shape == (2,) and space.dtype == np.float32
    assert space.low.dtype == np.float32
    assert space.contains([0.0, 2.0]) and space.contains([[-1.0, 0.0], [1.0, 1.0]])
    assert not space.contains([0.0, 2.1])
    assert not space.contains([0.0, 1.0, 0.0])
    assert not space.contains(np.zeros((2, 2, 2)))


def test_box_space_samples_inside_and_repeats_with_the_seed():
    space = BoxSpace([-1.0, 0.0, 5.0], [1.0, 2.0, 5.0])
    a = space.sample(np.random.default_rng(3), 1000)
    b = space.sample(np.random.default_rng(3), 1000)
    assert a.shape == (1000, 3) and a.dtype == np.float32
    assert space.contains(a)
    np.testing.assert_array_equal(a, b)
    assert space.sample(np.random.default_rng(0)).shape == (3,)
    with pytest.raises(ValueError, match="bounded"):
        BoxSpace([-np.inf], [0.0]).sample(np.random.default_rng(0))


@pytest.mark.parametrize(
    "low, high", [([1.0], [0.0]), ([0.0, 0.0], [1.0]), ([], []), ([np.nan], [1.0])]
)
def test_box_space_rejects_bad_limits(low, high):
    with pytest.raises(ValueError):
        BoxSpace(low, high)


def test_box_space_is_read_only():
    space = BoxSpace([0.0], [1.0])
    with pytest.raises(ValueError):
        space.low[0] = -1.0


# ---- action ------------------------------------------------------------------------


def test_actions_map_onto_the_envelope_by_hand():
    actions = np.array(
        [
            [-1.0, 0.0, 0.0, 0.0],  # least thrust
            [1.0, 0.0, 0.0, 0.0],  # most thrust
            [0.0, 1.0, -1.0, 0.5],  # mid thrust, full roll right, full pitch down
            [-0.5, 0.2, 0.0, -1.0],
        ]
    )
    thrust, rates = LIMITS.commands(actions)
    np.testing.assert_allclose(thrust, [2.0, 22.0, 12.0, 7.0], atol=1e-12)
    np.testing.assert_allclose(
        rates,
        [[0, 0, 0], [0, 0, 0], [3.5, -3.5, 0.75], [0.7, 0.0, -1.5]],
        atol=1e-12,
    )


def test_actions_outside_the_range_are_clipped():
    thrust, rates = LIMITS.commands([[3.0, -7.0, 1.5, 0.0]])
    assert thrust[0] == 22.0
    np.testing.assert_array_equal(rates[0], [-3.5, 3.5, 0.0])


def test_actions_for_commands_are_the_inverse_inside_the_envelope():
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, size=(500, 4))
    back = LIMITS.actions(*LIMITS.commands(actions))
    np.testing.assert_allclose(back, actions, atol=1e-12)
    # Outside the envelope the action saturates.
    np.testing.assert_array_equal(
        LIMITS.actions([30.0, 0.0], [[9.0, 0.0, 0.0], [0.0, 0.0, -9.0]]),
        [[1.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, -1.0]],
    )


def test_the_iris_autopilot_envelope():
    limits = autopilot_envelope(IRIS, IRIS_GAINS)
    total = 4 * IRIS.max_rotor_thrust  # 4 x 7.07 N
    assert IRIS.max_rotor_thrust == pytest.approx(7.07, abs=0.01)
    assert limits.thrust_min == pytest.approx(0.1 * total)
    assert limits.thrust_max == pytest.approx(0.9 * total)
    np.testing.assert_array_equal(limits.max_rate, [3.5, 3.5, 1.5])
    assert limits.space.shape == (4,)
    assert limits.space.contains([[-1, -1, -1, -1], [1, 1, 1, 1]])
    # Hovering (thrust = weight) is inside the envelope, a little above the middle.
    weight = IRIS.mass * 9.80665
    hover = limits.actions([weight], [[0.0, 0.0, 0.0]])[0, 0]
    assert 0.0 < hover < 0.3


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(thrust_min=-1.0, thrust_max=1.0, max_rate=[1, 1, 1]),
        dict(thrust_min=5.0, thrust_max=5.0, max_rate=[1, 1, 1]),
        dict(thrust_min=0.0, thrust_max=1.0, max_rate=[1, 0, 1]),
        dict(thrust_min=0.0, thrust_max=1.0, max_rate=[1, 1]),
    ],
)
def test_action_limits_reject_bad_envelopes(kwargs):
    with pytest.raises(ValueError):
        ActionLimits(**kwargs)


# ---- the points of an opening -------------------------------------------------------


def test_rectangle_corners_by_hand():
    """Gate 0: 1.6 x 1.4 m at (5, 5, -1.5), flown North: u is East, v is down."""
    expected = [
        [5.0, 4.2, -2.2],  # top left: West and up
        [5.0, 5.8, -2.2],
        [5.0, 5.8, -0.8],
        [5.0, 4.2, -0.8],
    ]
    np.testing.assert_allclose(opening_points(COURSE.gates[0]), expected, atol=1e-12)


def test_turned_rectangle_corners_by_hand():
    """Gate 2: yaw 30°, so u = (-sin 30°, cos 30°, 0) = (-0.5, 0.866, 0)."""
    c30 = np.sqrt(3.0) / 2.0
    top_left = [14.0 + 0.8 * 0.5, 7.0 - 0.8 * c30, -1.5 - 0.7]
    np.testing.assert_allclose(opening_points(COURSE.gates[2])[0], top_left, atol=1e-12)


def test_circle_points_are_the_inscribed_square_by_hand():
    """Gate 1: radius 0.8 at (9, 3, -2), flown North: 0.8 / sqrt(2) each way."""
    h = 0.8 / np.sqrt(2.0)
    expected = [
        [9, 3 - h, -2 - h],
        [9, 3 + h, -2 - h],
        [9, 3 + h, -2 + h],
        [9, 3 - h, -2 + h],
    ]
    np.testing.assert_allclose(opening_points(COURSE.gates[1]), expected, atol=1e-12)


@pytest.mark.parametrize("index", range(4))
def test_every_point_is_on_the_edge_of_its_opening(index):
    gate = COURSE.gates[index]
    local = gate.to_local(opening_points(gate))
    np.testing.assert_allclose(local[:, 0], 0.0, atol=1e-12)  # on the gate's plane
    # On the edge: a hair inside it is in the opening (rounding may put the point
    # itself a hair outside).
    assert all(gate.in_opening(u * (1 - 1e-9), v * (1 - 1e-9)) for u, v in local[:, 1:])
    if isinstance(gate, RectangularGate):
        np.testing.assert_allclose(np.abs(local[:, 1]), gate.width / 2, atol=1e-12)
        np.testing.assert_allclose(np.abs(local[:, 2]), gate.height / 2, atol=1e-12)
    else:
        assert isinstance(gate, CircularGate)
        np.testing.assert_allclose(np.hypot(local[:, 1], local[:, 2]), gate.radius)
    # Top left, top right, bottom right, bottom left (u right, v down).
    np.testing.assert_array_equal(np.sign(local[:, 1]), [-1, 1, 1, -1])
    np.testing.assert_array_equal(np.sign(local[:, 2]), [-1, -1, 1, 1])


# ---- the observation ----------------------------------------------------------------


def observe(states, next_gate, previous=None, lookahead=2):
    observer = Observer(COURSE.gates, lookahead)
    states = np.atleast_2d(states)
    if previous is None:
        previous = np.zeros((len(states), 4))
    return observer(states, np.atleast_1d(next_gate), previous)


def test_layout_and_size():
    observer = Observer(COURSE.gates)
    assert observer.size == 3 + 9 + 3 + 24 + 4 == 43
    assert Observer(COURSE.gates, lookahead=3).size == 55
    assert observer.space.shape == (43,)


def test_level_heading_north_by_hand():
    """Level, nose North, 1 m up at the start, flying North at 2 m/s: body axes are
    NED, so the gate corners are just their offsets from the drone."""
    y = state([1.5, 5.0, -1.0], velocity=[2.0, 0.0, 0.0], rates=[0.1, -0.2, 0.3])
    obs = observe(y, 0, previous=[[0.1, 0.2, 0.3, 0.4]])[0]
    np.testing.assert_allclose(obs[0:3], [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(obs[3:12], np.eye(3).ravel(), atol=1e-12)
    np.testing.assert_allclose(obs[12:15], [0.1, -0.2, 0.3])
    gate0 = obs[15:27].reshape(4, 3)
    np.testing.assert_allclose(gate0[0], [3.5, -0.8, -1.2], atol=1e-12)  # top left
    np.testing.assert_allclose(gate0[2], [3.5, 0.8, 0.2], atol=1e-12)  # bottom right
    gate1 = obs[27:39].reshape(4, 3)
    h = 0.8 / np.sqrt(2.0)
    np.testing.assert_allclose(gate1[0], [7.5, -2.0 - h, -1.0 - h], atol=1e-12)
    np.testing.assert_allclose(obs[39:43], [0.1, 0.2, 0.3, 0.4])


def test_heading_east_by_hand():
    """Nose East: body x is East, body y (right) is South. Flying North at 2 m/s
    is flying to the left: body velocity (0, -2, 0). Gate 0's top left corner,
    (3.5, -0.8, -1.2) from the drone in NED, is (-0.8, -3.5, -1.2) in body axes."""
    y = state([1.5, 5.0, -1.0], velocity=[2.0, 0.0, 0.0], yaw_deg=90.0)
    obs = observe(y, 0)[0]
    np.testing.assert_allclose(obs[0:3], [0.0, -2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(obs[15:18], [-0.8, -3.5, -1.2], atol=1e-12)


def test_the_attitude_part_is_the_rotation_matrix_row_by_row():
    rng = np.random.default_rng(5)
    q = rng.normal(size=4)
    y = state([3.0, 4.0, -1.0])
    y[6:10] = q / np.linalg.norm(q)
    obs = observe(y, 1)[0]
    np.testing.assert_allclose(obs[3:12], quaternion.to_matrix(y[6:10]).ravel())


def test_any_attitude_gives_body_offsets_that_rotate_back_to_ned():
    """R (body offset) is the NED offset, for random attitudes and gates."""
    rng = np.random.default_rng(6)
    n = 200
    states = np.zeros((n, 13))
    states[:, 0:3] = rng.uniform([0, 0, -4], [18, 10, 0], size=(n, 3))
    q = rng.normal(size=(n, 4))
    states[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    states[:, 3:6] = rng.normal(size=(n, 3))
    next_gate = rng.integers(-1, 4, size=n)
    obs = observe(states, next_gate)
    observer = Observer(COURSE.gates)
    for i in range(n):
        rotation = quaternion.to_matrix(states[i, 6:10])
        np.testing.assert_allclose(rotation @ obs[i, 0:3], states[i, 3:6], atol=1e-12)
        for k, gate in enumerate(observer.targets(next_gate[i : i + 1])[0]):
            body = obs[i, 15 + 12 * k : 27 + 12 * k].reshape(4, 3)
            ned = body @ rotation.T + states[i, 0:3]
            np.testing.assert_allclose(
                ned, opening_points(COURSE.gates[gate]), atol=1e-12
            )


def test_the_last_gate_is_repeated_where_the_course_runs_out():
    observer = Observer(COURSE.gates, lookahead=3)
    np.testing.assert_array_equal(
        observer.targets(np.array([0, 1, 2, 3, -1])),
        [[0, 1, 2], [1, 2, 3], [2, 3, 3], [3, 3, 3], [3, 3, 3]],
    )
    y = state([15.0, 6.0, -1.5])
    after_finish = observe(y, -1)[0]
    at_last = observe(y, 3)[0]
    np.testing.assert_array_equal(after_finish, at_last)
    np.testing.assert_array_equal(at_last[15:27], at_last[27:39])


def test_many_drones_at_once_give_the_same_as_one_by_one():
    rng = np.random.default_rng(7)
    n = 50
    states = np.zeros((n, 13))
    states[:, 0:3] = rng.uniform([0, 0, -4], [18, 10, 0], size=(n, 3))
    q = rng.normal(size=(n, 4))
    states[:, 6:10] = q / np.linalg.norm(q, axis=1, keepdims=True)
    states[:, 3:6] = rng.normal(size=(n, 3))
    states[:, 10:13] = rng.normal(size=(n, 3))
    next_gate = rng.integers(-1, 4, size=n)
    previous = rng.uniform(-1, 1, size=(n, 4))
    together = observe(states, next_gate, previous)
    for i in range(n):
        alone = observe(states[i], next_gate[i], previous[i : i + 1])
        np.testing.assert_array_equal(alone[0], together[i])


def test_observer_rejects_bad_input():
    with pytest.raises(ValueError, match="gate"):
        Observer(())
    with pytest.raises(ValueError, match="lookahead"):
        Observer(COURSE.gates, lookahead=0)
