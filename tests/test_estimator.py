"""Tests for the sensors and the onboard estimate (phase 12, step 4): the
specific force; with perfect sensors and no delay the estimate is the true state
(and close with the delay); the motion-capture delay exact; with the lab's
sensors the errors have the spreads the noise and the filter's theory give;
biases per drone and per episode; restarts of chosen drones; seeds repeat."""

import dataclasses

import numpy as np
import pytest

from jineta.atalaya.estimator import (
    MOCAP_LAB,
    BatchEstimator,
    Sensors,
    specific_force,
)
from jineta.atalaya.kalman import (
    constant_velocity_transition,
    piecewise_white_acceleration,
    predict,
    update,
)
from jineta.campo.course import load_course
from jineta.core import quaternion as core_quaternion
from jineta.core.environment import GRAVITY
from jineta.doma.policy import Agent
from jineta.escuadron import quaternion
from jineta.picadero.env import CourseEnv

DT = 0.01
PERFECT = Sensors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def lap() -> np.ndarray:
    """The phase 11 policy's lap of the first course, every physics step
    (steps, 13): take-off, 8.6 m/s, tilts past 80°."""
    env = CourseEnv(load_course("courses/first_course.yaml"))
    agent = Agent.load("policies/first_course.npz")
    states = []
    step = env.flight.step

    def recording(controller):
        ended = step(controller)
        states.append(env.flight.states[0].copy())
        return ended

    env.flight.step = recording
    observation, _ = env.reset()
    states.append(env.flight.states[0].copy())
    for _ in range(400):
        observation, _, terminated, truncated, _ = env.step(agent.act(observation))
        if terminated[0] or truncated[0]:
            break
    return np.array(states[:-1])  # the last row is the restart


def estimates(sensors: Sensors, path: np.ndarray, n: int = 1, seed: int = 0):
    """The estimate of n drones that all follow `path` (steps, 13), every step
    after the first."""
    estimator = BatchEstimator(sensors, n, DT)
    rng = np.random.default_rng(seed)
    states = np.repeat(path[:, None, :], n, axis=1)
    estimator.reset(None, states[0], rng)
    return np.array(
        [
            estimator.step(states[k - 1], states[k], rng).copy()
            for k in range(1, len(states))
        ]
    )


def attitude_errors(estimated: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Rotation vectors from the true attitudes to the estimated ones [rad]."""
    return quaternion.to_rotation_vector(
        quaternion.multiply(quaternion.conjugate(true), estimated)
    )


def still(height: float = 1.0) -> np.ndarray:
    state = np.zeros(13)
    state[2], state[6] = -height, 1.0
    return state


def test_specific_force_by_hand():
    rest = still()[None]
    np.testing.assert_allclose(specific_force(rest, rest, DT), [[0, 0, -GRAVITY]])
    falling = rest.copy()
    falling[0, 5] = GRAVITY * DT  # free fall: the accelerometer reads nothing
    np.testing.assert_allclose(
        specific_force(rest, falling, DT), [[0, 0, 0]], atol=1e-12
    )
    # Accelerating North at 2 m/s^2 with the nose turned East: in body axes the
    # acceleration is to the left (-y), and 1 g up.
    turned = rest.copy()
    turned[0, 3] = 2.0 * DT
    turned[0, 6:10] = core_quaternion.from_euler(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(
        specific_force(rest, turned, DT), [[0.0, -2.0, -GRAVITY]], atol=1e-12
    )


def test_rotation_vectors_to_quaternions():
    rng = np.random.default_rng(1)
    vectors = rng.normal(0.0, 1.0, (200, 3))
    vectors[:5] *= 1e-14  # tiny rotations
    q = quaternion.from_rotation_vector(vectors)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-15)
    for v, row in zip(vectors[5:], q[5:], strict=True):
        angle = np.linalg.norm(v)
        np.testing.assert_allclose(
            row, core_quaternion.from_axis_angle(v / angle, angle), atol=1e-15
        )
    small = vectors[np.linalg.norm(vectors, axis=1) < np.pi]
    np.testing.assert_allclose(
        quaternion.to_rotation_vector(quaternion.from_rotation_vector(small)),
        small,
        atol=1e-13,
    )


def test_perfect_sensors_without_delay_give_the_true_state(lap):
    """Along the policy's whole lap: position and rates exactly, velocity and
    attitude to rounding."""
    estimated = estimates(PERFECT, lap)[:, 0]
    true = lap[1:]
    np.testing.assert_array_equal(estimated[:, 0:3], true[:, 0:3])
    np.testing.assert_array_equal(estimated[:, 10:13], true[:, 10:13])
    assert np.max(np.abs(estimated[:, 3:6] - true[:, 3:6])) < 1e-12
    assert np.max(np.abs(attitude_errors(estimated[:, 6:10], true[:, 6:10]))) < 1e-14


def test_perfect_sensors_with_the_delay_are_carried_forward(lap):
    """20 ms of motion-capture delay: the output predictor brings the delayed
    frame to now with the IMU; along the lap the estimate stays within 0.1 mm,
    0.1 mm/s and 0.015° (the rates are measured now: exact). Without the
    predictor the delayed frames would be up to 17 cm behind (8.6 m/s x 20 ms)."""
    estimated = estimates(dataclasses.replace(PERFECT, mocap_delay=0.02), lap)[:, 0]
    true = lap[1:]
    assert np.max(np.abs(estimated[:, 0:3] - true[:, 0:3])) < 1e-4
    assert np.max(np.abs(estimated[:, 3:6] - true[:, 3:6])) < 1e-4
    errors = attitude_errors(estimated[:, 6:10], true[:, 6:10])
    assert np.max(np.abs(errors)) < np.radians(0.015)
    np.testing.assert_array_equal(estimated[:, 10:13], true[:, 10:13])
    behind = np.max(np.linalg.norm(lap[3:, 0:3] - lap[1:-2, 0:3], axis=1))
    assert 0.15 < behind < 0.2


def test_the_frames_arrive_exactly_delay_steps_later():
    """A drone whose true position jumps 1 m (no velocity, so the IMU feels
    nothing): with perfect frames the estimate jumps exactly 3 steps later
    (30 ms)."""
    path = np.tile(still(), (12, 1))
    path[5:, 0] += 1.0
    estimated = estimates(dataclasses.replace(PERFECT, mocap_delay=0.03), path)
    north = estimated[:, 0, 0]  # step k of the path is row k - 1
    np.testing.assert_array_equal(north[:6], 0.0)
    np.testing.assert_array_equal(north[7:], 1.0)
    assert north[6] == 0.0 and path[5, 0] == 1.0  # 3 steps: row 7 is step 8
    assert MOCAP_LAB.delay_steps(DT) == 2
    assert dataclasses.replace(MOCAP_LAB, mocap_delay=0.026).delay_steps(DT) == 3


def fixed_gain_covariance(alpha, beta, sigma_input, sigma_position):
    """The steady-state covariance after the update of a filter with the fixed
    gain K = [alpha, beta / dt] (Joseph form, true for any gain), for piecewise
    white acceleration sigma_input and position noise sigma_position."""
    F = np.array([[1.0, DT], [0.0, 1.0]])
    g = np.array([[DT * DT / 2], [DT]])
    Q = sigma_input**2 * g @ g.T
    K = np.array([[alpha], [beta / DT]])
    I_KH = np.eye(2) - K @ np.array([[1.0, 0.0]])
    P = np.eye(2)
    for _ in range(5000):
        P = I_KH @ (F @ P @ F.T + Q) @ I_KH.T + sigma_position**2 * K @ K.T
    return P


def test_the_lab_errors_have_the_spreads_of_the_theory():
    """2000 drones hovering still, the lab's sensors without delay, 150 steps
    after 50 to settle: the gyroscope's error spread is N / sqrt(dt), the
    attitude's the frames' 0.25°, and position and velocity those of the
    filter's own steady state (sideways the attitude noise leaks gravity into
    the input; vertically it does not), each within 5 %."""
    sensors = dataclasses.replace(MOCAP_LAB, mocap_delay=0.0)
    path = np.tile(still(), (201, 1))
    estimated = estimates(sensors, path, n=2000, seed=3)[50:]
    true = path[0]
    rates = estimated[..., 10:13].reshape(-1, 3)
    assert np.std(rates) == pytest.approx(sensors.gyro_noise_density / 0.1, rel=0.05)
    assert abs(np.mean(rates)) < 1e-4
    tilt = attitude_errors(estimated[..., 6:10].reshape(-1, 4), true[6:10])
    assert np.std(tilt) == pytest.approx(sensors.mocap_attitude, rel=0.05)
    estimator = BatchEstimator(sensors, 1, DT)
    accel = sensors.accel_noise_density / np.sqrt(DT)
    for axes, sigma_input in (
        ((0, 1), np.hypot(accel, GRAVITY * sensors.mocap_attitude)),
        ((2,), accel),
    ):
        P = fixed_gain_covariance(
            estimator.alpha, estimator.beta, sigma_input, sensors.mocap_position
        )
        position = estimated[..., [a for a in axes]] - true[[a for a in axes]]
        velocity = estimated[..., [3 + a for a in axes]]
        assert np.std(position) == pytest.approx(np.sqrt(P[0, 0]), rel=0.05)
        assert np.std(velocity) == pytest.approx(np.sqrt(P[1, 1]), rel=0.05)
    # Sideways (where the input's error is largest) the gain is the optimal
    # one: its covariance is the Kalman filter's own steady state.
    sideways = np.hypot(accel, GRAVITY * sensors.mocap_attitude)
    x, P_optimal = np.zeros(2), np.eye(2)
    F = constant_velocity_transition(DT, 1)
    Q = piecewise_white_acceleration(DT, sideways, 1)
    for _ in range(5000):
        x, P_optimal = predict(x, P_optimal, F, Q)
        x, P_optimal, _, _ = update(
            x, P_optimal, [0.0], [[1.0, 0.0]], [[sensors.mocap_position**2]]
        )
    np.testing.assert_allclose(
        fixed_gain_covariance(
            estimator.alpha, estimator.beta, sideways, sensors.mocap_position
        ),
        P_optimal,
        rtol=1e-6,
    )
    # The filter smooths: its position spread is below the frames' own.
    assert np.std(estimated[..., 0] - true[0]) < 0.9 * sensors.mocap_position


def test_the_lab_errors_along_the_lap(lap):
    """200 drones flying the policy's lap with the lab's sensors (20 ms delay):
    position within 1 mm, velocity within 3 cm/s, attitude within 1.5°, rates
    within 0.05 rad/s at every step (measured: 0.43 mm, 1.2 cm/s, 1.1°,
    0.041 rad/s)."""
    estimated = estimates(MOCAP_LAB, lap, n=200, seed=1)
    true = lap[1:, None, :]
    assert np.max(np.abs(estimated[..., 0:3] - true[..., 0:3])) < 1e-3
    assert np.max(np.abs(estimated[..., 3:6] - true[..., 3:6])) < 0.03
    tilt = attitude_errors(
        estimated[..., 6:10].reshape(-1, 4),
        np.repeat(lap[1:, 6:10], 200, axis=0),
    )
    assert np.max(np.linalg.norm(tilt, axis=1)) < np.radians(1.5)
    assert np.max(np.abs(estimated[..., 10:13] - true[..., 10:13])) < 0.05


def test_biases_are_drawn_per_drone_and_per_episode():
    sensors = dataclasses.replace(
        MOCAP_LAB, gyro_bias=0.01, accel_bias=0.05, gyro_noise_density=0.0,
        accel_noise_density=0.0,
    )  # fmt: skip
    estimator = BatchEstimator(sensors, 3000, DT)
    rng = np.random.default_rng(4)
    state = np.tile(still(), (3000, 1))
    estimator.reset(None, state, rng)
    first = estimator.step(state, state, rng)[:, 10:13].copy()
    assert np.std(first) == pytest.approx(0.01, rel=0.05)
    assert np.std(estimator.accel_bias) == pytest.approx(0.05, rel=0.05)
    again = estimator.step(state, state, rng)[:, 10:13]
    np.testing.assert_array_equal(again, first)  # constant during an episode
    chosen = np.zeros(3000, dtype=bool)
    chosen[:10] = True
    estimator.reset(chosen, state[:10], rng)
    after = estimator.step(state, state, rng)[:, 10:13]
    assert np.all(after[:10] != first[:10])  # new biases for the restarted
    np.testing.assert_array_equal(after[10:], first[10:])


def test_a_restart_puts_chosen_drones_back_at_rest():
    """Three drones; drone 1 is restarted elsewhere: its estimate is its new
    start at rest, and the others go on as if nothing happened."""
    estimator = BatchEstimator(PERFECT, 3, DT)
    rng = np.random.default_rng(0)
    states = np.tile(still(), (3, 1))
    estimator.reset(None, states, rng)
    moved = states.copy()
    moved[:, 3] = 0.5
    moved[:, 0] = 0.005
    estimator.step(states, moved, rng)
    before = estimator.estimate.copy()
    start = still(2.0)
    start[0] = 4.0
    estimator.reset([1], start, rng)
    np.testing.assert_array_equal(estimator.estimate[1], start)
    np.testing.assert_array_equal(estimator.estimate[[0, 2]], before[[0, 2]])
    after = estimator.step(np.vstack([moved[0], start, moved[2]]),
                           np.vstack([moved[0], start, moved[2]]), rng)  # fmt: skip
    np.testing.assert_allclose(after[1], start, atol=1e-12)


def test_the_same_seed_gives_the_same_numbers(lap):
    a = estimates(MOCAP_LAB, lap[:50], n=4, seed=9)
    b = estimates(MOCAP_LAB, lap[:50], n=4, seed=9)
    c = estimates(MOCAP_LAB, lap[:50], n=4, seed=10)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_rejects_bad_sensors():
    for name in (
        "gyro_noise_density", "accel_noise_density", "gyro_bias", "accel_bias",
        "mocap_position", "mocap_attitude", "mocap_delay",
    ):  # fmt: skip
        with pytest.raises(ValueError):
            dataclasses.replace(MOCAP_LAB, **{name: -1e-3})
    with pytest.raises(ValueError):
        BatchEstimator(MOCAP_LAB, 0, DT)
    with pytest.raises(ValueError):
        BatchEstimator(MOCAP_LAB, 1, 0.0)
