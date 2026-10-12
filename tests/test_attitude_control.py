"""Tests for the multirotor attitude controller (middle loop of the cascade)."""

import dataclasses

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import (
    AttitudeController,
    AttitudeGains,
    RateController,
    RateGains,
)
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_ATTITUDE_GAINS, IRIS_RATE_GAINS

LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
WEIGHT = IRIS.mass * GRAVITY
DT = 0.004
KP = IRIS_ATTITUDE_GAINS.kp


def state_with(q):
    return np.concatenate([[0.0, 0.0, -100.0], np.zeros(3), q, np.zeros(3)])


# --- The control law ------------------------------------------------------------------


def test_rate_command_is_minus_kp_times_the_error():
    q = quaternion.from_axis_angle([1, 0, 0], 0.2)  # rolled 0.2 rad to the right
    rate = AttitudeController(IRIS_ATTITUDE_GAINS).update(LEVEL, state_with(q))
    assert np.allclose(rate, [-KP[0] * 0.2, 0.0, 0.0])


def test_error_is_measured_in_body_axes_from_the_command():
    """Commanded: nose East. Actual: nose East and rolled 0.1 rad. The error is a
    roll, about the body x axis (pointing East), not about North."""
    q_cmd = quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    roll = quaternion.from_axis_angle([1, 0, 0], 0.1)
    q = quaternion.multiply(q_cmd, roll)
    rate = AttitudeController(IRIS_ATTITUDE_GAINS).update(q_cmd, state_with(q))
    assert np.allclose(rate, [-KP[0] * 0.1, 0.0, 0.0])


def test_turns_the_short_way():
    """Heading 350 degrees, commanded 0: turn 10 degrees right, not 350 left."""
    q = quaternion.from_axis_angle([0, 0, 1], np.radians(350.0))
    rate = AttitudeController(IRIS_ATTITUDE_GAINS).update(LEVEL, state_with(q))
    assert np.allclose(rate, [0.0, 0.0, KP[2] * np.radians(10.0)])


def test_rate_command_is_clipped():
    q = quaternion.from_axis_angle([1, 0, 0], np.radians(170.0))
    rate = AttitudeController(IRIS_ATTITUDE_GAINS).update(LEVEL, state_with(q))
    assert np.allclose(rate, [-IRIS_ATTITUDE_GAINS.rate_limit[0], 0.0, 0.0])


def test_rejects_invalid_gains():
    with pytest.raises(ValueError):
        AttitudeGains(kp=[1.0, 1.0], rate_limit=[1.0, 1.0, 1.0])


# --- Closed loop ----------------------------------------------------------------------


def fly_cascade(params, rate_gains, q0, t_end):
    """Attitude loop -> rate loop -> quad, holding level, starting at attitude q0."""
    quad = Multirotor(params)
    attitude = AttitudeController(IRIS_ATTITUDE_GAINS)
    rate = RateController(params, rate_gains, DT)

    def pre_step(t, state):
        rate_command = attitude.update(LEVEL, state)
        quad.rotor_speed_command = rate.update(rate_command, WEIGHT, state)

    return run(
        quad.derivative,
        state_with(q0),
        dt=DT,
        t_end=t_end,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )


def test_roll_recovery_matches_the_exact_discrete_cascade():
    """Roll only, P-only rate loop, no rotor drag: the whole 6-DOF loop reduces to
    alpha_k = kp_r (-kp_a phi_k - p_k), held for one step, so
    p_(k+1) = p_k + alpha_k dt and phi_(k+1) = phi_k + p_k dt + alpha_k dt^2 / 2."""
    kp_rate = IRIS_RATE_GAINS.kp
    p_only = RateGains(kp=kp_rate, ki=[0, 0, 0], integral_limit=[0, 0, 0])
    no_rotor_drag = dataclasses.replace(IRIS, rotor_drag_coefficient=0.0)
    phi0 = np.radians(30.0)
    times, states = fly_cascade(
        no_rotor_drag, p_only, quaternion.from_axis_angle([1, 0, 0], phi0), 1.0
    )
    roll = 2.0 * np.arctan2(states[:, 7], states[:, 6])

    phi, p = phi0, 0.0
    expected = [phi]
    for _ in range(len(times) - 1):
        alpha = kp_rate[0] * (-KP[0] * phi - p)
        phi, p = phi + p * DT + 0.5 * alpha * DT**2, p + alpha * DT
        expected.append(phi)
    assert np.allclose(roll, expected, atol=1e-6)

    # Close to the continuous critically damped response (1 + w t) exp(-w t),
    # w = kp_rate / 2 = 10 rad/s; the difference is the discretisation.
    w = kp_rate[0] / 2
    continuous = phi0 * (1 + w * times) * np.exp(-w * times)
    assert np.allclose(roll, continuous, atol=0.01 * phi0)


def test_recovers_from_a_combined_tilt_with_the_real_gains():
    """Full model (rotor drag, PI rate loop): roll 30 and pitch -20 degrees at once.
    Behaviour test. The tilt (angle of the thrust axis from the vertical) is what
    moves a multirotor and comes back fast with no overshoot; the composed rotation
    also holds a 5 degree heading error, which the slower yaw loop removes later."""
    tilt = quaternion.multiply(
        quaternion.from_axis_angle([1, 0, 0], np.radians(30.0)),
        quaternion.from_axis_angle([0, 1, 0], np.radians(-20.0)),
    )
    times, states = fly_cascade(IRIS, IRIS_RATE_GAINS, tilt, 3.0)
    tilt_deg = np.degrees(
        [
            np.arccos(np.clip(quaternion.to_matrix(q)[2, 2], -1, 1))
            for q in states[:, 6:10]
        ]
    )
    error_deg = np.degrees([quaternion.to_rotation_vector(q) for q in states[:, 6:10]])

    assert np.all(tilt_deg[times >= 0.6] < 1.0)
    assert np.all(tilt_deg[times >= 1.0] < 0.3)
    assert np.all(np.abs(error_deg[times >= 2.0, 2]) < 0.5)  # heading
    # No overshoot: roll error starts positive, pitch error negative; neither crosses.
    assert error_deg[:, 0].min() > -0.1
    assert error_deg[:, 1].max() < 0.1
