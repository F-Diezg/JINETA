"""Tests for the multirotor rate controller (inner loop of the cascade)."""

import dataclasses

import numpy as np
import pytest

from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import RateController, RateGains
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_RATE_GAINS

# Without rotor drag each axis is exactly omega_dot = M / I, so the closed loop can
# be compared with closed-form results.
NO_ROTOR_DRAG = dataclasses.replace(IRIS, rotor_drag_coefficient=0.0)
WEIGHT = IRIS.mass * GRAVITY
DT = 0.004  # 250 Hz
HOVER = np.array([0.0, 0.0, -100.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])


def fly(controller, rate_command, t_end, quad=None):
    """Simulate the quad with the rate loop in pre_step; return times and states."""
    quad = quad if quad is not None else Multirotor(controller.params)

    def pre_step(t, state):
        quad.rotor_speed_command = controller.update(rate_command, WEIGHT, state)

    return run(
        quad.derivative,
        HOVER,
        dt=DT,
        t_end=t_end,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )


def p_only(kp):
    return RateGains(kp=kp, ki=[0.0, 0.0, 0.0], integral_limit=[0.0, 0.0, 0.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_p_only_step_is_an_exact_discrete_first_order(axis):
    """omega_dot = kp (omega_cmd - omega_k), held for one step, gives
    e_(k+1) = (1 - kp dt) e_k: the error shrinks by a fixed factor every step
    (the discrete version of exp(-t / tau), tau = 1 / kp)."""
    kp = IRIS_RATE_GAINS.kp
    controller = RateController(NO_ROTOR_DRAG, p_only(kp), DT)
    command = np.zeros(3)
    command[axis] = 0.5
    times, states = fly(controller, command, t_end=0.5)

    steps = np.arange(len(times))
    expected = 0.5 * (1.0 - (1.0 - kp[axis] * DT) ** steps)
    assert np.allclose(states[:, 10 + axis], expected, atol=1e-9)
    other_axes = [i for i in range(3) if i != axis]
    assert np.allclose(states[:, 10 + np.array(other_axes)], 0.0, atol=1e-9)


def test_gyroscopic_coupling_is_cancelled():
    """Holding p = r = 1 rad/s: Euler's equations alone would give
    q_dot = p r (Izz - Ixx) / Iyy != 0. With the omega x I omega term, zero error
    means zero angular acceleration."""
    controller = RateController(NO_ROTOR_DRAG, IRIS_RATE_GAINS, DT)
    quad = Multirotor(NO_ROTOR_DRAG)
    state = HOVER.copy()
    state[10:13] = [1.0, 0.0, 1.0]
    quad.rotor_speed_command = controller.update([1.0, 0.0, 1.0], WEIGHT, state)
    assert np.allclose(quad.derivative(0.0, state)[10:13], 0.0, atol=1e-12)


def disturbed_quad(moment):
    """A quad with a constant extra moment, like a centre of mass off its centre."""
    quad = Multirotor(NO_ROTOR_DRAG)
    clean = quad.force_moment

    def force_moment(t, state):
        force, m = clean(t, state)
        return force, m + moment

    quad.body.force_moment = force_moment
    return quad


def test_p_only_leaves_a_steady_error_under_a_disturbance():
    """In steady state I kp e + d = 0, so omega settles at d / (I kp), not at 0."""
    disturbance = np.array([0.05, 0.0, 0.0])  # [N·m]
    controller = RateController(NO_ROTOR_DRAG, p_only(IRIS_RATE_GAINS.kp), DT)
    _, states = fly(controller, np.zeros(3), 2.0, disturbed_quad(disturbance))
    expected = disturbance[0] / (IRIS.inertia[0, 0] * IRIS_RATE_GAINS.kp[0])
    assert np.isclose(states[-1, 10], expected, rtol=1e-6)


def test_integral_removes_the_steady_error():
    disturbance = np.array([0.05, 0.0, 0.0])
    controller = RateController(NO_ROTOR_DRAG, IRIS_RATE_GAINS, DT)
    _, states = fly(controller, np.zeros(3), 3.0, disturbed_quad(disturbance))
    assert abs(states[-1, 10]) < 1e-6


def test_integral_is_clamped():
    """A persistent error (e.g. a saturated rotor) must not grow the integral
    without bound, or the vehicle overshoots badly once the error goes away."""
    controller = RateController(IRIS, IRIS_RATE_GAINS, DT)
    for _ in range(1000):
        controller.update([100.0, -100.0, 100.0], WEIGHT, HOVER)
    limit = IRIS_RATE_GAINS.integral_limit
    assert np.array_equal(controller.integral, [limit[0], -limit[1], limit[2]])

    controller.reset()
    assert np.array_equal(controller.integral, np.zeros(3))


@pytest.mark.parametrize(
    "gains",
    [
        {"kp": [1.0, 1.0], "ki": [0, 0, 0], "integral_limit": [0, 0, 0]},
        {"kp": [1.0, 1.0, -1.0], "ki": [0, 0, 0], "integral_limit": [0, 0, 0]},
    ],
)
def test_rejects_invalid_gains(gains):
    with pytest.raises(ValueError):
        RateGains(**gains)


def test_rejects_non_positive_period():
    with pytest.raises(ValueError):
        RateController(IRIS, IRIS_RATE_GAINS, dt=0.0)
