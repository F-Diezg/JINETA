"""Tests for the multirotor force model (fixed rotor speed commands)."""

import dataclasses

import numpy as np
import pytest

from jineta.campo.atmosphere import isa_density
from jineta.campo.wind import ConstantWind
from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS

WEIGHT = IRIS.mass * GRAVITY
HOVER_SPEEDS = IRIS.rotor_speeds_for(np.full(4, WEIGHT / 4))
ROTOR_DRAG_ONLY = dataclasses.replace(IRIS, body_drag_area=np.zeros(3))


def make_quad(speeds, params=IRIS, **environment):
    quad = Multirotor(params, **environment)
    quad.rotor_speed_command = np.asarray(speeds, dtype=float)
    return quad


def make_state(velocity=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
    """Rigid-body state at 100 m altitude."""
    return np.concatenate([[0.0, 0.0, -100.0], velocity, q, omega])


# --- Thrust and moments ----------------------------------------------------------


def test_hover_is_an_equilibrium():
    quad = make_quad(HOVER_SPEEDS)
    derivative = quad.derivative(0.0, make_state())
    assert np.allclose(derivative[3:6], 0.0, atol=1e-12)  # no acceleration
    assert np.allclose(derivative[10:13], 0.0, atol=1e-12)  # no angular acceleration

    _, states = run(
        quad.derivative, make_state(), dt=0.01, t_end=10.0, post_step=normalize_attitude
    )
    assert np.allclose(states[-1, 0:3], [0.0, 0.0, -100.0], atol=1e-9)


def test_motors_off_is_free_fall():
    quad = make_quad(np.zeros(4))
    assert np.allclose(quad.derivative(0.0, make_state())[3:6], [0.0, 0.0, GRAVITY])


def test_ten_percent_more_speed_climbs():
    """Thrust grows with Omega^2: 1.1^2 = 1.21 times the weight."""
    quad = make_quad(1.1 * HOVER_SPEEDS)
    acceleration = quad.derivative(0.0, make_state())[3:6]
    assert np.allclose(acceleration, [0.0, 0.0, GRAVITY * (1.0 - 1.21)])


def test_rotor_speed_command_is_clipped():
    too_fast = make_quad(np.full(4, 2 * IRIS.max_rotor_speed))
    max_climb = GRAVITY - 4 * IRIS.max_rotor_thrust / IRIS.mass
    assert np.isclose(too_fast.derivative(0.0, make_state())[5], max_climb)

    negative = make_quad(np.full(4, -500.0))
    assert np.isclose(negative.derivative(0.0, make_state())[5], GRAVITY)


def test_thrust_scales_with_air_density():
    """At 3000 m the ISA density is ~74 % of sea level: hover speeds no longer hold."""
    quad = make_quad(HOVER_SPEEDS, density=isa_density)
    state = make_state()
    state[2] = -3000.0
    ratio = isa_density(3000.0) / IRIS.reference_density
    acceleration = quad.derivative(0.0, state)[3:6]
    assert np.allclose(acceleration, [0.0, 0.0, GRAVITY * (1.0 - ratio)])


@pytest.mark.parametrize(
    "moment",
    [
        [0.2, 0.0, 0.0],  # roll
        [0.0, 0.2, 0.0],  # pitch
        [0.0, 0.0, 0.05],  # yaw, from the reaction torques
    ],
)
def test_mixer_moment_gives_angular_acceleration(moment):
    """At rest, Euler's equations reduce to omega_dot = M / I (diagonal inertia)."""
    thrusts = IRIS.rotor_thrusts([WEIGHT, *moment])
    quad = make_quad(IRIS.rotor_speeds_for(thrusts))
    derivative = quad.derivative(0.0, make_state())
    assert np.allclose(derivative[3:6], 0.0, atol=1e-12)
    assert np.allclose(derivative[10:13], np.array(moment) / np.diag(IRIS.inertia))


# --- Drag ---------------------------------------------------------------------------


def test_body_drag_is_quadratic():
    """Motors off (no rotor drag): only the airframe drag, 0.5 rho CdA v^2."""
    for speed in (10.0, 20.0):
        quad = make_quad(np.zeros(4))
        acceleration = quad.derivative(0.0, make_state(velocity=(speed, 0.0, 0.0)))
        drag = 0.5 * 1.225 * IRIS.body_drag_area[0] * speed**2
        assert np.isclose(acceleration[3], -drag / IRIS.mass)


def test_rotor_drag_is_linear_in_airspeed():
    for speed in (2.0, 4.0):
        quad = make_quad(HOVER_SPEEDS, params=ROTOR_DRAG_ONLY)
        acceleration = quad.derivative(0.0, make_state(velocity=(speed, 0.0, 0.0)))
        drag = 4 * IRIS.rotor_drag_coefficient * HOVER_SPEEDS[0] * speed
        assert np.isclose(acceleration[3], -drag / IRIS.mass)


def test_rotor_drag_ignores_air_along_the_rotor_axis():
    quad = make_quad(HOVER_SPEEDS, params=ROTOR_DRAG_ONLY)
    derivative = quad.derivative(0.0, make_state(velocity=(0.0, 0.0, 3.0)))
    assert np.allclose(derivative[3:6], 0.0, atol=1e-12)
    assert np.allclose(derivative[10:13], 0.0, atol=1e-12)


def test_rotor_drag_pitches_nose_up_in_forward_flight():
    """The rotors sit 0.023 m above the CoM, so their drag has a lever arm."""
    speed = 2.0
    quad = make_quad(HOVER_SPEEDS, params=ROTOR_DRAG_ONLY)
    _, moment = quad.force_moment(0.0, make_state(velocity=(speed, 0.0, 0.0)))
    total_drag = 4 * IRIS.rotor_drag_coefficient * HOVER_SPEEDS[0] * speed
    assert np.allclose(moment, [0.0, 0.023 * total_drag, 0.0])


def test_yaw_rate_is_damped_by_rotor_drag():
    """Spinning in yaw moves each hub sideways: Mz = -c_rd Omega r sum(x^2 + y^2)."""
    yaw_rate = 1.0
    quad = make_quad(HOVER_SPEEDS, params=ROTOR_DRAG_ONLY)
    _, moment = quad.force_moment(0.0, make_state(omega=(0.0, 0.0, yaw_rate)))
    arm_squared = sum(r.position[0] ** 2 + r.position[1] ** 2 for r in IRIS.rotors)
    expected = -IRIS.rotor_drag_coefficient * HOVER_SPEEDS[0] * yaw_rate * arm_squared
    assert np.isclose(moment[2], expected)


def test_wind_acts_through_the_airspeed():
    """Moving at 5 m/s in still air = hovering still in a 5 m/s headwind."""
    moving = make_quad(HOVER_SPEEDS)
    in_wind = make_quad(HOVER_SPEEDS, wind=ConstantWind(np.array([-5.0, 0.0, 0.0])))
    a = moving.derivative(0.0, make_state(velocity=(5.0, 0.0, 0.0)))
    b = in_wind.derivative(0.0, make_state())
    assert np.allclose(a[3:13], b[3:13])


def test_attitude_changes_the_drag():
    """Same ground velocity, different attitude: in-plane air gives rotor drag, air
    along the rotor axis does not (the pending in-air decoupling test of phase 4)."""
    speed = 5.0
    quad = make_quad(HOVER_SPEEDS, params=ROTOR_DRAG_ONLY)
    level = make_state(velocity=(speed, 0.0, 0.0))
    nose_up = make_state(
        velocity=(speed, 0.0, 0.0), q=quaternion.from_axis_angle([0, 1, 0], np.pi / 2)
    )
    force_level, _ = quad.force_moment(0.0, level)
    force_nose_up, _ = quad.force_moment(0.0, nose_up)
    drag = 4 * IRIS.rotor_drag_coefficient * HOVER_SPEEDS[0] * speed
    assert np.allclose(force_level[0:2], [-drag, 0.0])
    assert np.allclose(force_nose_up[0:2], 0.0, atol=1e-12)


# --- Parameters ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"rotor_drag_coefficient": -1e-4},
        {"body_drag_area": [0.01, 0.01]},  # 2 values instead of 3
        {"body_drag_area": [-0.01, 0.0, 0.0]},
        {"reference_density": 0.0},
    ],
)
def test_rejects_invalid_drag_parameters(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(IRIS, **changes)


def test_iris_body_drag_matches_px4_ballistic_coefficient():
    """PX4 EKF2_BCOEF_X/Y = mass / CdA = 100 kg/m^2 (generic default)."""
    assert np.allclose(IRIS.mass / IRIS.body_drag_area[0:2], 100.0)
