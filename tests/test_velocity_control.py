"""Tests for the velocity loop and the full autopilot (outer loop of the cascade)."""

import dataclasses

import numpy as np
import pytest

from jineta.campo.atmosphere import isa_density
from jineta.campo.wind import ConstantWind
from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import Autopilot, VelocityController
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_VELOCITY_GAINS

WEIGHT = IRIS.mass * GRAVITY
DT = 0.01
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])
# P only, so that one update gives exactly a = kp * e.
P_ONLY = dataclasses.replace(
    IRIS_VELOCITY_GAINS, ki=[0.0, 0.0, 0.0], integral_limit=[0.0, 0.0, 0.0]
)


def state_with(velocity=(0.0, 0.0, 0.0), q=LEVEL, altitude=100.0):
    return np.concatenate([[0.0, 0.0, -altitude], velocity, q, np.zeros(3)])


def tilt_of(q):
    """Angle between the thrust axis and the vertical [rad]."""
    return np.arccos(np.clip(quaternion.to_matrix(q)[2, 2], -1.0, 1.0))


# --- The control law ------------------------------------------------------------------


def test_hover_asks_for_level_and_the_weight():
    attitude, thrust = VelocityController(IRIS, P_ONLY, DT).update(
        np.zeros(3), 0.0, state_with()
    )
    assert np.allclose(attitude, LEVEL)
    assert np.isclose(thrust, WEIGHT)


def test_heading_command_turns_the_nose():
    attitude, _ = VelocityController(IRIS, P_ONLY, DT).update(
        np.zeros(3), np.pi / 2, state_with()
    )
    expected = quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    assert np.allclose(quaternion.to_matrix(attitude), quaternion.to_matrix(expected))


def test_acceleration_north_pitches_the_nose_down():
    """Asking for a = 2 m/s^2 North: thrust m (a - g) leans forward, the nose goes
    down by atan(a / g), and from a level attitude the thrust is still m g (the
    projection on the current, vertical, thrust axis)."""
    a = 2.0
    velocity_command = [a / P_ONLY.kp[0], 0.0, 0.0]
    attitude, thrust = VelocityController(IRIS, P_ONLY, DT).update(
        velocity_command, 0.0, state_with()
    )
    nose = quaternion.to_matrix(attitude)[:, 0]
    assert np.isclose(np.arcsin(nose[2]), np.arctan(a / GRAVITY))  # nose down
    assert np.isclose(nose[1], 0.0)
    assert np.isclose(thrust, WEIGHT)


def test_tilt_is_limited_and_altitude_comes_first():
    attitude, thrust = VelocityController(IRIS, P_ONLY, DT).update(
        [100.0, 0.0, 0.0], 0.0, state_with()
    )
    assert np.isclose(tilt_of(attitude), P_ONLY.max_tilt)
    assert np.isclose(thrust, WEIGHT)  # vertical part kept, horizontal shortened


def test_total_thrust_is_limited():
    controller = VelocityController(IRIS, P_ONLY, DT)
    attitude, thrust = controller.update([0.0, 0.0, -100.0], 0.0, state_with())
    assert np.allclose(attitude, LEVEL)
    assert np.isclose(thrust, controller.max_thrust)
    assert np.isclose(controller.max_thrust, 0.9 * 4 * IRIS.max_rotor_thrust)


def test_thrust_is_projected_on_the_current_axis():
    """Rolled 60 degrees while asked to hover: push only m g cos(60) along the
    tilted axis, so the vertical part does not overshoot while levelling."""
    rolled = quaternion.from_axis_angle([1, 0, 0], np.radians(60.0))
    _, thrust = VelocityController(IRIS, P_ONLY, DT).update(
        np.zeros(3), 0.0, state_with(q=rolled)
    )
    assert np.isclose(thrust, WEIGHT * 0.5)


def test_no_thrust_when_upside_down():
    upside_down = quaternion.from_axis_angle([1, 0, 0], np.pi)
    _, thrust = VelocityController(IRIS, P_ONLY, DT).update(
        np.zeros(3), 0.0, state_with(q=upside_down)
    )
    assert thrust == 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"max_tilt": 0.0},
        {"max_tilt": np.pi / 2},
        {"thrust_range": (0.5, 0.4)},
    ],
)
def test_rejects_invalid_velocity_gains(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(IRIS_VELOCITY_GAINS, **changes)


# --- Closed loop: the whole autopilot ------------------------------------------------


def fly(params, velocity_command, t_end, altitude=100.0, **environment):
    quad = Multirotor(params, **environment)
    autopilot = Autopilot(quad, IRIS_GAINS, DT)
    autopilot.velocity_command = np.asarray(velocity_command, dtype=float)
    times, states = run(
        quad.derivative,
        state_with(altitude=altitude),
        dt=DT,
        t_end=t_end,
        pre_step=autopilot,
        post_step=normalize_attitude,
    )
    return times, states, quad


def test_velocity_step_behaviour():
    """Behaviour test with the PX4 gains, from hover to 5 m/s North. Drag leaves a
    steady error for the P term that the (slow) integral removes over seconds."""
    times, states, _ = fly(IRIS, [5.0, 0.0, 0.0], 10.0)
    north, east = states[:, 3], states[:, 4]
    assert north[times >= 1.0].min() > 4.0
    assert north[times >= 2.0].min() > 4.5
    assert abs(north[-1] - 5.0) < 0.1
    assert north.max() <= 5.0  # no overshoot
    assert np.allclose(east, 0.0, atol=1e-9)
    assert np.abs(states[:, 2] + 100.0).max() < 0.5  # altitude held
    tilts = [tilt_of(q) for q in states[:, 6:10]]
    assert max(tilts) < IRIS_VELOCITY_GAINS.max_tilt


def test_holding_still_in_wind_tilts_into_it_by_atan_drag_over_weight():
    """Wind of 10 m/s from the North, rotor drag removed so only the body drag
    D = 0.5 rho CdA w^2 acts (at the CoM). Holding zero ground velocity, the thrust
    must lean into the wind by atan(D / m g), with the nose (facing North) down."""
    params = dataclasses.replace(IRIS, rotor_drag_coefficient=0.0)
    wind_speed = 10.0
    _, states, _ = fly(
        params, np.zeros(3), 30.0, wind=ConstantWind(np.array([-wind_speed, 0, 0]))
    )
    drag = 0.5 * 1.225 * IRIS.body_drag_area[0] * wind_speed**2
    final = states[-1]
    assert np.isclose(tilt_of(final[6:10]), np.arctan(drag / WEIGHT), atol=1e-5)
    assert quaternion.to_matrix(final[6:10])[2, 0] > 0.0  # nose below the horizon
    assert np.allclose(final[3:6], 0.0, atol=1e-3)


def test_hover_at_altitude_needs_more_rotor_speed():
    """At 3000 m the air is thinner; the vertical integral finds the rotor speed
    that gives the weight again: Omega = Omega_sea_level * sqrt(rho_0 / rho)."""
    _, states, quad = fly(IRIS, np.zeros(3), 10.0, altitude=3000.0, density=isa_density)
    ratio = isa_density(-states[-1, 2]) / IRIS.reference_density
    sea_level_hover = IRIS.rotor_speeds_for(np.full(4, WEIGHT / 4))
    assert np.allclose(
        quad.rotor_speed_command, sea_level_hover / np.sqrt(ratio), rtol=1e-3
    )
    assert abs(states[-1, 5]) < 0.01  # vertical speed back to ~0
