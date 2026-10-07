"""Tests for point-mass dynamics against known physical results."""

import numpy as np
import pytest

from jineta.core.point_mass import GRAVITY, PointMass
from jineta.core.simulation import run


def initial_state(speed: float, elevation_deg: float) -> np.ndarray:
    """Launch from the origin heading north, climbing at the given angle (NED)."""
    angle = np.radians(elevation_deg)
    vx = speed * np.cos(angle)
    vz = -speed * np.sin(angle)  # negative = upwards in NED
    return np.array([0.0, 0.0, 0.0, vx, 0.0, vz])


def hit_ground(t: float, state: np.ndarray) -> bool:
    return state[2] > 0.0  # z > 0 means below ground level


def test_vacuum_trajectory_matches_analytic():
    body = PointMass(mass=1.0, drag_area=0.0)
    y0 = initial_state(speed=50.0, elevation_deg=45.0)
    times, states = run(body.derivative, y0, dt=0.01, t_end=3.0)

    t = times[-1]
    expected_x = y0[3] * t
    expected_z = y0[5] * t + 0.5 * GRAVITY * t**2
    assert np.isclose(states[-1, 0], expected_x, atol=1e-6)
    assert np.isclose(states[-1, 2], expected_z, atol=1e-6)


def test_falling_body_reaches_terminal_velocity():
    body = PointMass(mass=1.0, drag_area=0.05)
    y0 = np.zeros(6)  # at rest
    _, states = run(body.derivative, y0, dt=0.01, t_end=60.0)

    terminal = np.sqrt(2 * body.mass * GRAVITY / (body.density(0.0) * body.drag_area))
    assert np.isclose(states[-1, 5], terminal, rtol=1e-6)


def test_drag_reduces_range():
    y0 = initial_state(speed=50.0, elevation_deg=45.0)
    vacuum = PointMass(mass=1.0, drag_area=0.0)
    with_drag = PointMass(mass=1.0, drag_area=0.01)

    _, s_vacuum = run(vacuum.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground)
    _, s_drag = run(with_drag.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground)

    assert s_drag[-1, 0] < s_vacuum[-1, 0]


def test_thinner_air_at_altitude_increases_range():
    def exponential_density(altitude: float) -> float:
        return 1.225 * np.exp(-altitude / 8500.0)

    y0 = initial_state(speed=300.0, elevation_deg=60.0)
    constant = PointMass(mass=10.0, drag_area=0.005)
    thinning = PointMass(mass=10.0, drag_area=0.005, density=exponential_density)

    _, s_const = run(constant.derivative, y0, dt=0.01, t_end=120.0, stop=hit_ground)
    _, s_thin = run(thinning.derivative, y0, dt=0.01, t_end=120.0, stop=hit_ground)

    assert s_thin[-1, 0] > s_const[-1, 0]


def test_body_moving_with_the_wind_feels_no_drag():
    def steady_wind(t, position):
        return np.array([10.0, -5.0, 0.0])

    body = PointMass(mass=1.0, drag_area=0.05, wind=steady_wind)
    state = np.array([0.0, 0.0, -100.0, 10.0, -5.0, 0.0])  # moves with the air

    accel = body.derivative(0.0, state)[3:6]

    assert np.allclose(accel, [0.0, 0.0, GRAVITY])  # only gravity acts


@pytest.mark.parametrize(
    "wind_north, airspeed",
    [
        (-10.0, 60.0),  # headwind: air moves south, the body flies north
        (0.0, 50.0),  # still air (the default model)
        (10.0, 40.0),  # tailwind: air moves north with the body
    ],
)
def test_drag_depends_on_airspeed_not_ground_speed(wind_north, airspeed):
    mass, drag_area, ground_speed = 2.0, 0.01, 50.0

    def steady_wind(t, position):
        return np.array([wind_north, 0.0, 0.0])

    body = PointMass(mass=mass, drag_area=drag_area, wind=steady_wind)
    state = np.array([0.0, 0.0, 0.0, ground_speed, 0.0, 0.0])  # flying north

    accel_north = body.derivative(0.0, state)[3]

    expected = -0.5 * body.density(0.0) * drag_area * airspeed**2 / mass
    assert np.isclose(accel_north, expected)


def test_falling_body_drifts_to_the_wind_speed():
    wind = np.array([8.0, 3.0, 0.0])

    def steady_wind(t, position):
        return wind

    body = PointMass(mass=1.0, drag_area=0.05, wind=steady_wind)
    y0 = np.zeros(6)  # at rest
    _, states = run(body.derivative, y0, dt=0.01, t_end=60.0)

    terminal = np.sqrt(2 * body.mass * GRAVITY / (body.density(0.0) * body.drag_area))
    assert np.allclose(states[-1, 3:5], wind[:2], atol=1e-6)  # drifts with the air
    assert np.isclose(states[-1, 5], terminal, rtol=1e-6)  # same terminal fall speed


def test_wind_model_receives_time_and_position():
    calls = []

    def recording_wind(t, position):
        calls.append((t, position.copy()))
        return np.zeros(3)

    body = PointMass(mass=1.0, drag_area=0.01, wind=recording_wind)
    state = np.array([10.0, 20.0, -300.0, 5.0, 0.0, 0.0])

    body.derivative(3.0, state)

    t, position = calls[0]
    assert t == 3.0
    assert np.array_equal(position, [10.0, 20.0, -300.0])
