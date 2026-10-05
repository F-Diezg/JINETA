"""Tests for point-mass dynamics against known physical results."""

import numpy as np

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

    terminal = np.sqrt(2 * body.mass * GRAVITY / (body.air_density * body.drag_area))
    assert np.isclose(states[-1, 5], terminal, rtol=1e-6)


def test_drag_reduces_range():
    y0 = initial_state(speed=50.0, elevation_deg=45.0)
    vacuum = PointMass(mass=1.0, drag_area=0.0)
    with_drag = PointMass(mass=1.0, drag_area=0.01)

    _, s_vacuum = run(vacuum.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground)
    _, s_drag = run(with_drag.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground)

    assert s_drag[-1, 0] < s_vacuum[-1, 0]