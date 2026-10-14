"""Tests for the shared aerodynamic helpers (airspeed in body axes, alpha, beta)."""

import numpy as np
import pytest

from jineta.campo.wind import ConstantWind
from jineta.core import quaternion
from jineta.jinete.aero import air_data, airspeed_from_air_data, body_airspeed


def make_state(velocity, q=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, -100.0)):
    """13-element rigid-body state with zero angular rate."""
    return np.concatenate([position, velocity, q, np.zeros(3)])


@pytest.mark.parametrize(
    "axis, angle_deg, expected_body",
    [
        # Level, nose North, flying North: all the air comes onto the nose.
        ([0, 0, 1], 0.0, [10.0, 0.0, 0.0]),
        # Nose East (yaw +90°), flying North: the right wing points South, so
        # North is to the left -> negative v.
        ([0, 0, 1], 90.0, [0.0, -10.0, 0.0]),
        # Nose up 30° (pitch +30°), flying level North: the air comes from below
        # the nose -> positive w, angle of attack = atan2(w, u) = 30°.
        ([0, 1, 0], 30.0, [10.0 * np.cos(np.pi / 6), 0.0, 10.0 * np.sin(np.pi / 6)]),
        # Rolled right 90° (right wing down), flying North: still all on the nose.
        ([1, 0, 0], 90.0, [10.0, 0.0, 0.0]),
    ],
)
def test_ground_velocity_seen_from_the_body(axis, angle_deg, expected_body):
    q = quaternion.from_axis_angle(axis, np.radians(angle_deg))
    state = make_state(velocity=[10.0, 0.0, 0.0], q=q)
    assert np.allclose(body_airspeed(0.0, state), expected_body)


def test_flying_with_the_wind_feels_no_air():
    wind = ConstantWind(np.array([3.0, -4.0, 1.0]))
    state = make_state(velocity=[3.0, -4.0, 1.0])
    assert np.allclose(body_airspeed(0.0, state, wind), 0.0)


def test_hovering_in_a_headwind():
    """Still over the ground, nose North, wind blowing South: air hits the nose."""
    wind = ConstantWind(np.array([-5.0, 0.0, 0.0]))
    state = make_state(velocity=[0.0, 0.0, 0.0])
    assert np.allclose(body_airspeed(0.0, state, wind), [5.0, 0.0, 0.0])


def test_rotation_keeps_the_airspeed_magnitude():
    """Rotating a vector into another frame never changes its length."""
    rng = np.random.default_rng(seed=0)
    wind = ConstantWind(np.array([2.0, 7.0, -1.0]))
    for _ in range(20):
        q = quaternion.from_axis_angle(rng.normal(size=3), rng.uniform(0, 2 * np.pi))
        velocity = rng.normal(scale=10.0, size=3)
        state = make_state(velocity=velocity, q=q)
        expected = np.linalg.norm(velocity - wind.velocity)
        assert np.isclose(np.linalg.norm(body_airspeed(0.0, state, wind)), expected)


def test_wind_model_receives_time_and_position():
    calls = []

    def recording_wind(t, position):
        calls.append((t, position.copy()))
        return np.zeros(3)

    state = make_state(velocity=[1.0, 0.0, 0.0], position=(10.0, 20.0, -300.0))
    body_airspeed(4.5, state, recording_wind)

    assert len(calls) == 1
    t, position = calls[0]
    assert t == 4.5
    assert np.array_equal(position, [10.0, 20.0, -300.0])


# --- Angle of attack and sideslip -----------------------------------------------------


@pytest.mark.parametrize(
    "axis, angle_deg, alpha_deg, beta_deg",
    [
        # Level, nose along the flight path: the air comes straight onto the nose.
        ([0, 0, 1], 0.0, 0.0, 0.0),
        # Nose 30 deg up, still flying level: the air comes from below the nose.
        ([0, 1, 0], 30.0, 30.0, 0.0),
        # Nose 10 deg right of the flight path (yaw +10): the vehicle moves towards
        # its left wing, so the air comes from the left -> beta = -10.
        ([0, 0, 1], 10.0, 0.0, -10.0),
        # Nose 10 deg left of the flight path: air from the right -> beta = +10.
        ([0, 0, 1], -10.0, 0.0, 10.0),
        # Rolled 90 deg (knife edge) along the flight path: still all on the nose.
        ([1, 0, 0], 90.0, 0.0, 0.0),
    ],
)
def test_alpha_and_beta_for_known_attitudes(axis, angle_deg, alpha_deg, beta_deg):
    """Flying North at 20 m/s in still air with the nose turned by a known angle."""
    q = quaternion.from_axis_angle(axis, np.radians(angle_deg))
    state = make_state(velocity=[20.0, 0.0, 0.0], q=q)
    data = air_data(body_airspeed(0.0, state))
    assert np.isclose(data.speed, 20.0)
    assert np.isclose(np.degrees(data.alpha), alpha_deg)
    assert np.isclose(np.degrees(data.beta), beta_deg)


def test_air_data_round_trip():
    """V, alpha, beta -> [u, v, w] -> V, alpha, beta, over the whole range of angles
    (alpha in (-180, 180) deg, beta in (-90, 90) deg)."""
    rng = np.random.default_rng(seed=1)
    for _ in range(50):
        speed = rng.uniform(1.0, 60.0)
        alpha = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        beta = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
        data = air_data(airspeed_from_air_data(speed, alpha, beta))
        assert np.isclose(data.speed, speed)
        assert np.isclose(data.alpha, alpha)
        assert np.isclose(data.beta, beta)


def test_flying_tail_first_is_alpha_180():
    """atan2 keeps the quadrant: air from behind is alpha = 180 deg, not 0 (atan would
    have mixed it up with flying forwards)."""
    data = air_data([-15.0, 0.0, 0.0])
    assert np.isclose(abs(data.alpha), np.pi)
    assert data.beta == 0.0


def test_no_airspeed_gives_zero_angles():
    """Without air the angles are undefined; they are 0, never NaN."""
    for airspeed in ([0.0, 0.0, 0.0], [1e-9, -1e-9, 1e-9]):
        data = air_data(airspeed)
        assert data.speed < 1e-6
        assert data.alpha == 0.0
        assert data.beta == 0.0


@pytest.mark.parametrize("airspeed", [[1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
def test_air_data_rejects_wrong_size(airspeed):
    with pytest.raises(ValueError):
        air_data(airspeed)


def test_airspeed_from_air_data_rejects_negative_speed():
    with pytest.raises(ValueError):
        airspeed_from_air_data(-1.0, 0.0, 0.0)
