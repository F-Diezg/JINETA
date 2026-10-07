"""Tests for the wind models: conventions, interpolation and input validation."""

import numpy as np
import pytest

from jineta.campo.wind import ConstantWind, ProfileWind, meteorological_wind


def position_at(altitude: float) -> np.ndarray:
    """NED position at the given altitude, over the origin (z points down)."""
    return np.array([0.0, 0.0, -altitude])


@pytest.mark.parametrize(
    "from_deg, expected_ned",
    [
        (0.0, [-10.0, 0.0, 0.0]),  # from the north -> blows toward the south
        (90.0, [0.0, -10.0, 0.0]),  # from the east -> blows toward the west
        (180.0, [10.0, 0.0, 0.0]),  # from the south -> blows toward the north
        (270.0, [0.0, 10.0, 0.0]),  # from the west -> blows toward the east
    ],
)
def test_meteorological_wind_points_where_the_air_goes(from_deg, expected_ned):
    wind = meteorological_wind(speed=10.0, direction_from_deg=from_deg)
    assert np.allclose(wind, expected_ned, atol=1e-12)


def test_meteorological_wind_keeps_speed_and_has_no_vertical_part():
    wind = meteorological_wind(speed=7.0, direction_from_deg=37.0)
    assert np.isclose(np.linalg.norm(wind), 7.0)
    assert wind[2] == 0.0


def test_constant_wind_does_not_depend_on_time_or_position():
    wind = ConstantWind([3.0, -4.0, 0.0])
    samples = [
        (0.0, position_at(-50.0)),
        (50.0, position_at(5_000.0)),
        (7.0, np.array([10_000.0, -20_000.0, -300.0])),
    ]
    for t, position in samples:
        assert np.array_equal(wind(t, position), [3.0, -4.0, 0.0])


def test_constant_wind_returns_a_copy():
    wind = ConstantWind([3.0, -4.0, 0.0])
    wind(0.0, position_at(0.0))[0] = 99.0  # modify the returned array
    assert np.array_equal(wind(0.0, position_at(0.0)), [3.0, -4.0, 0.0])


def shear_wind() -> ProfileWind:
    """Wind table with easy numbers: calm at the ground, stronger and turning above."""
    return ProfileWind(
        altitudes=[0.0, 100.0, 1_000.0],
        velocities=[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-3.0, 10.0, 1.0]],
    )


def test_profile_wind_returns_table_values_at_table_altitudes():
    wind = shear_wind()
    for altitude, velocity in zip(wind.altitudes, wind.velocities):
        assert np.allclose(wind(0.0, position_at(altitude)), velocity)


def test_profile_wind_interpolates_linearly():
    wind = shear_wind()
    # Altitude is -z: 50 m up means z = -50 in NED
    assert np.allclose(wind(0.0, position_at(50.0)), [0.0, 2.5, 0.0])  # rows 0 and 1
    assert np.allclose(wind(0.0, position_at(550.0)), [-1.5, 7.5, 0.5])  # rows 1 and 2


def test_profile_wind_holds_end_values_outside_the_table():
    wind = shear_wind()
    assert np.allclose(wind(0.0, position_at(-20.0)), [0.0, 0.0, 0.0])  # below
    assert np.allclose(wind(0.0, position_at(8_000.0)), [-3.0, 10.0, 1.0])  # above


def test_profile_wind_ignores_time_and_horizontal_position():
    wind = shear_wind()
    reference = wind(0.0, position_at(550.0))
    far_away = np.array([20_000.0, -35_000.0, -550.0])  # same altitude, elsewhere
    assert np.array_equal(wind(123.0, far_away), reference)


def test_profile_wind_interpolates_components_not_angles():
    """Wind turning from 350 to 010 degrees must pass through north (0 degrees)."""
    wind = ProfileWind(
        altitudes=[0.0, 100.0],
        velocities=[meteorological_wind(10.0, 350.0), meteorological_wind(10.0, 10.0)],
    )
    halfway = wind(0.0, position_at(50.0))
    assert np.isclose(halfway[1], 0.0, atol=1e-12)  # no east component
    assert np.isclose(halfway[0], -10.0 * np.cos(np.radians(10.0)))  # from the north


def test_profile_wind_rejects_unsorted_altitudes():
    with pytest.raises(ValueError):
        ProfileWind(altitudes=[0.0, 100.0, 50.0], velocities=np.zeros((3, 3)))


def test_profile_wind_rejects_wrong_velocity_shape():
    with pytest.raises(ValueError):
        ProfileWind(altitudes=[0.0, 100.0], velocities=np.zeros((2, 2)))


def test_constant_wind_rejects_wrong_vector_size():
    with pytest.raises(ValueError):
        ConstantWind([1.0, 2.0])
