"""Tests for the wind models: conventions, interpolation and input validation."""

import numpy as np
import pytest

from jineta.campo.wind import (
    CombinedWind,
    ConstantWind,
    DiscreteGust,
    PowerLawWind,
    ProfileWind,
    meteorological_wind,
)
from jineta.core.point_mass import GRAVITY, PointMass


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
    wind = ConstantWind(np.array([3.0, -4.0, 0.0]))
    samples = [
        (0.0, position_at(-50.0)),
        (50.0, position_at(5_000.0)),
        (7.0, np.array([10_000.0, -20_000.0, -300.0])),
    ]
    for t, position in samples:
        assert np.array_equal(wind(t, position), [3.0, -4.0, 0.0])


def test_constant_wind_returns_a_copy():
    wind = ConstantWind(np.array([3.0, -4.0, 0.0]))
    wind(0.0, position_at(0.0))[0] = 99.0  # modify the returned array
    assert np.array_equal(wind(0.0, position_at(0.0)), [3.0, -4.0, 0.0])


def shear_wind() -> ProfileWind:
    """Wind table with easy numbers: calm at the ground, stronger and turning above."""
    return ProfileWind(
        altitudes=np.array([0.0, 100.0, 1_000.0]),
        velocities=np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-3.0, 10.0, 1.0]]),
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
        altitudes=np.array([0.0, 100.0]),
        velocities=np.array(
            [meteorological_wind(10.0, 350.0), meteorological_wind(10.0, 10.0)]
        ),
    )
    halfway = wind(0.0, position_at(50.0))
    assert np.isclose(halfway[1], 0.0, atol=1e-12)  # no east component
    assert np.isclose(halfway[0], -10.0 * np.cos(np.radians(10.0)))  # from the north


def test_profile_wind_plugs_into_point_mass():
    """PointMass reads the wind at altitude = -z: moving with that wind, no drag."""
    wind = shear_wind()
    altitude = 550.0
    air_velocity = wind(0.0, position_at(altitude))  # [-1.5, 7.5, 0.5]
    body = PointMass(mass=1.0, drag_area=0.05, wind=wind)
    state = np.concatenate([position_at(altitude), air_velocity])

    accel = body.derivative(0.0, state)[3:6]

    assert np.allclose(accel, [0.0, 0.0, GRAVITY])


def test_profile_wind_rejects_unsorted_altitudes():
    with pytest.raises(ValueError):
        ProfileWind(altitudes=np.array([0.0, 100.0, 50.0]), velocities=np.zeros((3, 3)))


def test_profile_wind_rejects_wrong_velocity_shape():
    with pytest.raises(ValueError):
        ProfileWind(altitudes=np.array([0.0, 100.0]), velocities=np.zeros((2, 2)))


def test_constant_wind_rejects_wrong_vector_size():
    with pytest.raises(ValueError):
        ConstantWind(np.array([1.0, 2.0]))


def test_power_law_wind_has_the_reference_speed_at_the_reference_height():
    wind = PowerLawWind(reference_speed=6.0, direction_from_deg=270.0)
    assert np.isclose(np.linalg.norm(wind(0.0, position_at(10.0))), 6.0)


def test_power_law_wind_follows_the_exponent():
    # With exponent 1/2, four times the reference height means twice the speed
    wind = PowerLawWind(3.0, 0.0, reference_height=10.0, exponent=0.5)
    assert np.isclose(np.linalg.norm(wind(0.0, position_at(40.0))), 6.0)


def test_power_law_wind_blows_toward_the_opposite_of_its_direction():
    wind = PowerLawWind(reference_speed=5.0, direction_from_deg=270.0)  # from the west
    assert np.allclose(wind(0.0, position_at(10.0)), [0.0, 5.0, 0.0], atol=1e-12)


def test_power_law_wind_is_zero_at_and_below_the_ground():
    wind = PowerLawWind(5.0, 270.0)
    for altitude in (0.0, -30.0):
        assert np.array_equal(wind(0.0, position_at(altitude)), np.zeros(3))


def test_power_law_wind_is_constant_above_the_gradient_height():
    wind = PowerLawWind(5.0, 270.0, gradient_height=300.0)
    at_gradient_height = wind(0.0, position_at(300.0))
    assert np.array_equal(wind(0.0, position_at(5_000.0)), at_gradient_height)
    expected_speed = 5.0 * (300.0 / 10.0) ** (1.0 / 7.0)
    assert np.isclose(np.linalg.norm(at_gradient_height), expected_speed)


def test_power_law_wind_with_zero_exponent_is_uniform():
    wind = PowerLawWind(5.0, 90.0, exponent=0.0)
    for altitude in (0.0, 50.0, 400.0):
        assert np.isclose(np.linalg.norm(wind(0.0, position_at(altitude))), 5.0)


@pytest.mark.parametrize(
    "invalid",
    [
        {"reference_speed": -1.0},
        {"reference_height": 0.0},
        {"exponent": -0.1},
        {"gradient_height": 0.0},
    ],
)
def test_power_law_wind_rejects_invalid_parameters(invalid):
    parameters = {"reference_speed": 5.0, "direction_from_deg": 270.0}
    with pytest.raises(ValueError):
        PowerLawWind(**(parameters | invalid))


def sample_gust() -> DiscreteGust:
    """A 4 s gust that starts at t = 10 s and peaks at 8 m/s toward the east."""
    return DiscreteGust(peak=np.array([0.0, 8.0, 0.0]), start_time=10.0, duration=4.0)


def test_gust_is_zero_outside_its_duration():
    gust = sample_gust()
    for t in (0.0, 9.9, 14.1, 100.0):
        assert np.array_equal(gust(t, position_at(0.0)), np.zeros(3))


def test_gust_starts_and_ends_at_zero():
    gust = sample_gust()
    assert np.allclose(gust(10.0, position_at(0.0)), 0.0, atol=1e-12)
    assert np.allclose(gust(14.0, position_at(0.0)), 0.0, atol=1e-12)


def test_gust_reaches_the_peak_in_the_middle():
    assert np.allclose(sample_gust()(12.0, position_at(0.0)), [0.0, 8.0, 0.0])


def test_gust_is_half_the_peak_at_a_quarter_of_its_duration():
    # 0.5 * (1 - cos(pi / 2)) = 0.5
    assert np.allclose(sample_gust()(11.0, position_at(0.0)), [0.0, 4.0, 0.0])


def test_gust_is_symmetric_in_time():
    gust = sample_gust()
    early = gust(10.0 + 1.3, position_at(0.0))
    late = gust(14.0 - 1.3, position_at(0.0))
    assert np.allclose(early, late)


def test_gust_average_is_half_the_peak():
    gust = sample_gust()
    times = np.linspace(10.0, 14.0, 10_001)
    mean = np.mean([gust(t, position_at(0.0)) for t in times], axis=0)
    assert np.allclose(mean, [0.0, 4.0, 0.0], atol=1e-3)


def test_gust_does_not_depend_on_position():
    gust = sample_gust()
    far_away = np.array([5_000.0, -8_000.0, -400.0])
    assert np.array_equal(gust(12.0, far_away), gust(12.0, position_at(0.0)))


def test_gust_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        DiscreteGust(peak=np.array([0.0, 8.0, 0.0]), start_time=0.0, duration=0.0)


def test_gust_rejects_wrong_peak_size():
    with pytest.raises(ValueError):
        DiscreteGust(peak=np.array([1.0, 2.0]), start_time=0.0, duration=1.0)


def test_combined_wind_adds_its_components():
    mean = ConstantWind(np.array([3.0, 1.0, 0.0]))
    combined = CombinedWind([mean, sample_gust()])
    assert np.allclose(combined(0.0, position_at(0.0)), [3.0, 1.0, 0.0])  # no gust yet
    assert np.allclose(combined(12.0, position_at(0.0)), [3.0, 9.0, 0.0])  # gust peak


def test_combined_wind_without_models_is_calm():
    assert np.array_equal(CombinedWind([])(0.0, position_at(0.0)), np.zeros(3))
