"""Tests for the ground radar: geometry, noise statistics, schedule and range."""

import math

import numpy as np
import pytest

from jineta.atalaya.profiles import RADAR_PROFILES, SHORT_RANGE_RADAR
from jineta.atalaya.radar import (
    Measurement,
    Radar,
    RadarParams,
    cartesian,
    spherical,
    wrap_angle,
)

SITE = np.array([100.0, -50.0, 0.0])  # on the ground


def noise_free(rate: float = 10.0, max_range: float = 5000.0) -> RadarParams:
    return RadarParams(0.0, 0.0, 0.0, rate=rate, max_range=max_range)


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        ([100.0, 0.0, 0.0], (100.0, 0.0, 0.0)),  # North, on the horizon
        ([0.0, 100.0, 0.0], (100.0, math.pi / 2, 0.0)),  # East
        ([0.0, -100.0, 0.0], (100.0, -math.pi / 2, 0.0)),  # West
        ([-100.0, 0.0, 0.0], (100.0, math.pi, 0.0)),  # South
        ([0.0, 0.0, -100.0], (100.0, 0.0, math.pi / 2)),  # straight up
        ([300.0, 0.0, -400.0], (500.0, 0.0, math.atan2(400, 300))),  # 3-4-5, above
        ([100.0, 100.0, 50.0], (150.0, math.pi / 4, -math.asin(50 / 150))),  # below
    ],
)
def test_spherical_coordinates_of_known_directions(offset, expected):
    assert np.allclose(spherical(offset), expected, rtol=0, atol=1e-12)


def test_cartesian_is_the_inverse_of_spherical():
    rng = np.random.default_rng(3)
    for offset in rng.uniform(-2000, 2000, size=(100, 3)):
        assert np.allclose(cartesian(*spherical(offset)), offset, rtol=0, atol=1e-9)


def test_wrap_angle():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-math.pi) == math.pi
    assert wrap_angle(math.pi) == math.pi
    assert wrap_angle(0.25) == 0.25


def test_without_noise_the_measurement_is_the_geometry():
    radar = Radar(noise_free(), SITE)
    target = np.array([900.0, 400.0, -120.0])
    m = radar.measure("drone", 2.5, target)
    assert isinstance(m, Measurement)
    assert (m.target, m.time) == ("drone", 2.5)
    assert np.allclose(m.values, spherical(target - SITE), rtol=0, atol=1e-12)
    assert np.allclose(m.position, target, rtol=0, atol=1e-9)


def test_noise_has_the_mean_and_spread_of_the_profile():
    """20 000 measurements of a still point: each of range, azimuth and elevation
    is centred on the truth (within 4 standard errors) with the profile's spread
    (within 3 %; the spread of a sample standard deviation is about 0.5 % here),
    and the three are uncorrelated."""
    radar = Radar(SHORT_RANGE_RADAR, SITE, seed=11)
    target = SITE + np.array([600.0, 300.0, -150.0])
    truth = np.array(spherical(target - SITE))
    n = 20_000
    values = np.array([radar.measure("x", 0.0, target).values for _ in range(n)])
    sigma = np.sqrt(np.diag(SHORT_RANGE_RADAR.noise_covariance))
    assert np.all(np.abs(values.mean(axis=0) - truth) < 4 * sigma / math.sqrt(n))
    assert np.allclose(values.std(axis=0), sigma, rtol=0.03)
    correlation = np.corrcoef(values.T)
    assert np.all(np.abs(correlation[np.triu_indices(3, 1)]) < 0.03)


@pytest.mark.parametrize("distance", [100.0, 1000.0])
def test_the_sideways_error_grows_with_the_range(distance):
    """For a target due North on the horizon, the measured point scatters East by
    r * sigma_azimuth, up and down by r * sigma_elevation, and North by sigma_range
    (within 3 %): the angle errors become distances that grow with the range."""
    params = RadarParams(2.0, math.radians(0.5), math.radians(1.0), 10.0, 5000.0)
    radar = Radar(params, SITE, seed=5)
    target = SITE + np.array([distance, 0.0, 0.0])
    points = np.array([radar.measure("x", 0.0, target).position for _ in range(20_000)])
    spread = points.std(axis=0)
    expected = [2.0, distance * math.radians(0.5), distance * math.radians(1.0)]
    assert np.allclose(spread, expected, rtol=0.03)


def test_scans_at_its_own_rate():
    """At 10 Hz in a 100 Hz loop it measures every 10th step: t = 0, 0.1, ... with
    the times accumulated as the simulation does (0.01 + 0.01 + ...)."""
    radar = Radar(noise_free(rate=10.0), SITE)
    target = {"drone": SITE + np.array([500.0, 0.0, -100.0])}
    scan_times = []
    t = 0.0
    for _ in range(300):
        if radar.scan(t, target):
            scan_times.append(t)
        t += 0.01
    assert len(scan_times) == 30
    assert np.allclose(scan_times, np.arange(30) * 0.1, rtol=0, atol=1e-9)


def test_a_rate_that_does_not_divide_the_step_measures_at_the_next_step():
    """At 13 Hz with 0.01 s steps the scan due at k / 13 happens at the first step
    at or after it: 130 scans in 10 s, each less than one step late."""
    radar = Radar(noise_free(rate=13.0), SITE)
    target = {"drone": SITE + np.array([500.0, 0.0, -100.0])}
    scan_times = [k * 0.01 for k in range(1000) if radar.scan(k * 0.01, target)]
    assert len(scan_times) == 130
    lateness = np.array(scan_times) - np.arange(130) / 13.0
    assert np.all((lateness > -1e-9) & (lateness < 0.01))


def test_a_long_step_skips_the_scans_in_between():
    radar = Radar(noise_free(rate=10.0), SITE)
    target = {"drone": SITE + np.array([500.0, 0.0, -100.0])}
    assert radar.scan(0.0, target)
    assert radar.scan(0.55, target)  # 0.1 ... 0.5 were missed: one scan, not five
    assert not radar.scan(0.58, target)
    assert radar.next_time == pytest.approx(0.6)


def test_measures_only_what_is_in_range():
    radar = Radar(noise_free(max_range=1000.0), SITE)
    near = SITE + np.array([600.0, 0.0, -100.0])
    far = SITE + np.array([1500.0, 0.0, -100.0])
    found = radar.scan(0.0, {"near": near, "far": far})
    assert [m.target for m in found] == ["near"]
    assert radar.measure("edge", 0.0, SITE + np.array([1000.0, 0.0, 0.0])) is not None
    assert radar.measure("out", 0.0, SITE + np.array([1000.001, 0.0, 0.0])) is None


def test_the_same_seed_repeats_the_run_and_another_does_not():
    target = {
        "a": SITE + np.array([700.0, 200.0, -90.0]),
        "b": SITE + np.array([-300.0, 800.0, -60.0]),
    }

    def run(radar: Radar) -> np.ndarray:
        found = []
        for k in range(50):
            found += radar.scan(k * 0.01, target)
        return np.array([m.values for m in found])

    radar = Radar(SHORT_RANGE_RADAR, SITE, seed=42)
    first = run(radar)
    radar.reset()
    assert np.array_equal(run(radar), first)
    assert np.array_equal(run(Radar(SHORT_RANGE_RADAR, SITE, seed=42)), first)
    assert not np.array_equal(run(Radar(SHORT_RANGE_RADAR, SITE, seed=43)), first)


def test_azimuth_stays_in_range_behind_the_radar():
    """Due South the azimuth is +-pi; with noise it must stay in (-pi, pi]."""
    radar = Radar(SHORT_RANGE_RADAR, SITE, seed=8)
    target = SITE + np.array([-800.0, 0.0, -100.0])
    azimuths = np.array([radar.measure("x", 0.0, target).azimuth for _ in range(2000)])
    assert np.all((azimuths > -math.pi) & (azimuths <= math.pi))
    assert np.any(azimuths > 0) and np.any(azimuths < 0)  # both sides of South


def test_short_range_profile():
    assert RADAR_PROFILES["short_range"] is SHORT_RANGE_RADAR
    p = SHORT_RANGE_RADAR
    assert (p.range_noise, p.rate, p.max_range) == (1.0, 10.0, 2000.0)
    assert math.degrees(p.azimuth_noise) == pytest.approx(0.5)
    assert math.degrees(p.elevation_noise) == pytest.approx(1.0)
    assert np.allclose(
        np.diag(p.noise_covariance),
        [1.0, math.radians(0.5) ** 2, math.radians(1.0) ** 2],
    )
    with pytest.raises(AttributeError):
        p.rate = 20.0  # frozen


@pytest.mark.parametrize(
    "build",
    [
        lambda: RadarParams(-1.0, 0.01, 0.01, 10.0, 1000.0),
        lambda: RadarParams(1.0, -0.01, 0.01, 10.0, 1000.0),
        lambda: RadarParams(1.0, 0.01, 0.01, 0.0, 1000.0),
        lambda: RadarParams(1.0, 0.01, 0.01, 10.0, 0.0),
        lambda: Radar(SHORT_RANGE_RADAR, [0.0, 0.0]),
        lambda: Radar(SHORT_RANGE_RADAR, SITE).measure("x", 0.0, [1.0, 2.0]),
        lambda: spherical([1.0, 2.0]),
    ],
)
def test_rejects_invalid_input(build):
    with pytest.raises(ValueError):
        build()
