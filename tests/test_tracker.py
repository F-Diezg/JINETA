"""Tests for the tracks: the EKF on spherical measurements, against known results."""

import math

import numpy as np
import pytest
from scipy.stats import chi2

from jineta.atalaya.kalman import constant_velocity_transition, nees
from jineta.atalaya.kalman import white_noise_acceleration as cwna
from jineta.atalaya.profiles import SHORT_RANGE_RADAR
from jineta.atalaya.radar import Measurement, Radar, RadarParams, spherical
from jineta.atalaya.tracker import (
    Track,
    measurement_jacobian,
    position_covariance,
    state_jacobian,
)
from jineta.jinete.targets import TurningTarget

ORIGIN = np.zeros(3)


def exact(target: str, t: float, position, site=ORIGIN, R=None) -> Measurement:
    """A measurement with no noise, but with the noise covariance R the filter is
    told (a filter cannot be told that a measurement is perfect: S would be
    singular)."""
    site = np.asarray(site, dtype=float)
    r, az, el = spherical(np.asarray(position, dtype=float) - site)
    if R is None:
        R = SHORT_RANGE_RADAR.noise_covariance
    return Measurement(target, t, site, r, az, el, R)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    h = 1e-4
    for offset in rng.uniform(-1500, 1500, size=(50, 3)):
        numeric = np.array(
            [
                (np.array(spherical(offset + h * e)) - spherical(offset - h * e))
                / (2 * h)
                for e in np.eye(3)
            ]
        ).T
        assert np.allclose(measurement_jacobian(offset), numeric, rtol=1e-6, atol=0)


def test_state_jacobian_does_not_see_the_velocity():
    site = np.array([10.0, 20.0, 0.0])
    x = np.array([500.0, 300.0, -100.0, 12.0, -3.0, 1.0])
    H = state_jacobian(site, x)
    assert H.shape == (3, 6)
    assert np.array_equal(H[:, 3:], np.zeros((3, 3)))
    assert np.allclose(H[:, :3], measurement_jacobian(x[:3] - site))


def test_measured_point_covariance_is_long_across_the_line_of_sight():
    """Due North at 1 km on the horizon: sigma_range along North, r sigma_azimuth
    East, r sigma_elevation up and down."""
    m = exact("t", 0.0, [1000.0, 0.0, 0.0])
    p = SHORT_RANGE_RADAR
    expected = np.diag(
        [
            p.range_noise**2,
            (1000 * p.azimuth_noise) ** 2,
            (1000 * p.elevation_noise) ** 2,
        ]
    )
    assert np.allclose(position_covariance(m), expected, rtol=1e-12, atol=1e-12)


def test_a_track_starts_with_one_measurement():
    """The first measurement gives the position, with that point's covariance, and
    a still target with the speed prior on each velocity component."""
    track = Track("drone", process_noise=1.0, speed_prior=30.0)
    assert not track.ready
    position = np.array([800.0, 300.0, -120.0])
    first = exact("drone", 2.0, position)
    track.update(first)
    assert track.ready and track.updates == 1 and track.time == 2.0
    assert np.allclose(track.x, [*position, 0.0, 0.0, 0.0], rtol=0, atol=1e-9)
    assert np.allclose(track.P[:3, :3], position_covariance(first))
    assert np.array_equal(track.P[:3, 3:], np.zeros((3, 3)))
    assert np.allclose(track.P[3:, 3:], 900.0 * np.eye(3))


def test_perfect_measurements_of_a_straight_line_bring_the_track_onto_it():
    """Starting from a still target, the track learns the velocity from the
    measurements: after 10 s it is on the line to the millimetre, and it predicts
    along it between measurements."""
    track = Track("drone", process_noise=2.0, speed_prior=30.0)
    start, velocity = np.array([1200.0, -400.0, -90.0]), np.array([-20.0, 10.0, 0.0])
    for k in range(100):
        track.update(exact("drone", 0.1 * k, start + 0.1 * k * velocity))
    assert np.linalg.norm(track.x[:3] - (start + 9.9 * velocity)) < 1e-3
    assert np.linalg.norm(track.x[3:] - velocity) < 1e-3
    later = 9.9 + 0.35
    assert np.linalg.norm(track.position(later) - (start + later * velocity)) < 2e-3
    assert np.linalg.norm(track.velocity(later) - velocity) < 1e-3


def test_with_no_process_noise_the_track_is_the_least_squares_line():
    """A target moving straight away from the radar along North: the range noise
    falls entirely on North. With q = 0 the filter is the least-squares fit of a
    straight line to the N ranges, whose variance at the last point is
        sigma^2 * 2 (2N - 1) / (N (N + 1)).
    The speed prior is made huge, so the start adds no information. The filter's
    own P says so (the median within 0.5 %, every run within 2 %: each is
    linearised around a slightly wrong point) and so does the scatter of 400
    seeded runs (within 15 %; the sampling error of a variance from 400 runs is
    about 7 %)."""
    params = RadarParams(2.0, math.radians(0.5), math.radians(1.0), 10.0, 5000.0)
    n, runs = 30, 400
    least_squares = 2.0**2 * 2 * (2 * n - 1) / (n * (n + 1))
    errors, variances = [], []
    for seed in range(runs):
        radar = Radar(params, ORIGIN, seed=seed)
        track = Track("t", process_noise=0.0, speed_prior=1e4)
        for k in range(n):
            truth = np.array([800.0 + 20.0 * 0.1 * k, 0.0, 0.0])
            track.update(radar.measure("t", 0.1 * k, truth))
        errors.append(track.x[0] - truth[0])
        variances.append(track.P[0, 0])
    assert np.median(variances) == pytest.approx(least_squares, rel=0.005)
    assert np.allclose(variances, least_squares, rtol=0.02)
    assert np.var(errors) == pytest.approx(least_squares, rel=0.15)


def test_the_filter_is_consistent_on_a_noisy_target():
    """The truth moves with white-noise acceleration (q = 1) 0.8 to 1.5 km from the
    radar, measured with the short-range profile; the filter is told the same q.
    Its average NEES over 200 seeded runs stays inside the 99 % chi-square interval
    at every checked time: its covariance is honest also through the
    linearisation. (In the first few updates it is below the interval: the speed
    prior of 30 m/s is wider than the true 10 m/s, which errs on the safe side.)"""
    q, dt, runs, steps = 1.0, 0.1, 200, 80
    checked = (10, 40, 79)
    F = constant_velocity_transition(dt)
    chol = np.linalg.cholesky(cwna(dt, q))
    rng = np.random.default_rng(99)
    total = np.zeros(len(checked))
    for seed in range(runs):
        radar = Radar(SHORT_RANGE_RADAR, ORIGIN, seed=1000 + seed)
        track = Track("t", process_noise=q, speed_prior=30.0)
        truth = np.array([900.0, 500.0, -150.0, -8.0, 6.0, 0.0])
        for k in range(steps):
            if k:
                truth = F @ truth + chol @ rng.standard_normal(6)
            track.update(radar.measure("t", k * dt, truth[:3]))
            if k in checked:
                total[checked.index(k)] += nees(track.x - truth, track.P)
    average = total / runs
    low, high = chi2.ppf(0.005, 6 * runs) / runs, chi2.ppf(0.995, 6 * runs) / runs
    assert np.all((average > low) & (average < high)), (low, average, high)


def test_the_track_beats_the_raw_measurements():
    """On a straight line 1 km away the filtered position error (RMS over the last
    3 s of 10 s, 50 runs) is several times smaller than that of the measured
    points themselves."""
    q, raw, filtered = 0.5, [], []
    for seed in range(50):
        radar = Radar(SHORT_RANGE_RADAR, ORIGIN, seed=seed)
        track = Track("t", process_noise=q, speed_prior=30.0)
        for k in range(100):
            truth = np.array([1000.0 - 10.0 * 0.1 * k, 300.0, -120.0])
            m = radar.measure("t", 0.1 * k, truth)
            track.update(m)
            if k >= 70:
                raw.append(np.sum((m.position - truth) ** 2))
                filtered.append(np.sum((track.x[:3] - truth) ** 2))
    assert math.sqrt(np.mean(filtered)) < math.sqrt(np.mean(raw)) / 3


def test_a_turning_target_is_tracked_with_a_lag_that_a_larger_q_reduces():
    """A constant-velocity filter on a target in a steady turn (centripetal
    acceleration V w = 1.5 m/s^2) stays behind it on the outside of the turn, as in
    the 1-D alpha-beta case; telling the filter a larger q shrinks that lag. Perfect
    measurements, so only the bias is left."""
    turn = TurningTarget(
        [900.0, -300.0, -100.0], speed=15.0, heading=1.0, turn_rate=0.1
    )

    def outward_lag(q: float) -> float:
        track = Track("t", process_noise=q, speed_prior=30.0)
        for k in range(300):
            t = 0.1 * k
            track.update(exact("t", t, turn(t)[0]))
        error = track.x[:3] - turn(t)[0]
        heading = 1.0 + 0.1 * t
        right = np.array([-np.sin(heading), np.cos(heading), 0.0])  # turn centre side
        return float(-error @ right)  # > 0: displaced away from the centre

    small, large = outward_lag(0.1), outward_lag(10.0)
    assert small > 0.1 and large > 0.0
    assert large < small / 4


def test_azimuth_innovation_is_wrapped_across_south():
    """A target crossing due South of the radar has its azimuth jump from +180 to
    -180 deg (at t = 5 s); the innovation is wrapped, so the track does not
    notice: once it has learnt the velocity it stays within 0.1 m of the truth."""
    track = Track("t", process_noise=1.0, speed_prior=30.0)
    for k in range(100):
        t = 0.1 * k
        truth = np.array([-700.0, 40.0 - 8.0 * t, -80.0])  # East to West, behind it
        track.update(exact("t", t, truth))
        if t >= 3.0:
            assert np.linalg.norm(track.x[:3] - truth) < 0.1
    assert truth[1] < 0  # it did cross South


def test_estimate_does_not_change_the_track():
    track = Track("t", process_noise=1.0, speed_prior=30.0)
    track.update(exact("t", 0.0, [500.0, 0.0, -50.0]))
    track.update(exact("t", 0.1, [499.0, 0.5, -50.0]))
    x, P = track.x.copy(), track.P.copy()
    _, P_later = track.estimate(3.0)
    assert np.array_equal(track.x, x) and np.array_equal(track.P, P)
    assert np.all(np.diag(P_later) > np.diag(P))  # the uncertainty grows ahead


@pytest.mark.parametrize(
    "call",
    [
        lambda: Track("t", process_noise=-1.0, speed_prior=30.0),
        lambda: Track("t", process_noise=1.0, speed_prior=0.0),
        lambda: Track("t", 1.0, 30.0).estimate(0.0),  # not started
        lambda: Track("t", 1.0, 30.0).update(exact("other", 0.0, [500.0, 0.0, 0.0])),
        lambda: measurement_jacobian([0.0, 0.0, -100.0]),  # straight above
    ],
)
def test_rejects_invalid_input(call):
    with pytest.raises(ValueError):
        call()


def test_rejects_measurements_out_of_order_and_estimates_in_the_past():
    track = Track("t", process_noise=1.0, speed_prior=30.0)
    track.update(exact("t", 1.0, [500.0, 0.0, -50.0]))
    with pytest.raises(ValueError):
        track.update(exact("t", 1.0, [500.0, 0.0, -50.0]))
    track.update(exact("t", 1.1, [499.0, 0.0, -50.0]))
    with pytest.raises(ValueError):
        track.estimate(1.0)
