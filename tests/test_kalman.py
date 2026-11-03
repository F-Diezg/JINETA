"""Tests for the Kalman filter against results known in closed form."""

import numpy as np
import pytest
from scipy.stats import chi2

from jineta.atalaya.kalman import (
    alpha_beta_gains,
    constant_velocity_transition,
    nees,
    piecewise_white_acceleration,
    predict,
    update,
    white_noise_acceleration,
)

H_POSITION = np.hstack([np.eye(3), np.zeros((3, 3))])  # measures the position only


def test_constant_velocity_prediction_moves_the_position():
    x = np.array([1.0, 2.0, -3.0, 10.0, -5.0, 2.0])
    F = constant_velocity_transition(0.5)
    x_new, P_new = predict(x, np.eye(6), F, np.zeros((6, 6)))
    assert np.allclose(x_new, [6.0, -0.5, -2.0, 10.0, -5.0, 2.0])
    # With no process noise the position uncertainty grows with the velocity's
    # (var(x + v dt) = 1 + dt^2 for unit, independent errors).
    assert np.isclose(P_new[0, 0], 1.25)
    assert np.isclose(P_new[0, 3], 0.5)


def test_a_constant_measured_again_and_again_is_the_running_mean():
    """F = 1, Q = 0: the filter is the running mean, and its variance after k
    measurements is sigma^2 / k, exactly."""
    rng = np.random.default_rng(1)
    sigma = 3.0
    z = 7.0 + sigma * rng.standard_normal(50)
    x, P = np.array([z[0]]), np.array([[sigma**2]])  # starts at the first one
    for k in range(1, len(z)):
        x, P = predict(x, P, np.eye(1), np.zeros((1, 1)))
        x, P, _, _ = update(x, P, [z[k]], np.eye(1), [[sigma**2]])
        assert np.isclose(x[0], np.mean(z[: k + 1]), rtol=0, atol=1e-12)
        assert np.isclose(P[0, 0], sigma**2 / (k + 1), rtol=1e-12)


def test_sequential_updates_equal_one_stacked_update():
    """Two independent measurements, one after the other, give the same estimate as
    both at once (their information adds up). This is what lets several sensors
    feed one filter."""
    x0 = np.array([100.0, -50.0, -80.0, 5.0, 0.0, 1.0])
    P0 = np.diag([400.0, 400.0, 100.0, 25.0, 25.0, 9.0])
    z1, R1 = np.array([103.0, -48.0, -79.0]), np.diag([16.0, 9.0, 4.0])
    z2, R2 = np.array([101.0, -51.0, -82.0]), np.diag([1.0, 2.0, 3.0])
    x1, P1, _, _ = update(x0, P0, z1, H_POSITION, R1)
    x_seq, P_seq, _, _ = update(x1, P1, z2, H_POSITION, R2)
    H = np.vstack([H_POSITION, H_POSITION])
    R = np.block([[R1, np.zeros((3, 3))], [np.zeros((3, 3)), R2]])
    x_all, P_all, _, _ = update(x0, P0, np.r_[z1, z2], H, R)
    assert np.allclose(x_seq, x_all, rtol=0, atol=1e-10)
    assert np.allclose(P_seq, P_all, rtol=0, atol=1e-10)
    # Information form: P^-1 = P0^-1 + H^T R^-1 H.
    info = np.linalg.inv(P0) + H.T @ np.linalg.inv(R) @ H
    assert np.allclose(np.linalg.inv(P_all), info, rtol=1e-10)


def test_update_returns_the_innovation_and_does_not_touch_its_inputs():
    x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    P = 4.0 * np.eye(6)
    z = np.array([2.0, 2.0, 1.0])
    R = np.eye(3)
    copies = [a.copy() for a in (x, P, z, R)]
    x_new, P_new, y, S = update(x, P, z, H_POSITION, R)
    assert np.allclose(y, [1.0, 0.0, -2.0])
    assert np.allclose(S, 5.0 * np.eye(3))
    # Gain 4 / 5 on the measured components; the velocity is not touched (no
    # correlation with the position yet).
    assert np.allclose(x_new, [1.8, 2.0, 1.4, 0.0, 0.0, 0.0])
    assert np.allclose(np.diag(P_new), [0.8, 0.8, 0.8, 4.0, 4.0, 4.0])
    for original, after in zip(copies, (x, P, z, R), strict=True):
        assert np.array_equal(original, after)


def test_continuous_noise_does_not_depend_on_how_the_interval_is_split():
    """Predicting 0.1 s in one go or in ten 0.01 s pieces gives the same P with
    white-noise acceleration, but not with piecewise acceleration (one random
    acceleration per piece is a different model). The trackers predict at the
    simulation step between measurements, so they use the first one."""
    P0 = np.diag([25.0, 25.0, 25.0, 4.0, 4.0, 4.0])
    x0 = np.zeros(6)
    for model, splits_alike in (
        (lambda dt: white_noise_acceleration(dt, q=3.0), True),
        (lambda dt: piecewise_white_acceleration(dt, sigma=3.0), False),
    ):
        _, P_once = predict(x0, P0, constant_velocity_transition(0.1), model(0.1))
        P = P0
        for _ in range(10):
            _, P = predict(x0, P, constant_velocity_transition(0.01), model(0.01))
        assert np.allclose(P, P_once, rtol=1e-12, atol=1e-12) == splits_alike


@pytest.mark.parametrize("tracking_index", [0.05, 0.5, 5.0])
def test_steady_state_gain_is_the_alpha_beta_filter(tracking_index):
    """With piecewise white acceleration the 1-D filter settles at the gains of the
    alpha-beta filter, given in closed form by Kalata."""
    T, sigma_w = 0.1, 2.0
    sigma_a = tracking_index * sigma_w / T**2
    F = constant_velocity_transition(T, dimensions=1)
    Q = piecewise_white_acceleration(T, sigma_a, dimensions=1)
    H, R = np.array([[1.0, 0.0]]), np.array([[sigma_w**2]])
    x, P = np.zeros(2), 1e6 * np.eye(2)
    for _ in range(2000):
        x, P = predict(x, P, F, Q)
        x, P, _, _ = update(x, P, [0.0], H, R)
    _, P_pred = predict(x, P, F, Q)
    gain = P_pred @ H.T / (H @ P_pred @ H.T + R)
    alpha, beta = alpha_beta_gains(tracking_index)
    assert np.allclose(gain.ravel(), [alpha, beta / T], rtol=1e-9)


def test_an_accelerating_target_is_tracked_with_the_alpha_beta_lag():
    """A constant-velocity filter on a target with constant acceleration a settles
    at a fixed lag (noise-free measurements, so only the bias is left):
    predicted position a T^2 / beta, updated position (1 - alpha) a T^2 / beta,
    velocity (alpha - beta / 2) a T / beta, all behind the truth."""
    T, sigma_w, sigma_a, a = 0.1, 2.0, 3.0, 4.0
    F = constant_velocity_transition(T, dimensions=1)
    Q = piecewise_white_acceleration(T, sigma_a, dimensions=1)
    H, R = np.array([[1.0, 0.0]]), np.array([[sigma_w**2]])
    x, P = np.zeros(2), 100.0 * np.eye(2)
    for k in range(3000):
        t = k * T
        if k:
            x, P = predict(x, P, F, Q)
        predicted = x.copy()
        x, P, _, _ = update(x, P, [0.5 * a * t**2], H, R)
    alpha, beta = alpha_beta_gains(sigma_a * T**2 / sigma_w)
    assert np.isclose(0.5 * a * t**2 - predicted[0], a * T**2 / beta, rtol=1e-9)
    assert np.isclose(0.5 * a * t**2 - x[0], (1 - alpha) * a * T**2 / beta, rtol=1e-9)
    assert np.isclose(a * t - x[1], (alpha - beta / 2) * a * T / beta, rtol=1e-9)


def _nees_runs(q_filter: float, r_filter: float) -> np.ndarray:
    """Average NEES at a few times over 300 runs. The truth moves with white-noise
    acceleration of density q = 2 and the position is measured with sigma = 3 m
    every 0.1 s; the filter believes q_filter and r_filter (variance)."""
    rng = np.random.default_rng(2026)
    T, q, sigma, steps, runs = 0.1, 2.0, 3.0, 60, 300
    F = constant_velocity_transition(T)
    Q_true = white_noise_acceleration(T, q)
    Q_filter = white_noise_acceleration(T, q_filter)
    R_filter = r_filter * np.eye(3)
    chol = np.linalg.cholesky(Q_true)
    checked = (5, 20, 59)
    total = np.zeros(len(checked))
    for _ in range(runs):
        truth = np.array([500.0, 200.0, -100.0, -10.0, 5.0, 0.0])
        z = truth[:3] + sigma * rng.standard_normal(3)
        x = np.r_[z, 0.0, 0.0, 0.0]
        P = np.diag([sigma**2] * 3 + [20.0**2] * 3)
        x[3:] = truth[3:] + 20.0 * rng.standard_normal(3)  # an honest prior
        for k in range(1, steps):
            truth = F @ truth + chol @ rng.standard_normal(6)
            x, P = predict(x, P, F, Q_filter)
            z = truth[:3] + sigma * rng.standard_normal(3)
            x, P, _, _ = update(x, P, z, H_POSITION, R_filter)
            if k in checked:
                total[checked.index(k)] += nees(x - truth, P)
    return total / runs


def test_a_filter_with_the_right_model_is_consistent():
    """When the filter's model matches the truth, its covariance is honest: the
    average NEES over N runs is a chi-square with 6N degrees of freedom divided by
    N, inside its 99 % interval at every checked time."""
    runs = 300
    low = chi2.ppf(0.005, 6 * runs) / runs
    high = chi2.ppf(0.995, 6 * runs) / runs
    average = _nees_runs(q_filter=2.0, r_filter=9.0)
    assert np.all((average > low) & (average < high)), (low, average, high)


def test_a_filter_that_trusts_itself_too_much_fails_the_check():
    """The same check catches an overconfident filter (process and measurement
    noise both underestimated): its errors are far larger than its P says."""
    runs = 300
    high = chi2.ppf(0.995, 6 * runs) / runs
    average = _nees_runs(q_filter=0.02, r_filter=1.0)
    assert np.all(average > 2 * high)


def test_alpha_beta_gains_limits():
    """A sloppy sensor or a steady target (small index) smooths: alpha ~ sqrt(2
    lambda), beta ~ lambda. A precise sensor or an agile target trusts the
    measurement: alpha -> 1, beta -> 2."""
    alpha, beta = alpha_beta_gains(1e-6)
    assert np.isclose(alpha, np.sqrt(2e-6), rtol=1e-3)
    assert np.isclose(beta, 1e-6, rtol=1e-3)
    alpha, beta = alpha_beta_gains(1e6)
    assert np.isclose(alpha, 1.0, atol=1e-5)
    assert np.isclose(beta, 2.0, atol=1e-5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: predict(np.zeros(2), np.eye(3), np.eye(2), np.eye(2)),
        lambda: predict(np.zeros(2), np.eye(2), np.eye(2), np.zeros(2)),
        lambda: update(np.zeros(2), np.eye(2), [0.0], np.eye(2), [[1.0]]),
        lambda: update(np.zeros(2), np.eye(2), [0.0], [[1.0, 0.0]], np.eye(2)),
        lambda: constant_velocity_transition(-0.1),
        lambda: white_noise_acceleration(0.1, -1.0),
        lambda: piecewise_white_acceleration(-0.1, 1.0),
        lambda: alpha_beta_gains(0.0),
    ],
)
def test_rejects_invalid_input(call):
    with pytest.raises(ValueError):
        call()
