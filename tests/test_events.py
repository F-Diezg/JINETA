"""Tests for the closest approach interpolated inside a time step."""

import numpy as np
import pytest

from jineta.core.events import closest_approach
from jineta.jinete.targets import TurningTarget, track_positions


def straight_flyby(times):
    """Relative position r(t) = r0 + v t, closest at t* = -r0.v / v.v = 1.2345 s,
    at a distance of exactly 3 m (r0 and v chosen so)."""
    v = np.array([-10.0, 0.0, 0.0])
    r0 = np.array([12.345, 3.0, 0.0])
    return r0 + np.outer(times, v)


def test_straight_flyby_is_exact_between_samples():
    times = np.arange(0.0, 3.0 + 1e-9, 0.1)
    result = closest_approach(times, straight_flyby(times))
    assert np.isclose(result.time, 1.2345, atol=1e-12)
    assert np.isclose(result.distance, 3.0, atol=1e-12)
    assert times[result.step] <= result.time <= times[result.step + 1]


def test_interpolation_beats_the_nearest_sample():
    times = np.arange(0.0, 3.0 + 1e-9, 0.1)
    nearest_sample = np.linalg.norm(straight_flyby(times), axis=1).min()
    assert nearest_sample > 3.0 + 0.01  # the samples miss the true minimum
    assert closest_approach(times, straight_flyby(times)).distance < nearest_sample


@pytest.mark.parametrize(
    "velocity, expected_time",
    [
        ([10.0, 0.0, 0.0], 0.0),  # moving apart from the start: closest at t = 0
        ([-1.0, 0.0, 0.0], 3.0),  # still closing at the end: closest at the last time
    ],
)
def test_closest_at_the_ends(velocity, expected_time):
    times = np.linspace(0.0, 3.0, 31)
    r = np.array([20.0, 1.0, 0.0]) + np.outer(times, velocity)
    assert np.isclose(closest_approach(times, r).time, expected_time)


def test_close_pass_on_a_curved_path():
    """A turning target (30 m/s) passes 0.5 m above an observer at t = 1.2345 s, so
    the true miss is 0.5 m exactly. Near a close pass the distance is V-shaped, and
    samples 0.1 s apart can miss the bottom by up to v dt / 2; the straight line
    inside the step follows the V and lands within a fraction of a millimetre."""
    target = TurningTarget(
        start=[-60.0, 8.0, -50.0], speed=30.0, heading=0.0, turn_rate=0.15
    )
    observer = target(1.2345)[0] + np.array([0.0, 0.0, 0.5])
    times = np.arange(0.0, 4.0 + 1e-9, 0.1)
    r = track_positions(target, times) - observer

    result = closest_approach(times, r)
    assert abs(result.distance - 0.5) < 1e-4
    assert abs(result.time - 1.2345) < 1e-5
    assert np.linalg.norm(r, axis=1).min() - 0.5 > 0.1  # nearest sample: way off


def test_no_relative_motion():
    times = np.linspace(0.0, 1.0, 5)
    r = np.tile([3.0, 4.0, 0.0], (5, 1))
    result = closest_approach(times, r)
    assert np.isclose(result.distance, 5.0)
    assert result.time == 0.0


@pytest.mark.parametrize(
    "times, positions",
    [
        ([0.0], [[1.0, 0.0, 0.0]]),  # a single sample
        ([0.0, 1.0], [[1.0, 0.0], [2.0, 0.0]]),  # 2 components
        ([0.0, 1.0, 2.0], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),  # lengths differ
    ],
)
def test_rejects_invalid_input(times, positions):
    with pytest.raises(ValueError):
        closest_approach(times, positions)
