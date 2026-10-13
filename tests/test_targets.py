"""Tests for the scripted targets (closed-form threat paths)."""

import numpy as np
import pytest

from jineta.jinete.targets import StraightLineTarget, TurningTarget, track_positions


def test_straight_line():
    target = StraightLineTarget(start=[100.0, 0.0, -50.0], velocity=[-20.0, 5.0, 0.0])
    position, velocity = target(3.0)
    assert np.allclose(position, [40.0, 15.0, -50.0])
    assert np.allclose(velocity, [-20.0, 5.0, 0.0])


def test_turn_keeps_speed_and_altitude():
    target = TurningTarget(start=[0, 0, -80.0], speed=25.0, heading=0.3, turn_rate=0.2)
    for t in np.linspace(0.0, 30.0, 7):
        position, velocity = target(t)
        assert np.isclose(np.linalg.norm(velocity), 25.0)
        assert np.isclose(position[2], -80.0)
        assert velocity[2] == 0.0


@pytest.mark.parametrize("turn_rate", [0.2, -0.2])
def test_turn_is_a_circle_of_radius_speed_over_turn_rate(turn_rate):
    """Heading North, a right turn (> 0) has its centre to the East, a left turn to
    the West, at a distance V / |w|."""
    speed = 20.0
    target = TurningTarget(
        start=[0, 0, -50.0], speed=speed, heading=0.0, turn_rate=turn_rate
    )
    radius = speed / abs(turn_rate)
    centre = np.array([0.0, np.sign(turn_rate) * radius, -50.0])
    for t in np.linspace(0.0, 20.0, 9):
        assert np.isclose(np.linalg.norm(target(t)[0] - centre), radius)


def test_full_turn_comes_back_to_the_start():
    target = TurningTarget(
        start=[10, 20, -30.0], speed=15.0, heading=1.0, turn_rate=0.5
    )
    position, _ = target(2 * np.pi / 0.5)
    assert np.allclose(position, [10, 20, -30.0])


def test_velocity_is_the_derivative_of_the_position():
    """Central differences of the closed-form position give the velocity."""
    target = TurningTarget(start=[0, 0, -50.0], speed=30.0, heading=-0.7, turn_rate=0.3)
    h = 1e-5
    for t in (0.0, 4.0, 11.0):
        numerical = (target(t + h)[0] - target(t - h)[0]) / (2 * h)
        assert np.allclose(numerical, target(t)[1], atol=1e-5)


def test_zero_turn_rate_is_a_straight_line():
    turning = TurningTarget(
        start=[5, 5, -40.0], speed=10.0, heading=np.pi / 2, turn_rate=0.0
    )
    straight = StraightLineTarget(start=[5, 5, -40.0], velocity=[0.0, 10.0, 0.0])
    times = np.linspace(0.0, 10.0, 5)
    assert np.allclose(
        track_positions(turning, times), track_positions(straight, times)
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: StraightLineTarget(start=[0.0, 0.0], velocity=[1.0, 0.0, 0.0]),
        lambda: StraightLineTarget(start=[0.0, 0.0, 0.0], velocity=[1.0]),
        lambda: TurningTarget(start=[0, 0, 0], speed=-1.0, heading=0.0, turn_rate=0.1),
    ],
)
def test_rejects_invalid_targets(build):
    with pytest.raises(ValueError):
        build()
