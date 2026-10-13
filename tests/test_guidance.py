"""Tests for the guidance laws, pure pursuit against its closed-form curve.

The pursuit curve (Bouguer). A pursuer of speed Vp always heads at a target that
flies a straight line at constant speed Vt. With
    r    the distance between them,
    phi  the angle between the line of sight (pursuer -> target) and the target
         velocity,
    k    = Vp / Vt,
the geometry gives
    dr/dt   = Vt cos(phi) - Vp
    dphi/dt = -Vt sin(phi) / r
Dividing one by the other and integrating,
    r = (C / 2) * s**(k - 1) * (1 + s**2),   s = tan(phi / 2)
with C constant along the whole trajectory. Two consequences are tested:
  - k > 1: the pursuer catches the target, at the time
        T = r0 (k + cos(phi0)) / (Vt (k**2 - 1))
  - k < 1: it never does. The separation is smallest when cos(phi) = k, which is
        r_min = C s*^(k - 1) / (1 + k),   s*^2 = (1 - k) / (1 + k)
"""

import numpy as np
import pytest

from jineta.core.events import closest_approach
from jineta.core.simulation import run
from jineta.jinete.guidance import pure_pursuit_velocity
from jineta.jinete.targets import StraightLineTarget, track_positions


def test_command_points_at_the_target_with_the_given_speed():
    command = pure_pursuit_velocity([0.0, 0.0, 0.0], [3.0, 0.0, -4.0], speed=10.0)
    assert np.allclose(command, [6.0, 0.0, -8.0])  # 3-4-5 triangle, 10 m/s


@pytest.mark.parametrize("distance", [0.01, 1.0, 1000.0])
def test_speed_does_not_depend_on_the_distance(distance):
    command = pure_pursuit_velocity(
        [5.0, 5.0, -50.0], [5.0, 5.0 + distance, -50.0], 7.0
    )
    assert np.isclose(np.linalg.norm(command), 7.0)
    assert np.allclose(command, [0.0, 7.0, 0.0])


def test_zero_command_when_the_positions_coincide():
    command = pure_pursuit_velocity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], speed=10.0)
    assert np.array_equal(command, np.zeros(3))


@pytest.mark.parametrize(
    "pursuer, target, speed",
    [
        ([0, 0, 0], [1, 0, 0], -1.0),  # negative speed
        ([0, 0], [1, 0, 0], 1.0),  # 2 components
        ([0, 0, 0], [1, 0, 0, 0], 1.0),  # 4 components
    ],
)
def test_rejects_invalid_input(pursuer, target, speed):
    with pytest.raises(ValueError):
        pure_pursuit_velocity(pursuer, target, speed)


# --- Against the closed-form pursuit curve -------------------------------------------


def pursue(speed, target, dt, t_end):
    """Ideal point pursuer (it obeys the command instantly) from the origin, using
    the same RK4 loop as the rest of the simulator. Returns the times and the
    relative position r = target - pursuer at each sample."""

    def derivative(t, position):
        return pure_pursuit_velocity(position, target(t)[0], speed)

    times, positions = run(derivative, np.zeros(3), dt, t_end)
    return times, track_positions(target, times) - positions


def invariant(distance, phi, k):
    """C = 2 r / (s**(k - 1) (1 + s**2)), s = tan(phi / 2): constant on the curve."""
    s = np.tan(phi / 2)
    return 2 * distance / (s ** (k - 1) * (1 + s**2))


def unit(*components):
    vector = np.array(components, dtype=float)
    return vector / np.linalg.norm(vector)


def capture_time(vp, vt, start, direction):
    """T = r0 (k + cos(phi0)) / (Vt (k**2 - 1))."""
    r0 = np.linalg.norm(start)
    cos_phi0 = np.dot(start, direction) / r0
    k = vp / vt
    return r0 * (k + cos_phi0) / (vt * (k**2 - 1))


# (pursuer speed, target speed, target start, target direction): phi0 = 90, ~20 and
# ~150 degrees, the last two with a vertical component (the curve is planar anyway).
CATCHABLE = [
    (30.0, 20.0, [100.0, 0.0, 0.0], unit(0, 1, 0)),
    (24.0, 8.0, [100.0, 0.0, 0.0], unit(np.cos(0.35), np.sin(0.35), 0)),
    (15.0, 10.0, [60.0, 40.0, -30.0], unit(-1, 0.3, 0.4)),
]


@pytest.mark.parametrize("vp, vt, start, direction", CATCHABLE)
def test_pursuit_curve_invariant_is_constant(vp, vt, start, direction):
    k = vp / vt
    target = StraightLineTarget(start=start, velocity=vt * direction)
    _, r = pursue(
        vp, target, dt=0.002, t_end=0.9 * capture_time(vp, vt, start, direction)
    )
    distance = np.linalg.norm(r, axis=1)
    phi = np.arccos(np.clip(r @ direction / distance, -1.0, 1.0))
    far = distance > 0.05 * distance[0]  # the last metres are very sensitive
    c = invariant(distance[far], phi[far], k)
    assert (c.max() - c.min()) / c.mean() < 1e-5


@pytest.mark.parametrize("vp, vt, start, direction", CATCHABLE)
def test_capture_time_matches_the_closed_form(vp, vt, start, direction):
    expected = capture_time(vp, vt, start, direction)
    target = StraightLineTarget(start=start, velocity=vt * direction)
    times, r = pursue(vp, target, dt=0.002, t_end=1.3 * expected)
    distance = np.linalg.norm(r, axis=1)
    first_inside = np.argmax(distance < 0.05)  # first sample within 5 cm
    assert distance[first_inside] < 0.05  # it did get there
    assert abs(times[first_inside] - expected) < 0.02


def test_a_faster_target_is_never_caught():
    """k = 0.5, target crossing at phi0 = 90 degrees: the separation falls to the
    closed-form minimum and then grows without bound."""
    vp, vt, start, direction = 10.0, 20.0, [100.0, 0.0, 0.0], unit(0, 1, 0)
    k = vp / vt
    target = StraightLineTarget(start=start, velocity=vt * direction)
    times, r = pursue(vp, target, dt=0.005, t_end=30.0)

    s0 = np.tan(np.pi / 4)  # phi0 = 90 degrees
    c = 2 * 100.0 / (s0 ** (k - 1) * (1 + s0**2))
    s_star = np.sqrt((1 - k) / (1 + k))
    r_min = c * s_star ** (k - 1) / (1 + k)

    result = closest_approach(times, r)
    assert abs(result.distance - r_min) < 1e-4 * r_min
    assert result.distance > 80.0  # nowhere near a capture (it is ~87.7 m)
    assert np.linalg.norm(r[-1]) > 100.0  # and it ends up farther than it started


def test_a_receding_faster_target_never_gets_closer():
    """Target flying straight away (phi0 = 0) and faster: the closest point is the
    start."""
    target = StraightLineTarget(start=[50.0, 0.0, 0.0], velocity=[20.0, 0.0, 0.0])
    times, r = pursue(15.0, target, dt=0.01, t_end=10.0)
    result = closest_approach(times, r)
    assert result.time == 0.0
    assert np.isclose(result.distance, 50.0)
