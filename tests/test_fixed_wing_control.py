"""Tests for the fixed-wing autopilot (successive loop closure).

The gains follow the design rules (surface used fully at the saturation error,
outer loops slower than inner ones). In closed loop, on the full 6-DOF model: the
autopilot holds the trim and cures the spiral instability, and steps in course,
altitude and airspeed settle without going past the bank and pitch limits or
moving a surface past its limit. A crosswind is held with the textbook crab angle.
The orbit guidance points along the circle on it and at it from far away, and the
autopilot following it holds the circle.
"""

import dataclasses

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.aero import air_data, body_airspeed
from jineta.jinete.fixed_wing import FixedWing
from jineta.jinete.fixed_wing_control import (
    FixedWingAutopilot,
    orbit_course,
    wrap_angle,
)
from jineta.jinete.profiles import (
    SMALL_FIXED_WING,
    SMALL_FIXED_WING_AUTOPILOT,
    SMALL_FIXED_WING_PROPULSION,
)
from jineta.jinete.trim import trim_level_flight

P = SMALL_FIXED_WING
PROP = SMALL_FIXED_WING_PROPULSION
DESIGN = SMALL_FIXED_WING_AUTOPILOT
DT = 0.01  # 100 Hz: the roll loop closes at 11-19 rad/s


def autopilot(speed=25.0, wind=None):
    plane = FixedWing(P, PROP) if wind is None else FixedWing(P, PROP, wind=wind)
    trim = trim_level_flight(plane, speed)
    return FixedWingAutopilot(plane, trim, DESIGN, DT)


def fly(pilot, duration, start=None):
    """Fly under the autopilot; return times, states and the commands used."""
    plane = pilot.vehicle
    commands = []

    def pre_step(t, state):
        pilot(t, state)
        s = plane.surfaces
        commands.append((s.elevator, s.aileron, s.rudder, plane.throttle))

    state = np.array(pilot.trim.state) if start is None else start
    times, states = run(
        plane.derivative,
        state,
        DT,
        duration,
        post_step=normalize_attitude,
        pre_step=pre_step,
    )
    return times, states, np.array(commands)


def euler(states):
    return np.array([quaternion.to_euler(q) for q in states[:, 6:10]])


def course(states):
    return np.arctan2(states[:, 4], states[:, 3])


def airspeed(pilot, times, states):
    return np.array(
        [
            air_data(body_airspeed(t, y, pilot.vehicle.wind)).speed
            for t, y in zip(times, states, strict=True)
        ]
    )


def assert_within_limits(states, commands):
    angles = euler(states)
    assert np.all(np.abs(angles[:, 0]) < DESIGN.max_bank + np.radians(2.0))
    assert np.all(np.abs(angles[:, 1]) < DESIGN.max_pitch + np.radians(1.0))
    assert np.all(np.abs(commands[:, :3]) <= DESIGN.surface_limit + 1e-12)
    assert np.all((commands[:, 3] >= 0.0) & (commands[:, 3] <= 1.0))


# ---- design ----------------------------------------------------------------------


def test_roll_and_pitch_use_the_whole_surface_at_the_saturation_error():
    gains = autopilot().gains
    assert np.isclose(
        abs(gains.roll_kp) * DESIGN.roll_error_at_saturation, DESIGN.surface_limit
    )
    assert np.isclose(
        abs(gains.pitch_kp) * DESIGN.pitch_error_at_saturation, DESIGN.surface_limit
    )
    assert gains.roll_kp > 0.0  # positive aileron rolls right
    assert gains.pitch_kp < 0.0  # positive elevator pitches the nose down


def test_outer_loops_are_slower_than_inner_loops():
    gains = autopilot().gains
    assert np.isclose(
        gains.roll_frequency / gains.course_frequency, DESIGN.course_separation
    )
    assert np.isclose(
        gains.pitch_frequency / gains.altitude_frequency, DESIGN.altitude_separation
    )
    assert 0.0 < gains.pitch_dc_gain < 1.0


def test_faster_flight_gives_faster_inner_loops():
    # Same gains in surface per error, more dynamic pressure: a stiffer response.
    slow, fast = autopilot(18.0).gains, autopilot(30.0).gains
    assert fast.roll_frequency > slow.roll_frequency
    assert fast.pitch_frequency > slow.pitch_frequency


def test_wrap_angle_turns_the_short_way():
    assert np.isclose(wrap_angle(np.radians(350.0)), np.radians(-10.0))
    assert np.isclose(wrap_angle(np.radians(-190.0)), np.radians(170.0))
    assert wrap_angle(0.5) == 0.5


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        dataclasses.replace(DESIGN, max_bank=0.0)
    plane = FixedWing(P, PROP)
    trim = trim_level_flight(plane, 25.0)
    with pytest.raises(ValueError):
        FixedWingAutopilot(plane, trim, DESIGN, 0.0)
    with pytest.raises(ValueError):
        FixedWingAutopilot(FixedWing(P), trim, DESIGN, DT)


# ---- closed loop -----------------------------------------------------------------


def test_commands_start_at_the_trim():
    pilot = autopilot()
    assert np.isclose(pilot.course_command, 0.0)
    assert np.isclose(pilot.altitude_command, 100.0)
    assert np.isclose(pilot.airspeed_command, 25.0)


def test_holds_the_trim_and_cures_the_spiral():
    # 10 deg of bank: left alone the spiral mode would grow it (doubling in ~8 s);
    # the autopilot rolls the wings level and comes back to the course.
    pilot = autopilot()
    start = np.array(pilot.trim.state)
    bank = quaternion.from_axis_angle([1.0, 0.0, 0.0], np.radians(10.0))
    start[6:10] = quaternion.multiply(start[6:10], bank)
    _, states, commands = fly(pilot, 30.0, start)
    angles = euler(states)
    assert abs(angles[-1, 0]) < np.radians(0.1)
    assert abs(course(states)[-1]) < np.radians(0.1)
    assert abs(-states[-1, 2] - 100.0) < 0.1
    assert_within_limits(states, commands)


def test_course_step_settles_within_the_bank_limit():
    pilot = autopilot()
    pilot.course_command = np.radians(90.0)
    times, states, commands = fly(pilot, 30.0)
    chi = course(states)
    settled = times >= 15.0
    assert np.all(np.abs(chi[settled] - np.radians(90.0)) < np.radians(0.5))
    assert np.max(chi) < np.radians(92.0)  # (almost) no overshoot
    assert np.all(np.abs(-states[:, 2] - 100.0) < 1.0)  # altitude held in the turn
    assert np.max(np.abs(euler(states)[:, 0])) > np.radians(30.0)  # banked hard
    assert_within_limits(states, commands)


def test_altitude_step_settles_within_the_pitch_limit():
    pilot = autopilot()
    pilot.altitude_command = 120.0
    _, states, commands = fly(pilot, 40.0)
    altitude = -states[:, 2]
    assert abs(altitude[-1] - 120.0) < 0.1
    assert altitude.max() < 123.0  # overshoot < 15 %
    assert np.all(np.abs(course(states)) < np.radians(0.5))
    assert_within_limits(states, commands)


@pytest.mark.parametrize("target", [20.0, 30.0])
def test_airspeed_step_settles_with_the_throttle(target):
    pilot = autopilot()
    pilot.airspeed_command = target
    times, states, commands = fly(pilot, 40.0)
    speed = airspeed(pilot, times, states)
    assert abs(speed[-1] - target) < 0.05
    assert np.all(np.abs(-states[:, 2] - 100.0) < 1.0)
    assert_within_limits(states, commands)


def test_fast_descending_reversal_does_not_dive():
    # Regression: with the pitch gain set by a fixed natural frequency, at 30 m/s
    # the pitch loop had almost no stiffness and this manoeuvre ended 117 m lower.
    pilot = autopilot(30.0)
    pilot.course_command = np.radians(179.0)
    pilot.altitude_command = 80.0
    _, states, commands = fly(pilot, 40.0)
    altitude = -states[:, 2]
    assert altitude.min() > 70.0
    assert abs(altitude[-1] - 80.0) < 0.2
    assert abs(wrap_angle(course(states)[-1] - np.radians(179.0))) < np.radians(0.5)
    assert_within_limits(states, commands)


def test_crosswind_is_flown_with_the_crab_angle():
    # 5 m/s blowing West (from the East), ground track North: the air must come
    # from the East as much as the wind does, so the nose points East of North, into
    # the wind, by asin(5 / 25) = 11.5 deg. The course integrator removes the drift.
    wind = np.array([0.0, -5.0, 0.0])
    pilot = autopilot(wind=lambda t, p: wind)
    # The trim in this wind flies the air North, so the ground track starts at
    # atan2(-5, 25) = -11.3 deg; command the ground track North instead.
    assert np.isclose(pilot.course_command, np.arctan2(-5.0, 25.0))
    pilot.course_command = 0.0
    _, states, commands = fly(pilot, 40.0)
    assert abs(course(states)[-1]) < np.radians(0.2)
    heading = euler(states)[-1, 2]
    assert np.isclose(heading, np.arcsin(5.0 / 25.0), atol=np.radians(0.3))
    assert_within_limits(states, commands)


def test_reset_gives_the_same_flight_twice():
    pilot = autopilot()
    pilot.course_command = np.radians(45.0)
    _, first, _ = fly(pilot, 5.0)
    pilot.reset()
    _, second, _ = fly(pilot, 5.0)
    assert np.array_equal(first, second)


# ---- orbit guidance --------------------------------------------------------------

CENTER = np.array([100.0, -50.0, -100.0])


@pytest.mark.parametrize("clockwise, course_deg", [(True, 90.0), (False, -90.0)])
def test_orbit_course_is_the_tangent_on_the_circle(clockwise, course_deg):
    # North of the centre: clockwise (seen from above) flies East, anticlockwise West.
    on_circle = CENTER + np.array([200.0, 0.0, 0.0])
    course = orbit_course(on_circle, CENTER, 200.0, clockwise)
    assert np.isclose(course, np.radians(course_deg))


def test_orbit_course_points_at_the_centre_from_far_away():
    far = CENTER + np.array([0.0, 1e6, 0.0])  # far to the East: the centre is West
    course = orbit_course(far, CENTER, 200.0, clockwise=True)
    assert abs(wrap_angle(course - np.radians(-90.0))) < np.radians(0.1)


def test_orbit_course_points_away_from_the_centre_inside():
    # Near the centre, gain 2: chi = phi + (90 deg - atan(2)) = 26.6 deg off radial.
    near = CENTER + np.array([1e-3, 0.0, 0.0])  # just North of the centre
    course = orbit_course(near, CENTER, 200.0, clockwise=True, gain=2.0)
    assert np.isclose(course, np.pi / 2 - np.arctan(2.0), atol=1e-4)


def test_orbit_course_rejects_invalid_input():
    with pytest.raises(ValueError):
        orbit_course(CENTER, CENTER, 0.0)
    with pytest.raises(ValueError):
        orbit_course(CENTER, CENTER, 100.0, gain=0.0)


def test_autopilot_holds_the_orbit():
    # 18 m/s from 400 m outside the circle: it turns onto the 150 m circle and stays
    # on it, banked at atan(V^2 / (g R)) = 12.4 deg, at constant altitude.
    pilot = autopilot(18.0)
    center = np.array([0.0, 0.0, -100.0])
    start = np.array(pilot.trim.state)
    start[0:2] = [-550.0, 0.0]

    def pre_step(t, state):
        pilot.course_command = orbit_course(state[:3], center, 150.0)
        pilot(t, state)

    plane = pilot.vehicle
    times, states = run(
        plane.derivative,
        start,
        DT,
        120.0,
        post_step=normalize_attitude,
        pre_step=pre_step,
    )
    late = times >= 90.0
    radius = np.hypot(states[late, 0], states[late, 1])
    assert np.all(np.abs(radius - 150.0) < 0.5)
    assert np.all(np.abs(-states[late, 2] - 100.0) < 0.1)
    bank = euler(states[late])[:, 0]
    assert np.allclose(
        bank, np.arctan(18.0**2 / (9.80665 * 150.0)), atol=np.radians(1.0)
    )
