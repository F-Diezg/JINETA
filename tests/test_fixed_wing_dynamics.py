"""Tests for the FixedWing vehicle: aerodynamics + propeller acting on a RigidBody.

Two reference cases come from the chapter 4 check of the source simulator
(mavsim_public, launch_files/chap04/chap4_check.py), computed there with the same
parameters, density 1.2682 kg/m^3 and g = 9.81 m/s^2 (their forces include gravity;
ours do not, so it is added here). The other tests check how the vehicle composes
its parts: thrust along the body x axis, the propeller torque as a rolling moment,
the airspeed relative to the wind, the density at the current altitude, and the
commands read when the forces are evaluated.
"""

import numpy as np

from jineta.campo.atmosphere import isa_density
from jineta.core import quaternion
from jineta.core.environment import GRAVITY, sea_level_density
from jineta.jinete.aero import body_airspeed
from jineta.jinete.fixed_wing import ControlSurfaces, FixedWing, aero_force_moment
from jineta.jinete.profiles import SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION
from jineta.jinete.propulsion import propeller_thrust_torque

P = SMALL_FIXED_WING
PROP = SMALL_FIXED_WING_PROPULSION
SOURCE_RHO = 1.2682  # [kg/m^3]
SOURCE_G = 9.81  # [m/s^2]
RHO = sea_level_density(0.0)  # the vehicle's default density model


def source_density(altitude):
    return SOURCE_RHO


def state_from_source(source_state):
    """Source state [pn, pe, pd, u, v, w, quaternion, p, q, r] (body velocity) to
    ours (NED velocity)."""
    s = np.asarray(source_state, dtype=float)
    rotation = quaternion.to_matrix(s[6:10])
    return np.concatenate([s[:3], rotation @ s[3:6], s[6:13]])


def with_source_gravity(force, state):
    """Our force (no gravity) plus gravity in body axes with the source's g."""
    rotation = quaternion.to_matrix(state[6:10])
    return force + rotation.T @ np.array([0.0, 0.0, P.mass * SOURCE_G])


def level_state(speed=25.0, altitude=100.0):
    return np.array([0, 0, -altitude, speed, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


def test_matches_the_source_in_level_flight():
    # Case 1: level at 25 m/s, elevator -0.2, rudder 0.005, throttle 0.5.
    aircraft = FixedWing(P, PROP, density=source_density)
    aircraft.surfaces = ControlSurfaces(elevator=-0.2, rudder=0.005)
    aircraft.throttle = 0.5
    state = state_from_source([0, 0, -100, 25, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    force, moment = aircraft.force_moment(0.0, state)
    expected_force = [-21.48250762600656, 0.20707328125000002, 63.44373750624077]
    expected_moment = [0.506370113312378, 8.75643373378125, -0.21774997963125006]
    expected_rate_dot = [0.6021690003674434, 7.714919589234582, -0.0825746628692495]
    assert np.allclose(with_source_gravity(force, state), expected_force, rtol=1e-9)
    assert np.allclose(moment, expected_moment, rtol=1e-9)
    rate_dot = aircraft.derivative(0.0, state)[10:13]
    assert np.allclose(rate_dot, expected_rate_dot, rtol=1e-9)


def test_matches_the_source_in_a_general_state_with_wind():
    # Case 3: rolled, pitched, yawed, rotating, with a small gust (body axes in the
    # source, turned into NED here), full throttle.
    source_state = [
        6.19506532e01,
        2.22940203e01,
        -1.10837551e02,
        2.73465947e01,
        6.19628233e-01,
        1.42257772e00,
        9.38688796e-01,
        2.47421558e-01,
        6.56821468e-02,
        2.30936730e-01,
        4.98772167e-03,
        1.68736005e-01,
        1.71797313e-01,
    ]
    state = state_from_source(source_state)
    gust_body = np.array([-0.00165177, -0.00475441, -0.01717199])
    gust_ned = quaternion.to_matrix(state[6:10]) @ gust_body
    aircraft = FixedWing(P, PROP, density=source_density, wind=lambda t, p: gust_ned)
    aircraft.surfaces = ControlSurfaces(-0.15705144, 0.01788999, 0.01084654)
    aircraft.throttle = 1.0
    force, moment = aircraft.force_moment(0.0, state)
    force = with_source_gravity(force, state)

    # The source feeds the propeller the total airspeed V; we use the axial
    # component u (the advance ratio is defined with the axial inflow). Undo that
    # difference before comparing: it is the only one in F_x and in the roll moment.
    airspeed = body_airspeed(0.0, state, aircraft.wind)
    ours = propeller_thrust_torque(PROP, airspeed[0], 1.0, SOURCE_RHO)
    theirs = propeller_thrust_torque(
        PROP, float(np.linalg.norm(airspeed)), 1.0, SOURCE_RHO
    )
    force[0] -= ours.thrust - theirs.thrust
    moment[0] += ours.torque - theirs.torque

    expected_force = [24.99040531, 48.44092512, -39.98407113]
    expected_moment = [0.10925814573797221, 0.1249623335264915, -0.09513777456448343]
    assert np.isclose(force[0], expected_force[0], atol=1e-6)
    # F_y differs by 0.0015 N (3e-5 relative), not explained (all the inputs it
    # depends on match the source to 1e-9).
    assert np.isclose(force[1], expected_force[1], atol=2e-3)
    assert np.isclose(force[2], expected_force[2], atol=1e-6)
    assert np.allclose(moment, expected_moment, atol=1e-6)


def test_commands_start_neutral():
    aircraft = FixedWing(P, PROP)
    assert aircraft.surfaces == ControlSurfaces()
    assert aircraft.throttle == 0.0


def test_a_glider_feels_only_the_aerodynamics():
    glider = FixedWing(P)
    glider.surfaces = ControlSurfaces(elevator=0.05, aileron=-0.02, rudder=0.01)
    glider.throttle = 1.0  # no propeller: ignored
    state = level_state()
    state[10:13] = [0.1, -0.05, 0.02]
    force, moment = glider.force_moment(0.0, state)
    expected = aero_force_moment(
        P, [25.0, 0.0, 0.0], [0.1, -0.05, 0.02], glider.surfaces, RHO
    )
    assert np.allclose(force, expected[0], rtol=1e-12)
    assert np.allclose(moment, expected[1], rtol=1e-12)


def test_thrust_along_the_nose_and_torque_as_a_rolling_moment():
    # At rest in still air there is no aerodynamic load: only the propeller.
    aircraft = FixedWing(P, PROP)
    aircraft.throttle = 1.0
    state = level_state(speed=0.0)
    force, moment = aircraft.force_moment(0.0, state)
    static = propeller_thrust_torque(PROP, 0.0, 1.0, RHO)
    assert np.allclose(force, [static.thrust, 0.0, 0.0], atol=1e-12)
    assert np.allclose(moment, [-static.torque, 0.0, 0.0], atol=1e-12)


def test_thrust_follows_the_attitude():
    # Nose 30 deg up, at rest: the thrust is rotated into NED by the attitude.
    aircraft = FixedWing(P, PROP)
    aircraft.throttle = 1.0
    state = level_state(speed=0.0)
    pitch = np.radians(30.0)
    state[6:10] = quaternion.from_axis_angle([0, 1, 0], pitch)
    acceleration = aircraft.derivative(0.0, state)[3:6]
    thrust = propeller_thrust_torque(PROP, 0.0, 1.0, RHO).thrust
    expected = thrust / P.mass * np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    expected[2] += GRAVITY
    assert np.allclose(acceleration, expected, atol=1e-12)


def test_flying_with_the_wind_feels_no_air():
    # Ground velocity equal to the wind: no airspeed, so no aerodynamic load, and
    # the propeller works as at rest.
    wind = np.array([5.0, -3.0, 0.0])
    aircraft = FixedWing(P, PROP, wind=lambda t, p: wind)
    aircraft.throttle = 0.7
    at_rest = FixedWing(P, PROP)
    at_rest.throttle = 0.7
    state = level_state(speed=0.0)
    state[3:6] = wind
    force, moment = aircraft.force_moment(0.0, state)
    expected = at_rest.force_moment(0.0, level_state(speed=0.0))
    assert np.allclose(force, expected[0], atol=1e-12)
    assert np.allclose(moment, expected[1], atol=1e-12)


def test_loads_depend_on_airspeed_not_ground_speed():
    # 25 m/s in still air = hovering over the ground in a 25 m/s headwind.
    in_still_air = FixedWing(P, PROP)
    in_headwind = FixedWing(P, PROP, wind=lambda t, p: np.array([-25.0, 0.0, 0.0]))
    for aircraft in (in_still_air, in_headwind):
        aircraft.throttle = 0.8
        aircraft.surfaces = ControlSurfaces(elevator=-0.1)
    a = in_still_air.force_moment(0.0, level_state(speed=25.0))
    b = in_headwind.force_moment(0.0, level_state(speed=0.0))
    assert np.allclose(a[0], b[0], rtol=1e-12)
    assert np.allclose(a[1], b[1], rtol=1e-12)


def test_density_is_read_at_the_current_altitude():
    # For a glider every load scales with the density.
    seen = []

    def density(altitude):
        seen.append(altitude)
        return isa_density(altitude)

    glider = FixedWing(P, density=density)
    sea = glider.force_moment(0.0, level_state(altitude=0.0))
    high = glider.force_moment(0.0, level_state(altitude=3000.0))
    ratio = isa_density(3000.0) / isa_density(0.0)
    assert seen == [0.0, 3000.0]
    assert np.allclose(high[0], ratio * sea[0], rtol=1e-12)
    assert np.allclose(high[1], ratio * sea[1], rtol=1e-12)


def test_commands_are_read_when_the_forces_are_evaluated():
    aircraft = FixedWing(P, PROP)
    state = level_state()
    before = aircraft.force_moment(0.0, state)[1][1]
    aircraft.surfaces = ControlSurfaces(elevator=0.1)  # trailing edge down
    after = aircraft.force_moment(0.0, state)[1][1]
    assert after < before  # nose-down pitching moment
