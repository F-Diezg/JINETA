"""Tests for the level-flight trim of the small fixed-wing UAV.

The trim must be an equilibrium of the full 6-DOF model (all accelerations zero),
stay steady when flown, and make physical sense: less angle of attack and more
throttle as the speed grows, more angle of attack in thinner air, the aileron
holding the propeller torque, the same controls in a uniform wind, and no trim
beyond the speed that full throttle can hold.
"""

import numpy as np
import pytest

from jineta.campo.atmosphere import isa_density
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.aero import air_data, body_airspeed
from jineta.jinete.fixed_wing import FixedWing
from jineta.jinete.profiles import SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION
from jineta.jinete.propulsion import propeller_thrust_torque
from jineta.jinete.trim import longitudinal_estimate, trim_level_flight

P = SMALL_FIXED_WING
PROP = SMALL_FIXED_WING_PROPULSION


def aircraft(**kwargs):
    return FixedWing(P, PROP, density=isa_density, **kwargs)


@pytest.mark.parametrize("speed", [18.0, 22.0, 25.0, 30.0])
def test_trim_is_an_equilibrium_of_level_flight(speed):
    plane = aircraft()
    trim = trim_level_flight(plane, speed)
    state = trim.state
    derivative = plane.derivative(0.0, state)
    assert np.allclose(derivative[3:13], 0.0, atol=1e-9)  # no acceleration at all
    assert np.isclose(np.linalg.norm(state[3:6]), speed, rtol=1e-12)
    assert abs(state[5]) < 1e-12  # level: no vertical speed
    air = air_data(body_airspeed(0.0, state))
    assert np.isclose(air.alpha, trim.alpha, atol=1e-12)
    assert np.isclose(air.beta, trim.beta, atol=1e-12)
    # The aircraft is left commanded with the trim.
    assert plane.surfaces == trim.surfaces
    assert plane.throttle == trim.throttle


def test_faster_needs_less_alpha_less_elevator_more_throttle():
    plane = aircraft()
    trims = [trim_level_flight(plane, v) for v in (18.0, 22.0, 26.0, 30.0)]
    alphas = [t.alpha for t in trims]
    elevators = [t.surfaces.elevator for t in trims]
    throttles = [t.throttle for t in trims]
    assert np.all(np.diff(alphas) < 0.0)
    assert np.all(np.diff(elevators) > 0.0)  # less trailing-edge-up (negative)
    assert np.all(np.diff(throttles) > 0.0)


def test_thinner_air_needs_more_alpha():
    # Same airspeed, lower density: less dynamic pressure, so more lift coefficient.
    low = trim_level_flight(aircraft(), 25.0, altitude=100.0)
    high = trim_level_flight(aircraft(), 25.0, altitude=3000.0)
    assert high.alpha > low.alpha + np.radians(1.0)


def test_aileron_holds_the_propeller_torque():
    # The propeller torque Q rolls the aircraft left (-Q); the aerodynamic rolling
    # moment of aileron, rudder and sideslip must cancel it exactly.
    plane = aircraft()
    trim = trim_level_flight(plane, 25.0)
    rho = isa_density(100.0)
    prop = propeller_thrust_torque(
        PROP, body_airspeed(0.0, trim.state)[0], trim.throttle, rho
    )
    qbar_s_b = 0.5 * rho * 25.0**2 * P.wing_area * P.span
    aero_roll = qbar_s_b * (
        P.c_ell_beta * trim.beta
        + P.c_ell_aileron * trim.surfaces.aileron
        + P.c_ell_rudder * trim.surfaces.rudder
    )
    assert prop.torque > 0.0
    assert trim.surfaces.aileron > 0.0  # rolls right, against the torque
    assert np.isclose(aero_roll, prop.torque, rtol=1e-6)
    assert abs(trim.beta) < np.radians(0.1)  # wings level, almost no sideslip


def test_trimmed_flight_stays_steady():
    plane = aircraft()
    trim = trim_level_flight(plane, 25.0)
    _, states = run(
        plane.derivative, trim.state, 0.02, 60.0, post_step=normalize_attitude
    )
    assert np.allclose(states[:, 2], trim.state[2], atol=1e-6)  # altitude
    assert np.allclose(states[:, 3:6], trim.state[3:6], atol=1e-9)  # velocity
    assert np.allclose(states[:, 6:10], trim.state[6:10], atol=1e-9)  # attitude
    assert np.allclose(states[:, 10:13], 0.0, atol=1e-9)  # no rotation
    assert np.isclose(states[-1, 0], 25.0 * 60.0, rtol=1e-9)  # 1500 m North


def test_uniform_wind_changes_the_ground_speed_not_the_trim():
    calm = trim_level_flight(aircraft(), 25.0)
    wind = np.array([-8.0, 3.0, 0.0])
    windy_plane = aircraft(wind=lambda t, p: wind)
    windy = trim_level_flight(windy_plane, 25.0)
    assert np.isclose(windy.alpha, calm.alpha, atol=1e-10)
    assert np.isclose(windy.throttle, calm.throttle, atol=1e-10)
    assert np.allclose(windy.state[3:6], calm.state[3:6] + wind, atol=1e-10)
    assert np.allclose(windy_plane.derivative(0.0, windy.state)[3:], 0.0, atol=1e-9)


def test_course_turns_the_whole_trim():
    north = trim_level_flight(aircraft(), 25.0)
    east = trim_level_flight(aircraft(), 25.0, course=np.pi / 2)
    assert np.allclose(east.state[3:6], [0.0, 25.0, 0.0], atol=1e-9)
    assert np.isclose(east.throttle, north.throttle, atol=1e-10)
    assert np.isclose(east.surfaces.aileron, north.surfaces.aileron, atol=1e-10)


def test_full_throttle_limits_the_level_speed():
    # ~32.5 m/s at 100 m: 32.4 m/s still trims (throttle > 0.99), 33 m/s does not.
    assert trim_level_flight(aircraft(), 32.4).throttle > 0.99
    with pytest.raises(ValueError):
        trim_level_flight(aircraft(), 33.0)


def test_trims_at_every_speed_up_to_the_maximum():
    # Regression: started from a fixed guess, the solver jumped past full throttle
    # (where the clipped thrust has no slope) and failed at some speeds that do
    # trim, e.g. 32.05 m/s. The textbook estimate starts it close enough.
    plane = aircraft()
    for speed in np.arange(15.0, 32.45, 0.25):
        trim = trim_level_flight(plane, speed)
        assert 0.0 <= trim.throttle <= 1.0


def test_textbook_estimate_is_close_to_the_trim():
    plane = aircraft()
    alpha, elevator, throttle = longitudinal_estimate(plane, 25.0, 100.0)
    trim = trim_level_flight(plane, 25.0)
    assert abs(alpha - trim.alpha) < np.radians(0.1)
    assert abs(elevator - trim.surfaces.elevator) < np.radians(0.2)
    assert abs(throttle - trim.throttle) < 0.01


def test_rejects_a_glider_and_a_non_positive_speed():
    with pytest.raises(ValueError):
        trim_level_flight(FixedWing(P), 25.0)
    with pytest.raises(ValueError):
        trim_level_flight(aircraft(), 0.0)


def test_trim_state_is_read_only():
    trim = trim_level_flight(aircraft(), 25.0)
    with pytest.raises(ValueError):
        trim.state[0] = 1.0
