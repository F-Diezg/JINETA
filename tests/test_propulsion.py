"""Tests for the electric motor + propeller model, with the small fixed-wing profile.

The reference values come from the chapter 4 check of the source simulator
(mavsim_public, launch_files/chap04/chap4_check.py, cases 1 and 2), computed there
with the same parameters and the source's density 1.2682 kg/m^3. The other tests
check the physics directly: the torque balance that defines the rotor speed, the
advance-ratio form of the thrust, and how thrust changes with throttle, airspeed and
density.
"""

import dataclasses

import numpy as np
import pytest

from jineta.campo.atmosphere import isa_density
from jineta.jinete.profiles import SMALL_FIXED_WING_PROPULSION
from jineta.jinete.propulsion import propeller_thrust_torque, rotor_speed

P = SMALL_FIXED_WING_PROPULSION
RHO = 1.225  # [kg/m^3]
SOURCE_RHO = 1.2682  # density used by the source for its reference values


@pytest.mark.parametrize(
    "throttle, thrust, torque",
    [
        (0.5, -12.43072534597213, -0.49879620097737803),  # case 1: windmilling
        (1.0, 37.7794805541605, 1.809846739787848),  # case 2
    ],
)
def test_matches_the_source_reference_values(throttle, thrust, torque):
    out = propeller_thrust_torque(P, 25.0, throttle, SOURCE_RHO)
    assert np.isclose(out.thrust, thrust, rtol=1e-12)
    assert np.isclose(out.torque, torque, rtol=1e-12)


@pytest.mark.parametrize("speed", [0.0, 10.0, 25.0, 35.0])
@pytest.mark.parametrize("throttle", [0.2, 0.6, 1.0])
def test_motor_torque_equals_propeller_torque(speed, throttle):
    # The rotor speed is where Q_motor = K (i - i0), i = (V_in - K Omega) / R,
    # equals the torque of the propeller.
    out = propeller_thrust_torque(P, speed, throttle, RHO)
    k = P.motor_constant
    current = (throttle * P.max_voltage - k * out.rotor_speed) / P.resistance
    motor_torque = k * (current - P.no_load_current)
    assert out.rotor_speed > 0.0
    assert np.isclose(motor_torque, out.torque, rtol=1e-9)


@pytest.mark.parametrize("speed", [5.0, 25.0])
def test_thrust_is_rho_n2_d4_ct_of_the_advance_ratio(speed):
    out = propeller_thrust_torque(P, speed, 0.8, RHO)
    n = out.rotor_speed / (2 * np.pi)
    j = speed / (n * P.diameter)
    c_t = P.c_thrust[0] + P.c_thrust[1] * j + P.c_thrust[2] * j**2
    c_q = P.c_torque[0] + P.c_torque[1] * j + P.c_torque[2] * j**2
    assert np.isclose(out.thrust, RHO * n**2 * P.diameter**4 * c_t, rtol=1e-12)
    assert np.isclose(out.torque, RHO * n**2 * P.diameter**5 * c_q, rtol=1e-12)


def test_no_throttle_at_rest_gives_nothing():
    # Below the no-load voltage the motor cannot turn the propeller: it stays stopped
    # (the quadratic's root would be a motor spinning backwards).
    out = propeller_thrust_torque(P, 0.0, 0.0, RHO)
    assert out.rotor_speed == 0.0
    assert out.thrust == 0.0
    assert out.torque == 0.0


@pytest.mark.parametrize("speed", [0.0, 20.0])
def test_thrust_and_rotor_speed_grow_with_throttle(speed):
    throttles = np.linspace(0.1, 1.0, 10)
    outs = [propeller_thrust_torque(P, speed, x, RHO) for x in throttles]
    assert np.all(np.diff([o.thrust for o in outs]) > 0.0)
    assert np.all(np.diff([o.rotor_speed for o in outs]) > 0.0)


def test_thrust_falls_with_airspeed():
    speeds = np.linspace(0.0, 40.0, 9)
    thrusts = [propeller_thrust_torque(P, v, 0.7, RHO).thrust for v in speeds]
    assert np.all(np.diff(thrusts) < 0.0)


def test_zero_throttle_in_flight_windmills_and_brakes():
    # With no voltage the motor is a shorted generator: the air turns the propeller
    # and the propeller brakes the aircraft. "Throttle 0" is not "engine off".
    out = propeller_thrust_torque(P, 25.0, 0.0, RHO)
    assert out.rotor_speed > 0.0
    assert out.thrust < -20.0  # more than twice the drag of the airframe at 25 m/s
    assert out.torque < 0.0  # the air drives the rotation


def test_full_throttle_static_thrust():
    # ~82 N at sea level: thrust-to-weight ~0.76 for the 11 kg aircraft.
    out = propeller_thrust_torque(P, 0.0, 1.0, RHO)
    assert 80.0 < out.thrust < 84.0


def test_throttle_is_clipped():
    above = propeller_thrust_torque(P, 15.0, 1.5, RHO)
    full = propeller_thrust_torque(P, 15.0, 1.0, RHO)
    below = propeller_thrust_torque(P, 15.0, -0.2, RHO)
    off = propeller_thrust_torque(P, 15.0, 0.0, RHO)
    assert above == full
    assert below == off


def test_reverse_airflow_is_taken_as_none():
    assert propeller_thrust_torque(P, -5.0, 0.6, RHO) == propeller_thrust_torque(
        P, 0.0, 0.6, RHO
    )


def test_thinner_air_less_thrust_faster_propeller():
    # At 3000 m the propeller needs less torque per turn, so the motor spins it
    # faster, but not enough to make up for the thinner air.
    sea = propeller_thrust_torque(P, 20.0, 0.8, RHO)
    high = propeller_thrust_torque(P, 20.0, 0.8, isa_density(3000.0))
    assert high.thrust < sea.thrust
    assert high.rotor_speed > sea.rotor_speed


def test_rotor_speed_function_matches_the_output():
    out = propeller_thrust_torque(P, 18.0, 0.55, RHO)
    assert rotor_speed(P, 18.0, 0.55, RHO) == out.rotor_speed


@pytest.mark.parametrize(
    "change",
    [
        {"diameter": 0.0},
        {"motor_constant": -0.1},
        {"resistance": 0.0},
        {"no_load_current": -1.0},
        {"max_voltage": 0.0},
        {"c_thrust": (0.1, 0.0)},
        {"c_torque": (0.0, 0.005, -0.01)},
    ],
)
def test_rejects_invalid_parameters(change):
    with pytest.raises(ValueError):
        dataclasses.replace(P, **change)


def test_profile_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        P.diameter = 1.0
