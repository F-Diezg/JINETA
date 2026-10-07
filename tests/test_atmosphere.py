"""Tests for the ISA atmosphere against standard table values and physical laws."""

import numpy as np
import pytest

from jineta.campo.atmosphere import (
    STANDARD_GRAVITY,
    TROPOPAUSE_ALTITUDE,
    isa_density,
    isa_pressure,
    isa_speed_of_sound,
    isa_temperature,
)
from jineta.core.point_mass import PointMass

# Standard ISA table, geopotential altitude:
# (altitude [m], T [K], p [Pa], rho [kg/m^3], a [m/s])
ISA_TABLE = [
    (0.0, 288.15, 101_325.0, 1.2250, 340.294),
    (5_000.0, 255.65, 54_019.9, 0.73612, 320.529),
    (11_000.0, 216.65, 22_632.1, 0.36392, 295.070),
    (20_000.0, 216.65, 5_474.89, 0.088035, 295.070),
]


@pytest.mark.parametrize("altitude, temperature, pressure, density, sound", ISA_TABLE)
def test_matches_standard_isa_table(altitude, temperature, pressure, density, sound):
    assert np.isclose(isa_temperature(altitude), temperature, rtol=1e-4)
    assert np.isclose(isa_pressure(altitude), pressure, rtol=1e-4)
    assert np.isclose(isa_density(altitude), density, rtol=1e-4)
    assert np.isclose(isa_speed_of_sound(altitude), sound, rtol=1e-4)


def test_troposphere_temperature_drops_6_5_k_per_km():
    drop = isa_temperature(2_000.0) - isa_temperature(3_000.0)
    assert np.isclose(drop, 6.5, atol=1e-9)


def test_isothermal_layer_temperature_is_constant():
    temperatures = [isa_temperature(h) for h in (11_000.0, 15_000.0, 20_000.0)]
    assert np.allclose(temperatures, 216.65, atol=1e-9)


def test_pressure_is_continuous_at_tropopause():
    eps = 1e-6
    below = isa_pressure(TROPOPAUSE_ALTITUDE - eps)
    above = isa_pressure(TROPOPAUSE_ALTITUDE + eps)
    assert np.isclose(below, above, rtol=1e-9)


def test_density_decreases_with_altitude():
    altitudes = np.linspace(0.0, 20_000.0, 41)
    densities = [isa_density(h) for h in altitudes]
    assert np.all(np.diff(densities) < 0.0)


@pytest.mark.parametrize("altitude", [3_000.0, 8_000.0, 15_000.0])
def test_pressure_satisfies_hydrostatic_equilibrium(altitude):
    # Fluid at rest: dp/dh = -rho * g (checked with a central finite difference)
    dh = 1.0
    dp_dh = (isa_pressure(altitude + dh) - isa_pressure(altitude - dh)) / (2 * dh)
    assert np.isclose(dp_dh, -isa_density(altitude) * STANDARD_GRAVITY, rtol=1e-6)


def test_isa_density_plugs_into_point_mass():
    """PointMass must evaluate the density at altitude = -z (NED), not at z."""
    mass, drag_area, speed, altitude = 10.0, 0.005, 100.0, 8_000.0
    body = PointMass(mass=mass, drag_area=drag_area, density=isa_density)
    state = np.array([0.0, 0.0, -altitude, speed, 0.0, 0.0])  # flying north

    accel_x = body.derivative(0.0, state)[3]

    expected = -0.5 * isa_density(altitude) * drag_area * speed**2 / mass
    assert np.isclose(accel_x, expected, rtol=1e-12)
