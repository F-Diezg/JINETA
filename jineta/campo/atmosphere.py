"""International Standard Atmosphere (ISA): air properties versus altitude.

Every function takes the altitude above mean sea level [m] and returns an SI value.
The model has two layers:

- Troposphere (0 to 11 km): temperature falls linearly, 6.5 K per km.
- Isothermal layer (11 to 20 km): temperature stays constant.

The altitude is treated as geopotential altitude, the variable the ISA tables are
written in. It differs from geometric altitude by ~0.17 % at 11 km, well below the
uncertainty in vehicle aero parameters, so no conversion is applied. Above 20 km the
real ISA warms up again; that layer is not modelled, so results there are not valid.
Below sea level the troposphere formulas simply continue.
"""

import math

# ISA sea-level reference values
SEA_LEVEL_TEMPERATURE = 288.15  # [K]
SEA_LEVEL_PRESSURE = 101_325.0  # [Pa]

# ISA definition constants
STANDARD_GRAVITY = 9.80665  # g0 of the ISA hydrostatic equation [m/s^2]
GAS_CONSTANT_AIR = 287.05287  # specific gas constant of dry air [J/(kg K)]
HEAT_CAPACITY_RATIO = 1.4  # gamma of air [-]
LAPSE_RATE = 0.0065  # temperature drop with altitude in the troposphere [K/m]
TROPOPAUSE_ALTITUDE = 11_000.0  # top of the troposphere [m]

# Derived values at the tropopause (11 km)
TROPOPAUSE_TEMPERATURE = SEA_LEVEL_TEMPERATURE - LAPSE_RATE * TROPOPAUSE_ALTITUDE  # [K]
# Exponent g0 / (R * L) of the troposphere pressure law (~5.2559) [-]
PRESSURE_EXPONENT = STANDARD_GRAVITY / (GAS_CONSTANT_AIR * LAPSE_RATE)
TROPOPAUSE_PRESSURE = (
    SEA_LEVEL_PRESSURE
    * (TROPOPAUSE_TEMPERATURE / SEA_LEVEL_TEMPERATURE) ** PRESSURE_EXPONENT
)  # [Pa]


def isa_temperature(altitude: float) -> float:
    """Air temperature [K] at the given altitude [m]."""
    if altitude <= TROPOPAUSE_ALTITUDE:
        return SEA_LEVEL_TEMPERATURE - LAPSE_RATE * altitude
    return TROPOPAUSE_TEMPERATURE


def isa_pressure(altitude: float) -> float:
    """Air pressure [Pa] at the given altitude [m]."""
    if altitude <= TROPOPAUSE_ALTITUDE:
        temperature_ratio = isa_temperature(altitude) / SEA_LEVEL_TEMPERATURE
        return SEA_LEVEL_PRESSURE * temperature_ratio**PRESSURE_EXPONENT
    # Isothermal layer: pressure decays exponentially with the height above 11 km
    height = altitude - TROPOPAUSE_ALTITUDE
    decay = STANDARD_GRAVITY * height / (GAS_CONSTANT_AIR * TROPOPAUSE_TEMPERATURE)
    return TROPOPAUSE_PRESSURE * math.exp(-decay)


def isa_density(altitude: float) -> float:
    """Air density [kg/m^3] at the given altitude [m], from the ideal gas law."""
    return isa_pressure(altitude) / (GAS_CONSTANT_AIR * isa_temperature(altitude))


def isa_speed_of_sound(altitude: float) -> float:
    """Speed of sound [m/s] at the given altitude [m]."""
    temperature = isa_temperature(altitude)
    return math.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT_AIR * temperature)