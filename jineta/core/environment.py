"""Interfaces between the dynamics and the environment.

The dynamics in `core` never import the environment models in `campo`. They only
know these interfaces: the gravity constant, and the shape of a density model and of
a wind model, plus neutral defaults (sea-level density, still air). Any function or
callable object with the right shape can be plugged in, such as the ISA density or
the wind models in `campo`.
"""

from collections.abc import Callable

import numpy as np

GRAVITY = 9.80665  # standard gravity [m/s^2]

# Air density model: altitude [m] -> density [kg/m^3]
DensityModel = Callable[[float], float]

# Wind model: (time [s], position NED [m]) -> wind velocity NED [m/s]
WindModel = Callable[[float, np.ndarray], np.ndarray]


def sea_level_density(altitude: float) -> float:
    """Constant density (ISA sea level), the default. See campo.atmosphere for ISA."""
    return 1.225


def no_wind(t: float, position: np.ndarray) -> np.ndarray:
    """Still air: zero wind at every time and place, the default."""
    return np.zeros(3)
