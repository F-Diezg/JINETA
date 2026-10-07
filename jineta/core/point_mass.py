"""Point-mass dynamics: translation only, under gravity and aerodynamic drag.

Reference frame: NED (North-East-Down). z points DOWN, so altitude = -z
and gravity acts in the +z direction.
"""

from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass
class PointMass:
    """A body with mass but no orientation.

    State vector: [x, y, z, vx, vy, vz] in NED [m, m/s].
    Drag acts against the velocity relative to the air, v - wind.
    """

    mass: float  # [kg]
    drag_area: float  # Cd * A [m^2]; 0 means no drag (vacuum)
    density: DensityModel = sea_level_density
    wind: WindModel = no_wind

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt = [velocity, acceleration]."""
        position = state[0:3]
        velocity = state[3:6]
        altitude = -state[2]  # NED: z points down

        gravity_force = np.array([0.0, 0.0, self.mass * GRAVITY])

        rho = self.density(altitude)
        # Drag opposes the motion relative to the air, not relative to the ground
        air_velocity = velocity - self.wind(t, position)
        airspeed = np.linalg.norm(air_velocity)
        drag_force = -0.5 * rho * self.drag_area * airspeed * air_velocity

        acceleration = (gravity_force + drag_force) / self.mass
        return np.concatenate([velocity, acceleration])
