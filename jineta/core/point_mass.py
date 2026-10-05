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


def sea_level_density(altitude: float) -> float:
    """Constant density (ISA sea level). Placeholder until campo/ISA exists."""
    return 1.225


@dataclass
class PointMass:
    """A body with mass but no orientation.

    State vector: [x, y, z, vx, vy, vz] in NED [m, m/s].
    """

    mass: float  # [kg]
    drag_area: float  # Cd * A [m^2]; 0 means no drag (vacuum)
    density: DensityModel = sea_level_density

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt = [velocity, acceleration]."""
        velocity = state[3:6]
        altitude = -state[2]  # NED: z points down

        gravity_force = np.array([0.0, 0.0, self.mass * GRAVITY])

        rho = self.density(altitude)
        speed = np.linalg.norm(velocity)
        drag_force = -0.5 * rho * self.drag_area * speed * velocity

        acceleration = (gravity_force + drag_force) / self.mass
        return np.concatenate([velocity, acceleration])