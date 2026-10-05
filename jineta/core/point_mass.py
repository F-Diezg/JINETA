"""Point-mass dynamics: translation only, under gravity and aerodynamic drag.

Reference frame: NED (North-East-Down). z points DOWN, so altitude = -z
and gravity acts in the +z direction.
"""

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.80665  # standard gravity [m/s^2]


@dataclass
class PointMass:
    """A body with mass but no orientation.

    State vector: [x, y, z, vx, vy, vz] in NED [m, m/s].
    """

    mass: float  # [kg]
    drag_area: float  # Cd * A [m^2]; 0 means no drag (vacuum)
    air_density: float = 1.225  # [kg/m^3], sea-level ISA

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt = [velocity, acceleration]."""
        velocity = state[3:6]

        gravity_force = np.array([0.0, 0.0, self.mass * GRAVITY])

        speed = np.linalg.norm(velocity)
        drag_force = -0.5 * self.air_density * self.drag_area * speed * velocity

        acceleration = (gravity_force + drag_force) / self.mass
        return np.concatenate([velocity, acceleration])