"""Point-mass dynamics: translation only, under gravity and aerodynamic drag.

Reference frame: NED (North-East-Down). z points DOWN, so altitude = -z
and gravity acts in the +z direction.
"""

from dataclasses import dataclass

import numpy as np

from jineta.core.environment import (
    GRAVITY,
    DensityModel,
    WindModel,
    no_wind,
    sea_level_density,
)


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

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be > 0")
        if self.drag_area < 0.0:
            raise ValueError("drag_area must be >= 0")

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
