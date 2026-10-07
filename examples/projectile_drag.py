"""Compare a projectile trajectory in vacuum vs. with aerodynamic drag."""

import matplotlib.pyplot as plt
import numpy as np

from jineta.core.point_mass import PointMass
from jineta.core.simulation import run

speed, elevation = 80.0, np.radians(45.0)
y0 = np.array([0, 0, 0, speed * np.cos(elevation), 0, -speed * np.sin(elevation)])


def hit_ground(t, s):
    return s[2] > 0


for label, drag_area in [("Vacuum", 0.0), ("With drag", 0.01)]:
    body = PointMass(mass=1.0, drag_area=drag_area)
    _, states = run(body.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground)
    plt.plot(states[:, 0], -states[:, 2], label=label)  # altitude = -z

plt.xlabel("Downrange distance [m]")
plt.ylabel("Altitude [m]")
plt.title("Projectile trajectory: vacuum vs. drag")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
