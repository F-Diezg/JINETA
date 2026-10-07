"""Wind drift of a body dropped from altitude: still air, uniform wind and wind shear.

The body is light and draggy (terminal speed ~18 m/s). The shear profile is an
illustrative boundary layer: the speed grows with height following a power law with
exponent 1/7 (5 m/s at 10 m), and the wind veers clockwise with height, from 240
degrees at the ground to 270 at 500 m (directions are where the wind comes FROM).
"""

import matplotlib.pyplot as plt
import numpy as np

from jineta.campo.wind import ConstantWind, ProfileWind, meteorological_wind
from jineta.core.point_mass import PointMass
from jineta.core.simulation import run


def hit_ground(t, s):
    return s[2] > 0


heights = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
speeds = 5.0 * (heights / 10.0) ** (1 / 7)  # power-law boundary layer
directions = np.interp(heights, [0.0, 500.0], [240.0, 270.0])  # veering with height
shear = ProfileWind(
    altitudes=heights,
    velocities=np.array(
        [meteorological_wind(s, d) for s, d in zip(speeds, directions, strict=True)]
    ),
)
wind_aloft = shear(0.0, np.array([0.0, 0.0, -500.0]))  # wind at the release altitude

y0 = np.array([0, 0, -500.0, 0, 0, 0])  # released at rest, 500 m up
cases = [
    ("Still air", PointMass(mass=1.0, drag_area=0.05)),
    (
        "Uniform wind aloft",
        PointMass(mass=1.0, drag_area=0.05, wind=ConstantWind(wind_aloft)),
    ),
    ("Wind shear profile", PointMass(mass=1.0, drag_area=0.05, wind=shear)),
]

for label, body in cases:
    times, states = run(body.derivative, y0, dt=0.01, t_end=120.0, stop=hit_ground)
    north, east = states[-1, 0], states[-1, 1]
    print(
        f"{label:22s} drift = {np.hypot(north, east):6.1f} m   "
        f"(north {north:6.1f} m, east {east:6.1f} m)   fall time = {times[-1]:5.1f} s"
    )
    (line,) = plt.plot(states[:, 1], states[:, 0], label=label)
    plt.plot(east, north, "o", color=line.get_color())

plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("Ground track of a body dropped from 500 m")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
