"""Effect of altitude-dependent air density on a high-altitude trajectory."""

import matplotlib.pyplot as plt
import numpy as np

from jineta.campo.atmosphere import isa_density
from jineta.core.point_mass import PointMass, sea_level_density
from jineta.core.simulation import run


def exponential_density(altitude: float) -> float:
    """Simple exponential atmosphere, decent approximation up to ~10 km."""
    return 1.225 * np.exp(-altitude / 8500.0)


def hit_ground(t, s):
    return s[2] > 0


speed, elevation = 300.0, np.radians(60.0)
y0 = np.array([0, 0, 0, speed * np.cos(elevation), 0, -speed * np.sin(elevation)])

cases = [
    ("Vacuum", PointMass(mass=10.0, drag_area=0.0)),
    (
        "Constant density",
        PointMass(mass=10.0, drag_area=0.005, density=sea_level_density),
    ),
    (
        "Exponential density",
        PointMass(mass=10.0, drag_area=0.005, density=exponential_density),
    ),
    ("ISA density", PointMass(mass=10.0, drag_area=0.005, density=isa_density)),
]

for label, body in cases:
    times, states = run(body.derivative, y0, dt=0.01, t_end=120.0, stop=hit_ground)
    max_altitude = -states[:, 2].min()
    print(
        f"{label:22s} range = {states[-1, 0]:7.0f} m   "
        f"max altitude = {max_altitude:6.0f} m   flight time = {times[-1]:5.1f} s"
    )
    plt.plot(states[:, 0], -states[:, 2], label=label)

plt.xlabel("Downrange distance [m]")
plt.ylabel("Altitude [m]")
plt.title("Trajectory vs. atmosphere model")
plt.grid(True)
plt.legend()
plt.show()