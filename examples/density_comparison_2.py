"""High-altitude shot: how much the density model matters as the body climbs.

Same idea as density_comparison.py, but with a low-drag body fired faster and steeper,
so it climbs to ~15 km. The ISA model is only valid up to 20 km, so the shot is chosen
to stay below that. The right panel shows the density profiles themselves: on a
logarithmic axis the exponential model is a straight line, while the ISA bends at the
tropopause (11 km).
"""

import matplotlib.pyplot as plt
import numpy as np

from jineta.campo.atmosphere import TROPOPAUSE_ALTITUDE, isa_density
from jineta.core.point_mass import PointMass, sea_level_density
from jineta.core.simulation import run


def exponential_density(altitude: float) -> float:
    """Simple exponential atmosphere, decent approximation up to ~10 km."""
    return 1.225 * np.exp(-altitude / 8500.0)


def hit_ground(t, s):
    return s[2] > 0


MASS, DRAG_AREA = 10.0, 0.001  # [kg], [m^2]: a fifth of the drag area of example 1
speed, elevation = 800.0, np.radians(70.0)
y0 = np.array([0, 0, 0, speed * np.cos(elevation), 0, -speed * np.sin(elevation)])

density_models = {
    "Constant density": sea_level_density,
    "Exponential density": exponential_density,
    "ISA density": isa_density,
}
cases = [("Vacuum", PointMass(mass=MASS, drag_area=0.0))]
cases += [
    (label, PointMass(mass=MASS, drag_area=DRAG_AREA, density=model))
    for label, model in density_models.items()
]

fig, (ax_path, ax_density) = plt.subplots(1, 2, figsize=(12, 5))

for label, body in cases:
    times, states = run(body.derivative, y0, dt=0.01, t_end=400.0, stop=hit_ground)
    max_altitude = -states[:, 2].min()
    print(
        f"{label:22s} range = {states[-1, 0]:7.0f} m   "
        f"max altitude = {max_altitude:6.0f} m   flight time = {times[-1]:5.1f} s"
    )
    ax_path.plot(states[:, 0] / 1000, -states[:, 2] / 1000, label=label)

ax_path.set_xlabel("Downrange distance [km]")
ax_path.set_ylabel("Altitude [km]")
ax_path.set_title("Trajectory vs. atmosphere model")
ax_path.grid(True)
ax_path.legend()

altitudes = np.linspace(0.0, 20_000.0, 201)
for i, (label, model) in enumerate(density_models.items(), start=1):
    densities = [model(h) for h in altitudes]
    ax_density.semilogx(densities, altitudes / 1000, color=f"C{i}", label=label)

ax_density.axhline(TROPOPAUSE_ALTITUDE / 1000, color="gray", linestyle="--")
ax_density.annotate("tropopause", (0.1, TROPOPAUSE_ALTITUDE / 1000 + 0.3))
ax_density.set_xlabel("Air density [kg/m^3]")
ax_density.set_ylabel("Altitude [km]")
ax_density.set_title("Density profile")
ax_density.grid(True)
ax_density.legend()

plt.tight_layout()
plt.show()