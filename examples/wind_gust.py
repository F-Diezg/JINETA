"""Response of a falling body to a gust on top of a boundary-layer wind.

A light, draggy body (terminal speed ~18 m/s) is released at rest at 500 m. The mean
wind is a power-law boundary layer from the west; the second case adds a 10 m/s gust
from the west that lasts 6 s. The body does not follow the wind instantly: it lags
behind by a time that depends on its mass and drag.
"""

import matplotlib.pyplot as plt
import numpy as np

from jineta.campo.wind import CombinedWind, DiscreteGust, PowerLawWind
from jineta.core.point_mass import PointMass
from jineta.core.simulation import run


def hit_ground(t, s):
    return s[2] > 0


mean_wind = PowerLawWind(reference_speed=5.0, direction_from_deg=270.0)
gust = DiscreteGust(peak=np.array([0.0, 10.0, 0.0]), start_time=8.0, duration=6.0)

y0 = np.array([0, 0, -500.0, 0, 0, 0])  # released at rest, 500 m up
cases = [
    ("Mean wind only", mean_wind),
    ("Mean wind + gust", CombinedWind([mean_wind, gust])),
]

for label, wind in cases:
    body = PointMass(mass=1.0, drag_area=0.05, wind=wind)
    times, states = run(body.derivative, y0, dt=0.01, t_end=120.0, stop=hit_ground)
    wind_east = [wind(t, s[0:3])[1] for t, s in zip(times, states)]
    print(
        f"{label:22s} drift = {states[-1, 1]:6.1f} m   "
        f"peak east speed = {states[:, 4].max():5.2f} m/s   "
        f"fall time = {times[-1]:5.1f} s"
    )
    (line,) = plt.plot(times, states[:, 4], label=f"{label}: body")
    plt.plot(times, wind_east, "--", color=line.get_color(), label=f"{label}: wind")

plt.axvspan(gust.start_time, gust.start_time + gust.duration, color="gray", alpha=0.15)
plt.xlabel("Time [s]")
plt.ylabel("Eastward speed [m/s]")
plt.title("Body velocity vs. wind at its position")
plt.grid(True)
plt.legend()
plt.show()
