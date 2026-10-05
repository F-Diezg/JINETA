"""Dzhanibekov effect: rotation about the intermediate axis is unstable."""

import matplotlib.pyplot as plt
import numpy as np

from jineta.core.rotating_body import RotatingBody
from jineta.core.simulation import run

body = RotatingBody(inertia=np.diag([1.0, 2.0, 3.0]))  # axis 2 is intermediate
y0 = np.array([1.0, 0, 0, 0, 0.01, 2.0, 0.01])  # spin about axis 2 + tiny perturbation

times, states = run(body.derivative, y0, dt=0.01, t_end=60.0)

for i, label in enumerate(["p (axis 1, min inertia)", "q (axis 2, intermediate)",
                           "r (axis 3, max inertia)"]):
    plt.plot(times, states[:, 4 + i], label=label)

plt.xlabel("Time [s]")
plt.ylabel("Angular rate in body axes [rad/s]")
plt.title("Dzhanibekov effect: torque-free spin about the intermediate axis")
plt.grid(True)
plt.legend()
plt.show()