"""Tumbling projectile: spinning about the intermediate axis does not alter the fall.

A torque-free body launched in vacuum tumbles wildly (Dzhanibekov effect), yet
its centre of mass follows the exact parabola of a point mass: with no external
force, rotation and translation are decoupled.
"""

import matplotlib.pyplot as plt
import numpy as np

from jineta.core import quaternion
from jineta.core.point_mass import GRAVITY
from jineta.core.rigid_body import RigidBody, normalize_attitude
from jineta.core.simulation import run

speed, elevation = 70.0, np.radians(70.0)
vx, vz = speed * np.cos(elevation), -speed * np.sin(elevation)  # NED: up is -z

body = RigidBody(mass=1.0, inertia=np.diag([1.0, 2.0, 3.0]))  # axis 2 is intermediate
# Launched nose-north, spinning about axis 2 with a tiny perturbation on axes 1 and 3.
y0 = np.array([0, 0, 0, vx, 0, vz, 1, 0, 0, 0, 0.01, 2.0, 0.01])


def hit_ground(t, s):
    return s[2] > 0


times, states = run(
    body.derivative, y0, dt=0.01, t_end=60.0, stop=hit_ground, post_step=normalize_attitude
)

# Reference: the parabola a point mass would follow in vacuum.
x_ref = vx * times
z_ref = vz * times + 0.5 * GRAVITY * times**2
deviation = np.max(np.abs(states[:, [0, 2]] - np.column_stack([x_ref, z_ref])))

fig, (ax_path, ax_rates) = plt.subplots(1, 2, figsize=(13, 5))

ax_path.plot(states[:, 0], -states[:, 2], label="Tumbling body")
ax_path.plot(x_ref, -z_ref, "k--", label="Point-mass parabola")
for i in range(0, len(times), 50):  # nose direction every 0.5 s (projected on x-z plane)
    nose = quaternion.to_matrix(states[i, 6:10]) @ np.array([1.0, 0.0, 0.0])
    start = np.array([states[i, 0], -states[i, 2]])
    end = start + (0.04 * states[:, 0].max() * np.array([nose[0], -nose[2]]))
    ax_path.plot(*zip(start, end), color="tab:red", linewidth=1.5)
ax_path.plot([], [], color="tab:red", label="Nose direction")
ax_path.set_xlabel("Downrange distance [m]")
ax_path.set_ylabel("Altitude [m]")
ax_path.set_title(f"Centre of mass path (max deviation {deviation:.1e} m)")
ax_path.axis("equal")
ax_path.grid(True)
ax_path.legend()

for i, label in enumerate(["p (axis 1)", "q (axis 2, intermediate)", "r (axis 3)"]):
    ax_rates.plot(times, states[:, 10 + i], label=label)
ax_rates.set_xlabel("Time [s]")
ax_rates.set_ylabel("Angular rate in body axes [rad/s]")
ax_rates.set_title("Angular rates in body axes (tumbling)")
ax_rates.grid(True)
ax_rates.legend()

plt.tight_layout()
plt.show()