"""Iris quadrotor under the velocity autopilot: go North, stop, and find its top speed.

The Iris (PX4 SITL profile) hovers at 100 m. At t = 1 s it is asked to fly North at
10 m/s, and at t = 10 s to stop. The plots show the whole cascade at work: the
velocity follows the command, the tilt is how the thrust is leaned to accelerate and
brake, and the rotor speeds are what the mixer finally asks of each motor.

A second, separate run asks for 40 m/s, far more than the vehicle can do: the speed
it settles at, with the tilt at its 45 degree limit, is the top speed of the model.
"""

import matplotlib.pyplot as plt
import numpy as np

from jineta.core import quaternion
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS

DT = 0.004  # 250 Hz, simulation and control
HOVER_100M = np.array([0, 0, -100.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
ROTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def tilt_deg(attitudes):
    """Angle between the thrust axis and the vertical for each quaternion [deg]."""
    return np.degrees(
        [np.arccos(np.clip(quaternion.to_matrix(q)[2, 2], -1, 1)) for q in attitudes]
    )


def pitch_deg(attitudes):
    """Pitch angle (Euler, for display only) for each quaternion [deg]: negative is
    nose down. sin(pitch) = -R[2, 0], the up component of the nose."""
    return np.degrees([-np.arcsin(quaternion.to_matrix(q)[2, 0]) for q in attitudes])


# --- Run 1: go North at 10 m/s, then stop -------------------------------------------
quad = Multirotor(IRIS)
autopilot = Autopilot(quad, IRIS_GAINS, DT)
commands, rotor_speeds = [], []


def mission(t, state):
    """Velocity command over time, then the autopilot; logs what it asked for."""
    north = 10.0 if 1.0 <= t < 10.0 else 0.0
    autopilot.velocity_command = np.array([north, 0.0, 0.0])
    autopilot(t, state)
    commands.append(north)
    rotor_speeds.append(quad.rotor_speed_command.copy())


times, states = run(
    quad.derivative,
    HOVER_100M,
    dt=DT,
    t_end=20.0,
    pre_step=mission,
    post_step=normalize_attitude,
)
held = times[:-1]  # commands are held from each step start to the next
rotor_speeds = np.array(rotor_speeds)

# --- Run 2: ask for far more than it can do ----------------------------------------
fast_quad = Multirotor(IRIS)
fast_autopilot = Autopilot(fast_quad, IRIS_GAINS, 0.01)
fast_autopilot.velocity_command = np.array([40.0, 0.0, 0.0])
_, fast = run(
    fast_quad.derivative,
    HOVER_100M,
    dt=0.01,
    t_end=40.0,
    pre_step=fast_autopilot,
    post_step=normalize_attitude,
)
print(
    f"Top speed: {fast[-1, 3]:.1f} m/s ({3.6 * fast[-1, 3]:.0f} km/h) "
    f"at {tilt_deg(fast[-1:, 6:10])[0]:.1f} deg tilt"
)
print(f"Run 1: altitude change at most {np.abs(states[:, 2] + 100).max():.2f} m")

# --- Plots ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(8, 9))

axes[0].step(held, commands, where="post", color="#888780", ls="--", label="Command")
axes[0].plot(times, states[:, 3], color=ROTOR_COLORS[0], lw=2, label="Iris")
axes[0].set_ylabel("North speed [m/s]")
axes[0].legend()

axes[1].plot(times, pitch_deg(states[:, 6:10]), color=ROTOR_COLORS[0], lw=2)
axes[1].axhline(0.0, color="#888780", lw=1)
axes[1].set_ylabel("Pitch [deg]")

# In this manoeuvre (pitch only) the two front rotors turn at the same speed, and
# so do the two back ones: dashed lines (cw rotors) lie on top of solid ones (ccw).
names = [
    "0 front right (ccw)",
    "1 back left (ccw)",
    "2 front left (cw)",
    "3 back right (cw)",
]
styles = ["-", "-", "--", "--"]
for i, (name, color, style) in enumerate(zip(names, ROTOR_COLORS, styles, strict=True)):
    axes[2].plot(held, rotor_speeds[:, i], color=color, ls=style, lw=1.5, label=name)
axes[2].axhline(IRIS.max_rotor_speed, color="#888780", lw=1, ls=":")
axes[2].set_ylabel("Rotor speed [rad/s]")
axes[2].legend(ncol=2, fontsize="small")

axes[3].plot(times, -states[:, 2] - 100.0, color=ROTOR_COLORS[0], lw=2)
axes[3].set_ylabel("Altitude change [m]")
axes[3].set_xlabel("Time [s]")

for ax in axes:
    ax.grid(True, alpha=0.3)
fig.suptitle("Iris velocity step: 10 m/s North at t = 1 s, stop at t = 10 s")
fig.tight_layout()
plt.show()