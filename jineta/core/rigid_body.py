"""Full 6-DOF rigid body: translation and rotation in one 13-element state.

State layout: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    - position and velocity in NED axes
    - quaternion rotates BODY -> NED
    - angular rate omega = [p, q, r] in body axes
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from jineta.core import quaternion
from jineta.core.point_mass import GRAVITY

# (t, state) -> (force in body axes [N], moment in body axes [N·m]).
# Gravity is NOT included: the body adds it itself.
ForceMomentModel = Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray]]


def zero_force_moment(t: float, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No external forces or moments (only gravity acts on the body)."""
    return np.zeros(3), np.zeros(3)


def normalize_attitude(state: np.ndarray) -> np.ndarray:
    """Return a copy of the state with its quaternion renormalised to unit norm.

    Use as the post_step hook of simulation.run to cancel the drift that RK4
    introduces in the quaternion norm.
    """
    state = state.copy()
    state[6:10] = quaternion.normalize(state[6:10])
    return state


@dataclass
class RigidBody:
    """Rigid body with mass, inertia tensor and a force/moment model."""

    mass: float  # [kg]
    inertia: np.ndarray  # 3x3 inertia tensor in body axes [kg·m^2]
    force_moment: ForceMomentModel = zero_force_moment
    inertia_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.inertia = np.asarray(self.inertia, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt for the 13-element state."""
        velocity = state[3:6]
        q = state[6:10]
        omega = state[10:13]

        force_body, moment_body = self.force_moment(t, state)

        # Translation: rotate body force to NED, then add gravity (+z is down).
        acceleration = quaternion.to_matrix(q) @ force_body / self.mass
        acceleration = acceleration + np.array([0.0, 0.0, GRAVITY])

        # Rotation: kinematics + Euler's equations (same as RotatingBody).
        q_dot = quaternion.derivative(q, omega)
        omega_dot = self.inertia_inv @ (
            moment_body - np.cross(omega, self.inertia @ omega)
        )

        return np.concatenate([velocity, acceleration, q_dot, omega_dot])
