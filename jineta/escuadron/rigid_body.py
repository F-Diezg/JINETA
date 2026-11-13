"""Rigid bodies for many drones at once: the 6-DOF equations on a stack of states.

The same physics as `core.rigid_body.RigidBody`, written on arrays: the states of
N bodies are the rows of one (N, 13) array, laid out as the reference state

    [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]   (NED, quaternion BODY -> NED,
                                                      angular rate in body axes)

and every operation acts on all the rows at once. The bodies of a batch share one
mass and one inertia, unless `set_bodies` gives some of them their own (phase
12: drones randomised around their profile); then mass and inertia become one
per body, (N,) and (N, 3, 3).

Each row depends only on its own state: no sum, mean or matrix product mixes the
rows, so a drone gives the same numbers alone or in a batch of any size. The
reference engine stays the truth; the tests compare every drone with it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.environment import GRAVITY
from jineta.core.inertia import inertia_tensor
from jineta.core.vector import cross, matvec
from jineta.escuadron import quaternion

# (t, states (N, 13)) -> (forces (N, 3), moments (N, 3)), both in body axes [N],
# [N·m]. Gravity is NOT included: the bodies add it themselves. t is one time for
# every drone or one per drone, shape (N,).
BatchForceMomentModel = Callable[
    [float | np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]
]


def zero_force_moment(
    t: float | np.ndarray, states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """No external forces or moments on any body (only gravity acts)."""
    n = len(states)
    return np.zeros((n, 3)), np.zeros((n, 3))


def normalize_attitudes(states: np.ndarray) -> np.ndarray:
    """A copy of the states (N, 13 or more) with every quaternion renormalised to
    unit norm (the batch version of `core.rigid_body.normalize_attitude`)."""
    states = states.copy()
    states[:, 6:10] = quaternion.normalize(states[:, 6:10])
    return states


@dataclass
class BatchRigidBody:
    """N rigid bodies with the same mass and inertia, each with its own force and
    moment (from one `force_moment` model that returns them for all the bodies)."""

    mass: float | np.ndarray  # [kg]; (N,) once set per body
    inertia: np.ndarray  # 3x3 in body axes [kg·m^2]; (N, 3, 3) once set per body
    force_moment: BatchForceMomentModel = zero_force_moment
    inertia_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be > 0")
        self.inertia = inertia_tensor(self.inertia)
        inertia_inv = np.linalg.inv(self.inertia)
        inertia_inv.setflags(write=False)
        self.inertia_inv = inertia_inv

    def set_bodies(
        self,
        n: int,
        rows: np.ndarray,
        mass: npt.ArrayLike,
        inertia: npt.ArrayLike,
    ) -> None:
        """Give the bodies `rows` (indices into n bodies) their own mass (k,) [kg]
        and inertia (k, 3, 3) [kg·m^2] (each validated); from the first call the
        batch keeps one mass and one inertia per body, the others keeping the
        shared ones."""
        if np.ndim(self.mass) == 0:
            self.mass = np.full(n, float(self.mass))
            self.inertia = np.tile(self.inertia, (n, 1, 1))
            self.inertia_inv = np.tile(self.inertia_inv, (n, 1, 1))
        mass = np.asarray(mass, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        if np.any(~(mass > 0.0)):
            raise ValueError("mass must be > 0")
        tensors = np.array([inertia_tensor(tensor) for tensor in inertia]).reshape(
            -1, 3, 3
        )
        self.mass[rows] = mass
        self.inertia[rows] = tensors
        if len(tensors):
            self.inertia_inv[rows] = np.linalg.inv(tensors)

    def derivative(self, t: float | np.ndarray, states: np.ndarray) -> np.ndarray:
        """d(states)/dt, shape (N, 13), for the states (N, 13) (or longer: the
        first 13 columns are the bodies', the whole rows go to `force_moment`,
        as in `core.rigid_body.RigidBody.derivative`)."""
        velocity = states[:, 3:6]
        q = states[:, 6:10]
        omega = states[:, 10:13]

        force_body, moment_body = self.force_moment(t, states)

        # Translation: each body force rotated to NED by its own attitude, then
        # gravity (+z is down).
        mass = self.mass if np.ndim(self.mass) == 0 else self.mass[:, None]
        acceleration = matvec(quaternion.to_matrix(q), force_body) / mass
        acceleration[:, 2] += GRAVITY

        # Rotation: kinematics + Euler's equations, one row per body.
        q_dot = quaternion.derivative(q, omega)
        omega_dot = matvec(
            self.inertia_inv,
            moment_body - cross(omega, matvec(self.inertia, omega)),
        )

        return np.concatenate([velocity, acceleration, q_dot, omega_dot], axis=1)
