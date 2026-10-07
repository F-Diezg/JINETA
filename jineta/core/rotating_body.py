"""Rotational dynamics of a rigid body (attitude only, no translation).

State vector: [qw, qx, qy, qz, p, q, r]
    - quaternion (body -> NED)
    - angular velocity in body axes [rad/s]
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from jineta.core import quaternion
from jineta.core.inertia import inertia_tensor

# Torque model: (t, state) -> torque in body axes [N·m]
TorqueModel = Callable[[float, np.ndarray], np.ndarray]


def zero_torque(t: float, state: np.ndarray) -> np.ndarray:
    """No external torque."""
    return np.zeros(3)


@dataclass
class RotatingBody:
    """Rigid body that only rotates, governed by Euler's equations."""

    inertia: np.ndarray  # 3x3 inertia tensor in body axes [kg·m^2]
    torque: TorqueModel = zero_torque
    inertia_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.inertia = inertia_tensor(self.inertia)
        self.inertia_inv = np.linalg.inv(self.inertia)

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt = [q_dot, omega_dot]."""
        q = state[0:4]
        omega = state[4:7]

        q_dot = quaternion.derivative(q, omega)

        moment = self.torque(t, state)
        angular_momentum = self.inertia @ omega
        omega_dot = self.inertia_inv @ (moment - np.cross(omega, angular_momentum))

        return np.concatenate([q_dot, omega_dot])
