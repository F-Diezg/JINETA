"""Multirotor airframe: rotor layout, rotor constants and the motor mixer.

Each rotor pushes along -z_body (up, for a level vehicle) with thrust T_i and twists
the body with a reaction torque Q_i = moment_constant * T_i about z_body. The mixer
is the linear map between the rotor thrusts and their total effect on the vehicle,
the "wrench" [T_total, Mx, My, Mz] (total thrust along -z_body, moments in body
axes):

    wrench = A @ thrusts        A (4 x n rotors) is the allocation matrix

    row 0:  1                       every rotor adds to the total thrust
    row 1: -y_i                     roll: a rotor on the left (y < 0) rolls right
    row 2:  x_i                     pitch: a rotor ahead of the CoM pitches nose up
    row 3:  spin_i * moment_constant   yaw: reaction torque of the rotor

Rows 1 and 2 are the moment r_i x F_i of a force F_i = [0, 0, -T_i]. The height z_i
of the rotor does not appear: a force along z_body has no lever arm along z_body.

Rotor speed and air density do not appear here either: the mixer works with thrusts,
and the conversion thrust <-> rotor speed belongs to the force model.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.inertia import inertia_tensor


@dataclass(frozen=True)
class Rotor:
    """One rotor: where it is and which way it turns.

    position: [x, y, z] in body axes (FRD) relative to the centre of mass [m].
    spin: +1 if it turns counter-clockwise seen from above (PX4/Gazebo "ccw"),
        -1 if clockwise. The body reacts the opposite way, so a +1 rotor yaws
        the body to the right (positive r).
    """

    position: np.ndarray
    spin: int

    def __post_init__(self) -> None:
        position = np.array(self.position, dtype=float)
        if position.shape != (3,):
            raise ValueError("rotor position must have 3 components [x, y, z]")
        if self.spin not in (1, -1):
            raise ValueError("rotor spin must be +1 (ccw) or -1 (cw)")
        position.setflags(write=False)
        # Frozen dataclass: fields can only be set through object.__setattr__.
        object.__setattr__(self, "position", position)


def allocation_matrix(rotors: Sequence[Rotor], moment_constant: float) -> np.ndarray:
    """Allocation matrix A (4 x n): wrench [T, Mx, My, Mz] = A @ rotor thrusts."""
    columns = [
        [1.0, -rotor.position[1], rotor.position[0], rotor.spin * moment_constant]
        for rotor in rotors
    ]
    return np.array(columns).T


@dataclass(frozen=True)
class MultirotorParams:
    """Physical parameters of a multirotor: only data, validated once.

    thrust_coefficient: k_T in T = k_T * Omega^2 [N/(rad/s)^2] (PX4 motorConstant).
    moment_constant: reaction torque / thrust, Q = moment_constant * T [m]
        (PX4 momentConstant).
    max_rotor_speed: Omega_max [rad/s].
    """

    mass: float  # [kg]
    inertia: np.ndarray  # 3x3 about the centre of mass, body axes [kg·m^2]
    rotors: tuple[Rotor, ...]
    thrust_coefficient: float
    moment_constant: float
    max_rotor_speed: float
    allocation: np.ndarray = field(init=False, repr=False)
    allocation_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        positive = ("mass", "thrust_coefficient", "moment_constant", "max_rotor_speed")
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        rotors = tuple(self.rotors)
        allocation = allocation_matrix(rotors, self.moment_constant)
        if np.linalg.matrix_rank(allocation) < 4:
            raise ValueError(
                "rotor layout cannot control thrust, roll, pitch and yaw independently"
            )
        # Pseudo-inverse: the exact inverse for 4 rotors, the minimum-effort
        # solution for more (hexacopter, octocopter).
        allocation_inv = np.linalg.pinv(allocation)
        allocation.setflags(write=False)
        allocation_inv.setflags(write=False)
        object.__setattr__(self, "inertia", inertia_tensor(self.inertia))
        object.__setattr__(self, "rotors", rotors)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "allocation_inv", allocation_inv)

    @property
    def max_rotor_thrust(self) -> float:
        """Thrust of one rotor at full speed [N]."""
        return self.thrust_coefficient * self.max_rotor_speed**2

    def wrench(self, thrusts: npt.ArrayLike) -> np.ndarray:
        """Total effect [T, Mx, My, Mz] of the given rotor thrusts [N]."""
        return self.allocation @ np.asarray(thrusts, dtype=float)

    def rotor_thrusts(self, wrench: npt.ArrayLike) -> np.ndarray:
        """Rotor thrusts [N] that produce the requested wrench [T, Mx, My, Mz].

        Each thrust is clipped to [0, max_rotor_thrust]: a rotor cannot pull down
        or spin faster than its maximum. If clipping happens, the wrench actually
        produced differs from the one requested.
        """
        thrusts = self.allocation_inv @ np.asarray(wrench, dtype=float)
        return np.clip(thrusts, 0.0, self.max_rotor_thrust)
