"""The floor under many drones at once: `jinete.contact` on a stack of states.

The same model as `jinete.contact.contact_forces` (a spring-damper at each point of
the landing gear, never pulling, with smoothed Coulomb friction), for N drones that
share one landing gear. The reference skips a point that does not touch the floor;
here every point is computed for every drone and the ones not touching get exactly
zero force (adding zero changes nothing), so all the drones follow the same
operations whether they are flying or standing.
"""

from dataclasses import dataclass

import numpy as np

from jineta.core.vector import cross, matvec
from jineta.escuadron import quaternion
from jineta.jinete.contact import LandingGear


def contact_forces(
    gear: LandingGear, states: np.ndarray, floor_z: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Force [N] and moment [N·m] of the floor on each body, both in body axes,
    shapes (N, 3), for the states (N, 13).

    For each point r of the gear: its position p = x + R r and velocity
    v = v_cm + R (omega x r) in NED; depth = p_z - floor_z (positive below the
    floor); normal force N = max(k depth + c v_z, 0) upwards while depth > 0;
    friction -mu N v_t / sqrt(|v_t|^2 + v_s^2) against (v_north, v_east).
    """
    position, velocity = states[:, 0:3], states[:, 3:6]
    rotation = quaternion.to_matrix(states[:, 6:10])
    rotation_t = rotation.swapaxes(-1, -2)  # NED -> body
    omega = states[:, 10:13]
    force_ned = np.zeros((len(states), 3))
    moment_body = np.zeros((len(states), 3))
    for r in gear.points:
        p = position + matvec(rotation, r)
        depth = p[:, 2] - floor_z
        v = velocity + matvec(rotation, cross(omega, r))
        normal = np.where(
            depth > 0.0,
            np.maximum(gear.stiffness * depth + gear.damping * v[:, 2], 0.0),
            0.0,
        )
        slide = v[:, 0:2]
        speed_squared = slide[:, 0] * slide[:, 0] + slide[:, 1] * slide[:, 1]
        friction = (
            -gear.friction
            * normal[:, None]
            * slide
            / np.sqrt(speed_squared + gear.slip_speed**2)[:, None]
        )
        f_ned = np.column_stack([friction, -normal])
        force_ned = force_ned + f_ned
        moment_body = moment_body + cross(r, matvec(rotation_t, f_ned))
    return matvec(rotation_t, force_ned), moment_body


@dataclass
class BatchGroundContact:
    """The floor as a batch force model: callable (t, states) -> (forces, moments)
    in body axes, (N, 3) each, to be added to the vehicles' own forces."""

    gear: LandingGear
    floor_z: float = 0.0

    def __call__(
        self, t: float | np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return contact_forces(self.gear, states, self.floor_z)
