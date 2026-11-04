"""Contact with the floor: a spring-damper at each contact point, with friction.

A rigid floor would need the simulator to stop the body exactly at the surface,
which a fixed-step integrator cannot do. The usual way out is the penalty method:
the floor lets a contact point sink a little (a millimetre) and pushes back like a
very stiff spring with a damper,

    N = k * depth + c * (sinking speed),   only while the point is below the floor,

and never pulls (N >= 0): a point leaving the floor is free as soon as the spring
and the damper together stop pushing. Along the floor, friction opposes the
sliding velocity with at most mu * N (Coulomb). Exact Coulomb friction jumps from
-mu N to +mu N when the velocity changes sign, which a fixed-step integrator cannot
follow, so it is smoothed below a slip speed v_s:

    F_t = -mu * N * v_t / sqrt(|v_t|^2 + v_s^2),

which is mu * N against the motion when sliding fast, and fades to zero (like a
viscous damper) as the point comes to rest.

A vehicle carries its contact points (its landing gear, in body axes) as profile
data, like its rotors; `GroundContact` turns them into the force and moment on the
body, in body axes, and the vehicle adds them to its own forces. The floor is the
plane z = floor_z (NED; the room's floor is z = 0).

Stability at a fixed step: the contact is the stiffest part of the model. With
explicit RK4 a decaying mode with rate lambda stays stable only while |lambda dt|
is below ~2.8. Two contact modes set the limit: the overdamped tilt of the body on
its springs (grows with the damping ratio) and the yaw of the body braked by the
friction of its feet (rate ~ mu N r^2 / (v_s I_zz), grows as v_s shrinks). The
Iris gear is chosen inside that limit at 100 Hz, and a test checks it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.vector import cross


@dataclass(frozen=True)
class LandingGear:
    """Contact points of a vehicle and how they touch the floor.

    points: positions in body axes (FRD) relative to the centre of mass [m], one
        row per point.
    stiffness: k of each point's spring [N/m].
    damping: c of each point's damper [N·s/m].
    friction: Coulomb coefficient mu between a point and the floor.
    slip_speed: v_s, the sliding speed below which friction fades [m/s].
    """

    points: np.ndarray
    stiffness: float
    damping: float
    friction: float
    slip_speed: float

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points must have shape (n, 3), n >= 1")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        points.flags.writeable = False
        object.__setattr__(self, "points", points)
        if not self.stiffness > 0.0:
            raise ValueError("stiffness must be > 0")
        if not self.damping >= 0.0:
            raise ValueError("damping must be >= 0")
        if not self.friction >= 0.0:
            raise ValueError("friction must be >= 0")
        if not self.slip_speed > 0.0:
            raise ValueError("slip_speed must be > 0")

    @classmethod
    def from_damping_ratio(
        cls,
        points: npt.ArrayLike,
        mass: float,
        sink: float,
        damping_ratio: float,
        friction: float,
        slip_speed: float,
    ) -> "LandingGear":
        """Gear sized for a vehicle: at rest on its n points it sinks `sink` [m],
        and its vertical bounce has the damping ratio `damping_ratio`.

        All n springs carry the weight together: n k sink = m g, so
        k = m g / (n sink). The vertical bounce of the whole vehicle is a
        mass-spring-damper with n k and n c, whose damping ratio is
        zeta = n c / (2 sqrt(n k m)), so c = 2 zeta sqrt(n k m) / n.
        """
        n = len(np.atleast_2d(points))
        if not mass > 0.0 or not sink > 0.0:
            raise ValueError("mass and sink must be > 0")
        k = mass * GRAVITY / (n * sink)
        c = 2.0 * damping_ratio * np.sqrt(n * k * mass) / n
        return cls(points, k, c, friction, slip_speed)


def contact_forces(
    gear: LandingGear,
    state: np.ndarray,
    floor_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Force [N] and moment [N·m] of the floor on the body, both in body axes.

    For each point: its position p = x + R r and velocity v = v_cm + R (omega x r)
    in NED; depth = p_z - floor_z (positive below the floor); normal force
    N = max(k depth + c v_z, 0) upwards (-z); friction against (v_north, v_east).
    """
    position, velocity = state[0:3], state[3:6]
    rotation = quaternion.to_matrix(state[6:10])
    omega = state[10:13]
    force_ned = np.zeros(3)
    moment_body = np.zeros(3)
    for r in gear.points:
        p = position + rotation @ r
        depth = p[2] - floor_z
        if depth <= 0.0:
            continue
        v = velocity + rotation @ cross(omega, r)
        normal = max(gear.stiffness * depth + gear.damping * v[2], 0.0)
        if normal == 0.0:
            continue
        slide = v[:2]
        friction = (
            -gear.friction
            * normal
            * slide
            / np.sqrt(slide @ slide + gear.slip_speed**2)
        )
        f_ned = np.array([friction[0], friction[1], -normal])
        f_body = rotation.T @ f_ned
        force_ned = force_ned + f_ned
        moment_body = moment_body + cross(r, f_body)
    return rotation.T @ force_ned, moment_body


@dataclass
class GroundContact:
    """The floor as a force model: callable (t, state) -> (force, moment) in body
    axes, to be added to a vehicle's own forces."""

    gear: LandingGear
    floor_z: float = 0.0

    def __call__(self, t: float, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return contact_forces(self.gear, state, self.floor_z)


def box_bottom_corners(size: Sequence[float]) -> np.ndarray:
    """The four bottom corners of a box centred on the centre of mass, in FRD body
    axes: (+-length/2, +-width/2, +height/2) (z down, so the bottom is +z)."""
    length, width, height = size
    return np.array(
        [
            [sx * length / 2, sy * width / 2, height / 2]
            for sx in (1, -1)
            for sy in (1, -1)
        ]
    )


def resting_height(gear: LandingGear, mass: float) -> float:
    """Height [m] of the centre of mass above the floor when the vehicle stands
    level and still on its gear, every point carrying an equal share of the weight
    (gear symmetric about the centre of mass, as the Iris's).

    The lowest points (largest body z) touch; each of the n springs sinks
    m g / (n k), so the centre of mass is that much less than their height above it.
    Only the lowest points carry weight: a level vehicle stands on those.
    """
    z = gear.points[:, 2]
    lowest = z.max()
    n = int(np.sum(np.isclose(z, lowest)))
    sink = mass * GRAVITY / (n * gear.stiffness)
    return float(lowest - sink)
