"""Collisions of a flying vehicle with the room, its obstacles, gates and floor.

What is checked at every sample of a flight (the states that simulation.run keeps):
  - walls, ceiling, obstacles and gate frames, with the vehicle's spheres (its
    hitbox: centres in body axes and radii), placed by its position and attitude.
    A sphere touches a shape when the signed distance of its centre is less than
    its radius;
  - the floor, with the vehicle's floor points (its landing gear). A vehicle starts
    a lap standing on its gear, so the floor only counts once it has left it: from
    the first sample with every floor point above the floor, any floor point
    touching it again is a crash.

Margin: for each check, the room left before touching. For walls and obstacles it is
the distance of each sphere's centre minus its radius (the smallest over the
spheres); for the floor, the height of the lowest floor point. Negative means
touching.

Inside the step: the first sample that touches is already past the moment of
contact. That moment is estimated by linear interpolation of the margin between the
last sample that did not touch and this one. It is exact for a vehicle flying
straight at a flat face (the margin falls linearly); otherwise the error shrinks
like dt^2.

Tunnelling: only samples are checked, so a contact that starts and ends between two
samples is missed. Moving s = v dt per step:
  - straight through a slab of thickness w: missed when s > 2r + w (the stretch of
    path inside the slab inflated by the radius r is 2r + w long);
  - grazing a rounded edge of the inflated shape (radius rho: r for the edge of a
    box, r + R for a column of radius R): a contact up to rho - sqrt(rho^2 - s^2/4)
    ~ s^2 / (8 rho) deep can go unseen.
For the Iris (r = 0.384 m) at 100 Hz: 79 m/s to go through a thin wall unseen (its
top speed is 20.7 m/s); at 20 m/s a graze up to 1.3 cm deep can be missed.

The floor of the room is z = 0 (campo/indoor.py). This module only needs `core`:
the hitbox and the floor points come in as arrays.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from jineta.campo.indoor import Room
from jineta.core import quaternion


class Obstacle(Protocol):
    """Anything with a signed distance (Box, Cylinder)."""

    def signed_distance(self, points: npt.ArrayLike) -> float | np.ndarray: ...


@dataclass(frozen=True)
class Collision:
    """A contact: what was touched, when and where.

    kind: "room" (a wall or the ceiling), "obstacle", "gate" (its frame) or "floor".
    index: which obstacle or gate (its position in its list); None for the room and
        the floor.
    time: the moment of contact [s], interpolated inside the step.
    position: the centre of mass (NED) at that moment [m], interpolated.
    """

    kind: str
    index: int | None
    time: float
    position: np.ndarray


def _rows(value: npt.ArrayLike, what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{what} must have shape (n, 3), n >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite")
    array.flags.writeable = False
    return array


@dataclass
class CollisionMonitor:
    """Watches a flight for collisions; callable as the `stop` of simulation.run.

    room: the room (its walls and ceiling are checked with the spheres).
    obstacles: shapes with a signed distance, in the order `Collision.index` uses.
    sphere_centers, sphere_radii: the vehicle's hitbox (centres in body axes [m]).
    floor_points: the points that touch the floor (landing gear, body axes [m]).
    gates: gates whose frames are checked like obstacles (anything with a signed
        distance), reported as kind "gate".

    After a call that found a contact, `collision` holds it and every later call
    returns True. `airborne` tells whether the vehicle has left the floor.
    """

    room: Room
    obstacles: Sequence[Obstacle]
    sphere_centers: npt.ArrayLike
    sphere_radii: npt.ArrayLike
    floor_points: npt.ArrayLike
    gates: Sequence[Obstacle] = ()
    collision: Collision | None = field(default=None, init=False)
    airborne: bool = field(default=False, init=False)
    _previous: tuple[float, np.ndarray, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.obstacles = tuple(self.obstacles)
        self.gates = tuple(self.gates)
        self.sphere_centers = _rows(self.sphere_centers, "sphere_centers")
        radii = np.array(self.sphere_radii, dtype=float)
        if radii.shape != (len(self.sphere_centers),):
            raise ValueError("sphere_radii must have one value per centre")
        if not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise ValueError("every sphere radius must be finite and > 0")
        radii.flags.writeable = False
        self.sphere_radii = radii
        self.floor_points = _rows(self.floor_points, "floor_points")

    def margins(self, state: np.ndarray) -> np.ndarray:
        """Margin [m] of every check for one state:
        [room, obstacle 0, ..., gate 0, ..., floor].

        Body points go to NED as p = x + R(q) r; with the points as rows of an array
        that is x + points @ R^T.
        """
        position = state[0:3]
        rotation = quaternion.to_matrix(state[6:10])
        centers = position + self.sphere_centers @ rotation.T
        radii = self.sphere_radii
        room = np.min(self.room.wall_clearance(centers) - radii)
        obstacles = [
            np.min(np.asarray(o.signed_distance(centers)) - radii)
            for o in (*self.obstacles, *self.gates)
        ]
        feet = position + self.floor_points @ rotation.T
        floor = np.min(-feet[:, 2])  # height above the floor z = 0 (up is -z)
        return np.array([room, *obstacles, floor])

    def __call__(self, t: float, state: np.ndarray) -> bool:
        """Check one sample; True once a collision has been found."""
        if self.collision is not None:
            return True
        margins = self.margins(state)
        touching = margins < 0.0
        if not self.airborne:
            touching[-1] = False
        if np.any(touching):
            self.collision = self._first_contact(t, state, margins, touching)
            return True
        if margins[-1] > 0.0:
            self.airborne = True
        self._previous = (t, state[0:3].copy(), margins)
        return False

    def reset(self) -> None:
        """Forget the flight: no collision, not airborne, no previous sample."""
        self.collision = None
        self.airborne = False
        self._previous = None

    def _first_contact(
        self, t: float, state: np.ndarray, margins: np.ndarray, touching: np.ndarray
    ) -> Collision:
        """The earliest contact among the checks touching now, interpolated between
        the previous sample (margin m0 >= 0) and this one (m1 < 0): the margin is 0
        at the fraction m0 / (m0 - m1) of the step. Without a previous sample (the
        very first one already touches), the contact is put at this sample."""
        position = state[0:3]
        if self._previous is None:
            fractions = np.where(touching, 1.0, np.inf)
            t0, p0 = t, position
        else:
            t0, p0, m0 = self._previous
            with np.errstate(divide="ignore", invalid="ignore"):
                fractions = np.where(touching, m0 / (m0 - margins), np.inf)
        i = int(np.argmin(fractions))
        f = float(fractions[i])
        n_obstacles, n_gates = len(self.obstacles), len(self.gates)
        if i == 0:
            kind, index = "room", None
        elif i <= n_obstacles:
            kind, index = "obstacle", i - 1
        elif i <= n_obstacles + n_gates:
            kind, index = "gate", i - 1 - n_obstacles
        else:
            kind, index = "floor", None
        contact = p0 + f * (position - p0)
        contact.flags.writeable = False
        return Collision(kind, index, t0 + f * (t - t0), contact)
