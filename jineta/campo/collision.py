"""Collisions of a flying vehicle with the room, its obstacles, gates and floor.

What is checked at every sample of a flight (the states that simulation.run keeps):
  - walls, ceiling, obstacles and gate frames, with the vehicle's spheres (its
    hitbox: centres in body axes and radii), placed by its position and attitude.
    A sphere touches a shape when the signed distance of its centre is less than
    its radius;
  - the floor, with the vehicle's floor points (its landing gear). A vehicle starts
    a lap standing on its gear, so the floor only counts once it has left it: from
    the first sample with every floor point more than TAKEOFF_CLEARANCE above the
    floor, any floor point touching it again is a crash. (A gear that unloads
    without a take-off, thrust a little below the weight, can lift its feet a
    fraction of a millimetre off the floor; that hop is not a take-off.)

Margin: for each check, the room left before touching. For walls and obstacles it is
the distance of each sphere's centre minus its radius (the smallest over the
spheres); for the floor, the height of the lowest floor point. Negative means
touching.

Inside the step: the first sample that touches is already past the moment of
contact. For the floor that moment is estimated by linear interpolation of the
margin between the last sample that did not touch and this one: exact for a foot
moving straight at the floor, otherwise the error shrinks like dt^2. For walls,
obstacles and gates it is found by the swept check below.

Between the samples (the swept check): a contact can also start and end between
two samples that both touch nothing (a graze of a pillar's edge, or a thin wall
jumped by a fast vehicle). So each sphere is checked along its move between two
samples, taken as a straight line from its centre at one sample to its centre at
the next. A signed distance never changes faster than the point moves, so a sphere
whose gap to a shape is m can move m without touching it: from the first sample,
the move is advanced by the smallest m_i / |move_i| over the spheres, again and
again ("conservative advancement"), until a sphere comes within `SWEEP_TOLERANCE`
of the shape (a contact) or the next sample is reached (none). It never jumps a
contact on the straight moves. When the shape is touched at the next sample, the
last bit, from within SWEEP_TOLERANCE to the surface, is linear interpolation of
the gap, so a flat face approached in a straight line gives the exact moment, as
before; for a curved face approached at a slant this is much better than
interpolating between the samples (at 20 m/s past a pillar, that was up to 3 ms
late). Most of the time the check costs nothing: if the margins at the two samples
add up to more than the longest move, no sphere can touch in between, and nothing
is advanced.

What the straight moves leave out: the true path of a sphere between samples is
curved (the vehicle accelerates and turns). The gap between the curve and its
chord is ~ a dt^2 / 8 for an acceleration a (0.25 mm at 20 m/s^2 and 100 Hz) and
~ r (omega dt)^2 / 8 for a sphere r from the centre of mass turning at omega
(0.04 mm for a rotor at 3.5 rad/s): a graze shallower than that can still be missed.
Without the swept check, moving s = v dt per step, a sample check alone misses:
  - straight through a slab of thickness w: when s > 2r + w (the stretch of path
    inside the slab inflated by the radius r is 2r + w long);
  - grazing a rounded edge of the inflated shape (radius rho: r for the edge of a
    box, r + R for a column of radius R): a contact up to rho - sqrt(rho^2 - s^2/4)
    ~ s^2 / (8 rho) deep (3.1 cm for a rotor sphere of the Iris passing a 5 cm
    pillar at 20 m/s).
The floor needs no swept check: it is a plane, so a foot that moves in a straight
line and is above it at both samples is above it all the way.

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

# A sphere coming within this distance of a shape during the swept check counts as
# touching it [m]. It bounds the work of the advancement near a shape it does not
# quite touch: every advance is at least SWEEP_TOLERANCE / (longest move).
SWEEP_TOLERANCE = 1e-6

# A vehicle has left the floor once every floor point is higher than this [m]. Ten
# times the Iris gear's sink at rest (1 mm): a spring unloading with thrust below
# the weight lifts the feet at most about that much, and a lap flies decimetres up.
TAKEOFF_CLEARANCE = 0.01


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
    returns True. `airborne` tells whether the vehicle has left the floor (all its
    floor points higher than TAKEOFF_CLEARANCE at some sample).
    """

    room: Room
    obstacles: Sequence[Obstacle]
    sphere_centers: npt.ArrayLike
    sphere_radii: npt.ArrayLike
    floor_points: npt.ArrayLike
    gates: Sequence[Obstacle] = ()
    collision: Collision | None = field(default=None, init=False)
    airborne: bool = field(default=False, init=False)
    _previous: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = field(
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
        [room, obstacle 0, ..., gate 0, ..., floor]."""
        return self._margins(*self._placed(state))

    def _placed(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centres of the spheres and the floor points in NED [m], one row each.
        Body points go to NED as p = x + R(q) r; with the points as rows of an
        array that is x + points @ R^T."""
        rotation = quaternion.to_matrix(state[6:10])
        position = state[0:3]
        centers = position + self.sphere_centers @ rotation.T
        feet = position + self.floor_points @ rotation.T
        return centers, feet

    def _distances(self) -> list:
        """The distance function of every check but the floor, in margin order:
        the room's walls and ceiling, the obstacles, the gates."""
        return [
            self.room.wall_clearance,
            *(o.signed_distance for o in (*self.obstacles, *self.gates)),
        ]

    def _margins(self, centers: np.ndarray, feet: np.ndarray) -> np.ndarray:
        radii = self.sphere_radii
        spheres = [
            np.min(np.asarray(distance(centers)) - radii)
            for distance in self._distances()
        ]
        floor = np.min(-feet[:, 2])  # height above the floor z = 0 (up is -z)
        return np.array([*spheres, floor])

    def __call__(self, t: float, state: np.ndarray) -> bool:
        """Check one sample, and the moves of the spheres since the previous one;
        True once a collision has been found."""
        if self.collision is not None:
            return True
        centers, feet = self._placed(state)
        margins = self._margins(centers, feet)
        touching = margins < 0.0
        if not self.airborne:
            touching[-1] = False
        swept = self._swept_fractions(centers, margins, touching)
        if np.any(touching) or np.any(np.isfinite(swept)):
            self.collision = self._first_contact(t, state, margins, touching, swept)
            return True
        if margins[-1] > TAKEOFF_CLEARANCE:
            self.airborne = True
        self._previous = (t, state[0:3].copy(), margins, centers)
        return False

    def _swept_fractions(
        self, centers: np.ndarray, margins: np.ndarray, touching: np.ndarray
    ) -> np.ndarray:
        """For every check but the floor: the fraction of the step at which the
        first sphere touches it, moving in a straight line from the previous
        sample, or inf. Checks that cannot be touched in between are skipped."""
        fractions = np.full(len(margins), np.inf)
        if self._previous is None:
            return fractions
        _, _, previous_margins, previous_centers = self._previous
        move = centers - previous_centers
        lengths = np.sqrt(np.sum(move * move, axis=1))
        longest = np.max(lengths)
        for i, distance in enumerate(self._distances()):
            if previous_margins[i] + margins[i] > longest:
                continue  # no sphere can touch it in between
            s, gap = self._advance(distance, previous_centers, move, lengths)
            if touching[i]:
                # Within SWEEP_TOLERANCE of the shape at s and inside it at the
                # sample: the last bit by linear interpolation of the gap, which
                # is exact for a flat face.
                s = s + (1.0 - s) * gap / (gap - margins[i])
            fractions[i] = s
        return fractions

    def _advance(
        self,
        distance,
        start: np.ndarray,
        move: np.ndarray,
        lengths: np.ndarray,
    ) -> tuple[float, float]:
        """Conservative advancement of the spheres along their moves: the first
        fraction s in [0, 1] at which a sphere comes within SWEEP_TOLERANCE of the
        shape, and that sphere's gap there [m]; (inf, nan) if none does."""
        with np.errstate(divide="ignore"):
            inverse = np.where(lengths > 0.0, 1.0 / lengths, np.inf)
        s = 0.0
        while s <= 1.0:
            gaps = np.asarray(distance(start + s * move)) - self.sphere_radii
            gap = float(np.min(gaps))
            if gap <= SWEEP_TOLERANCE:
                return s, gap
            s += float(np.min(gaps * inverse))
        return np.inf, np.nan

    def reset(self) -> None:
        """Forget the flight: no collision, not airborne, no previous sample."""
        self.collision = None
        self.airborne = False
        self._previous = None

    def _first_contact(
        self,
        t: float,
        state: np.ndarray,
        margins: np.ndarray,
        touching: np.ndarray,
        swept: np.ndarray,
    ) -> Collision:
        """The earliest contact of this step, among every check: for the walls,
        obstacles and gates, the fraction their swept check found (touching now or
        only in between); for the floor, interpolated between the previous sample
        (margin m0 >= 0) and this one (m1 < 0), where the margin is 0 at the
        fraction m0 / (m0 - m1) of the step. Without a previous sample (the very
        first one already touches), the contact is put at this sample."""
        position = state[0:3]
        if self._previous is None:
            fractions = np.where(touching, 1.0, np.inf)
            t0, p0 = t, position
        else:
            t0, p0, m0, _ = self._previous
            with np.errstate(divide="ignore", invalid="ignore"):
                fractions = np.where(
                    np.isfinite(swept),
                    swept,
                    np.where(touching, m0 / (m0 - margins), np.inf),
                )
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
