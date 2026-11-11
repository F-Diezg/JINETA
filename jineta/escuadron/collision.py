"""Collisions of many drones at once: `campo.collision.CollisionMonitor` on a batch.

The same checks as the reference monitor, for n drones flying the same course:
every sample (the spheres against walls, ceiling, obstacles and gate frames; the
landing gear against the floor once a drone has left it) and the swept check of
the spheres' straight moves between samples (conservative advancement, a contact
within `SWEEP_TOLERANCE`). Each drone keeps its own memory: whether it has left
the floor, its previous sample, and its collision once it has one.

Shapes are asked about all the drones' spheres in one call (an (n * spheres, 3)
array of points). The advancement runs only for the drones whose margins allow a
contact in between; each round advances every drone still going, and a drone
drops out when it touches or reaches its next sample, as in the reference's loop.

A collision is kept as arrays: `collided` (n,), `check` (n,), the index of the
check in the margins' order [room, obstacles..., gates..., floor] (-1 for none),
`time` (n,) and `position` (n, 3). `collision(i)` gives drone i's as the
reference's `Collision`.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.campo.collision import SWEEP_TOLERANCE, Collision, Obstacle
from jineta.campo.indoor import Room
from jineta.escuadron import quaternion

# Which drones: None for all, a boolean mask (n,) or an array of indices.
Drones = npt.ArrayLike | None


def _mask(drones: Drones, n: int) -> np.ndarray:
    """A boolean mask (n,) of the chosen drones."""
    mask = np.zeros(n, dtype=bool)
    mask[slice(None) if drones is None else np.asarray(drones)] = True
    return mask


def _rows(value: npt.ArrayLike, what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{what} must have shape (k, 3), k >= 1")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite")
    array.flags.writeable = False
    return array


@dataclass
class BatchCollisionMonitor:
    """Watches n drones for collisions; callable once per step with the time and
    the states (n, 13), and the drones to check (all by default).

    The arguments are the reference's (`campo.collision.CollisionMonitor`), plus
    the number of drones n.
    """

    room: Room
    obstacles: Sequence[Obstacle]
    sphere_centers: npt.ArrayLike
    sphere_radii: npt.ArrayLike
    floor_points: npt.ArrayLike
    n: int
    gates: Sequence[Obstacle] = ()
    collided: np.ndarray = field(init=False)
    check: np.ndarray = field(init=False)
    time: np.ndarray = field(init=False)
    position: np.ndarray = field(init=False)
    airborne: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
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
        n, checks = self.n, 2 + len(self.obstacles) + len(self.gates)
        self.collided = np.zeros(n, dtype=bool)
        self.check = np.full(n, -1)
        self.time = np.full(n, np.nan)
        self.position = np.full((n, 3), np.nan)
        self.airborne = np.zeros(n, dtype=bool)
        self._has_previous = np.zeros(n, dtype=bool)
        self._previous_time = np.zeros(n)
        self._previous_position = np.zeros((n, 3))
        self._previous_margins = np.zeros((n, checks))
        self._previous_centers = np.zeros((n, len(self.sphere_centers), 3))

    @property
    def kinds(self) -> list[str]:
        """The kind of each check, in the margins' order."""
        return [
            "room",
            *["obstacle"] * len(self.obstacles),
            *["gate"] * len(self.gates),
            "floor",
        ]

    def collision(self, i: int) -> Collision | None:
        """Drone i's collision, as the reference reports it, or None."""
        if not self.collided[i]:
            return None
        check = int(self.check[i])
        kind = self.kinds[check]
        n_obstacles = len(self.obstacles)
        index = {"obstacle": check - 1, "gate": check - 1 - n_obstacles}.get(kind)
        position = self.position[i].copy()
        position.flags.writeable = False
        return Collision(kind, index, float(self.time[i]), position)

    def reset(self, drones: Drones = None) -> None:
        """Forget these drones' flights (all by default): no collision, not
        airborne, no previous sample."""
        rows = _mask(drones, self.n)
        self.collided[rows] = False
        self.check[rows] = -1
        self.time[rows] = np.nan
        self.position[rows] = np.nan
        self.airborne[rows] = False
        self._has_previous[rows] = False

    def _distances(self) -> list:
        """The distance function of every check but the floor, in margin order."""
        return [
            self.room.wall_clearance,
            *(o.signed_distance for o in (*self.obstacles, *self.gates)),
        ]

    def _placed(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sphere centres (k, spheres, 3) and floor points (k, points, 3) in NED:
        p = x + R r for every drone and point."""
        rotation = quaternion.to_matrix(states[:, 6:10])
        position = states[:, None, 0:3]
        centers = position + np.einsum("kij,sj->ksi", rotation, self.sphere_centers)
        feet = position + np.einsum("kij,pj->kpi", rotation, self.floor_points)
        return centers, feet

    def _sphere_gaps(self, distance, centers: np.ndarray) -> np.ndarray:
        """Gap of every sphere (k, spheres) to one shape: distance - radius."""
        k, spheres = centers.shape[0:2]
        flat = np.asarray(distance(centers.reshape(-1, 3)))
        return flat.reshape(k, spheres) - self.sphere_radii

    def margins(self, states: np.ndarray) -> np.ndarray:
        """Margins (k, checks) [m] of the states (k, 13): [room, obstacles...,
        gates..., floor], negative = touching."""
        centers, feet = self._placed(states)
        return self._margins(centers, feet)

    def _margins(self, centers: np.ndarray, feet: np.ndarray) -> np.ndarray:
        spheres = [
            np.min(self._sphere_gaps(distance, centers), axis=1)
            for distance in self._distances()
        ]
        floor = np.min(-feet[:, :, 2], axis=1)  # height above the floor z = 0
        return np.column_stack([*spheres, floor])

    def __call__(
        self, t: float | npt.ArrayLike, states: np.ndarray, drones: Drones = None
    ) -> np.ndarray:
        """Check one sample of these drones (all by default), and the moves of
        their spheres since their previous sample; returns `collided` (n,). Drones
        already collided, or not chosen, are left as they are."""
        times = np.broadcast_to(np.asarray(t, dtype=float), (self.n,))
        rows = np.flatnonzero(_mask(drones, self.n) & ~self.collided)
        if len(rows) == 0:
            return self.collided.copy()
        t_now = times[rows]
        positions = states[rows, 0:3]
        centers, feet = self._placed(states[rows])
        margins = self._margins(centers, feet)
        touching = margins < 0.0
        touching[:, -1] &= self.airborne[rows]
        swept = self._swept_fractions(rows, centers, margins, touching)
        hit = touching.any(axis=1) | np.isfinite(swept).any(axis=1)

        previous = self._has_previous[rows]
        with np.errstate(divide="ignore", invalid="ignore"):
            m0 = self._previous_margins[rows]
            interpolated = np.where(touching, m0 / (m0 - margins), np.inf)
        fractions = np.where(
            previous[:, None],
            np.where(np.isfinite(swept), swept, interpolated),
            np.where(touching, 1.0, np.inf),
        )
        t0 = np.where(previous, self._previous_time[rows], t_now)
        p0 = np.where(previous[:, None], self._previous_position[rows], positions)
        first = np.argmin(fractions, axis=1)
        f = fractions[np.arange(len(rows)), first]

        new = rows[hit]
        f, t0, p0 = f[hit], t0[hit], p0[hit]
        self.collided[new] = True
        self.check[new] = first[hit]
        self.time[new] = t0 + f * (t_now[hit] - t0)
        self.position[new] = p0 + f[:, None] * (positions[hit] - p0)

        free = ~hit
        going = rows[free]
        self.airborne[going] |= margins[free, -1] > 0.0
        self._has_previous[going] = True
        self._previous_time[going] = t_now[free]
        self._previous_position[going] = positions[free]
        self._previous_margins[going] = margins[free]
        self._previous_centers[going] = centers[free]
        return self.collided.copy()

    def _swept_fractions(
        self,
        rows: np.ndarray,
        centers: np.ndarray,
        margins: np.ndarray,
        touching: np.ndarray,
    ) -> np.ndarray:
        """For every drone in `rows` and every check but the floor: the fraction of
        the step at which a sphere first touches it along the straight moves since
        the previous sample, or inf (as the reference's `_swept_fractions`)."""
        fractions = np.full(margins.shape, np.inf)
        previous = self._has_previous[rows]
        start = self._previous_centers[rows]
        move = centers - start
        lengths = np.sqrt(np.sum(move * move, axis=2))
        longest = np.max(lengths, axis=1)
        previous_margins = self._previous_margins[rows]
        with np.errstate(divide="ignore"):
            inverse = np.where(lengths > 0.0, 1.0 / lengths, np.inf)
        for i, distance in enumerate(self._distances()):
            near = previous & (previous_margins[:, i] + margins[:, i] <= longest)
            if not np.any(near):
                continue
            k = np.flatnonzero(near)
            s, gap = self._advance(distance, start[k], move[k], inverse[k])
            refine = touching[k, i]
            with np.errstate(divide="ignore", invalid="ignore"):
                refined = s + (1.0 - s) * gap / (gap - margins[k, i])
            fractions[k, i] = np.where(refine, refined, s)
        return fractions

    def _advance(
        self,
        distance,
        start: np.ndarray,
        move: np.ndarray,
        inverse: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conservative advancement for k drones at once: for each, the first
        fraction s in [0, 1] at which a sphere comes within SWEEP_TOLERANCE of the
        shape and that gap, or (inf, nan). Each round advances the drones still
        going."""
        k = len(start)
        s = np.zeros(k)
        found_s = np.full(k, np.inf)
        found_gap = np.full(k, np.nan)
        going = np.arange(k)
        while len(going) > 0:
            points = start[going] + s[going, None, None] * move[going]
            gaps = self._sphere_gaps(distance, points)
            gap = np.min(gaps, axis=1)
            touched = gap <= SWEEP_TOLERANCE
            found_s[going[touched]] = s[going[touched]]
            found_gap[going[touched]] = gap[touched]
            going_on = going[~touched]
            s[going_on] = s[going_on] + np.min(
                gaps[~touched] * inverse[going_on], axis=1
            )
            going = going_on[s[going_on] <= 1.0]
        return found_s, found_gap
