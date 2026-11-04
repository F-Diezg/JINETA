"""The shape a vehicle collides with: spheres fixed to its body.

A sphere of radius r touches a shape exactly when the signed distance of its
centre to the shape is less than r (see campo/indoor.py), so spheres are the
cheapest shape to check against any wall or obstacle, and a set of them can follow
a vehicle's outline as closely as needed. A vehicle touches something when any of
its spheres does.

Building a hitbox from a vehicle's collision geometry:
  - a rotor is a thin cylinder (its propeller disc): one sphere at its hub holds
    it exactly (`rotor_spheres`);
  - a box is cut into a grid of equal cells, and each cell is held by the sphere
    through its corners (`box_spheres`): the cells fill the box, so the spheres
    cover it. More cells, smaller spheres, closer to the box;
  - one sphere around everything (`bounding_sphere`) is the simplest and the most
    cautious.

The floor is not judged with the spheres: a vehicle resting on its landing gear has
its rotors closer to the floor than their reach, so its spheres go through the
floor while it stands still. Its landing gear points (jinete/contact.py) say when it
touches the floor.

Like the rotors and the landing gear, a hitbox is profile data: centres in body axes
(FRD) relative to the centre of mass, and radii.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Hitbox:
    """Spheres fixed to a vehicle: centres in body axes (FRD, relative to the centre
    of mass) [m], one row per sphere, and their radii [m]."""

    centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=float)
        radii = np.array(self.radii, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            raise ValueError("centers must have shape (n, 3), n >= 1")
        if radii.shape != (len(centers),):
            raise ValueError("radii must have one value per centre")
        if not np.all(np.isfinite(centers)):
            raise ValueError("centers must be finite")
        if not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise ValueError("every radius must be finite and > 0")
        centers.flags.writeable = False
        radii.flags.writeable = False
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)

    def extent(self, direction: npt.ArrayLike) -> float:
        """How far the hitbox reaches from the centre of mass along a body direction
        [m]: the largest c . u + r over its spheres, with u the unit direction. For
        example extent([0, 0, -1]) is how far it reaches up when level."""
        u = np.asarray(direction, dtype=float)
        u = u / np.linalg.norm(u)
        return float(np.max(self.centers @ u + self.radii))

    def contains(self, points: npt.ArrayLike) -> np.ndarray:
        """True for each body point (n, 3) inside at least one sphere (or on it)."""
        p = np.atleast_2d(np.asarray(points, dtype=float))
        gaps = np.linalg.norm(p[:, None, :] - self.centers[None, :, :], axis=2)
        return np.any(gaps <= self.radii + 1e-12, axis=1)


def union(*hitboxes: Hitbox) -> Hitbox:
    """All the spheres of several hitboxes as one."""
    return Hitbox(
        np.vstack([h.centers for h in hitboxes]),
        np.concatenate([h.radii for h in hitboxes]),
    )


def disc_reach(center: npt.ArrayLike, radius: float, thickness: float = 0.0) -> float:
    """Farthest distance [m] from the centre of mass to a rotor disc.

    The disc is a cylinder of `radius` and `thickness` centred at `center` (body
    axes), with its axis along body z. A point of its outer edge is
    c + radius u + h e_z, with u a unit vector in the body x-y plane and
    |h| <= thickness / 2. The horizontal part |c_xy + radius u| is largest when u
    points along c_xy (it is then |c_xy| + radius), and the vertical part |c_z + h|
    when h has the sign of c_z: together sqrt((|c_xy| + radius)^2 +
    (|c_z| + thickness / 2)^2). (|c| + radius, from the triangle inequality, is a
    little more than that whenever c_z is not zero.)
    """
    c = np.asarray(center, dtype=float)
    return float(np.hypot(np.hypot(c[0], c[1]) + radius, abs(c[2]) + thickness / 2.0))


def box_corners(size: Sequence[float]) -> np.ndarray:
    """The eight corners of a box centred on the centre of mass and aligned with the
    body axes: (+-length/2, +-width/2, +-height/2)."""
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    )
    return signs * half


def bounding_sphere(
    rotor_hubs: npt.ArrayLike,
    propeller_radius: float,
    body_points: npt.ArrayLike,
    propeller_thickness: float = 0.0,
) -> Hitbox:
    """One sphere at the centre of mass that holds every rotor disc (hub positions in
    body axes, all with the same propeller radius and thickness) and every body
    point."""
    hubs = np.atleast_2d(np.asarray(rotor_hubs, dtype=float))
    points = np.atleast_2d(np.asarray(body_points, dtype=float))
    reach = max(
        max(disc_reach(hub, propeller_radius, propeller_thickness) for hub in hubs),
        float(np.max(np.linalg.norm(points, axis=1))),
    )
    return Hitbox(centers=[[0.0, 0.0, 0.0]], radii=[reach])


def rotor_spheres(
    rotor_hubs: npt.ArrayLike, propeller_radius: float, propeller_thickness: float
) -> Hitbox:
    """One sphere per rotor, at its hub: the farthest points of the disc from its
    centre are on its rim, sqrt(radius^2 + (thickness / 2)^2) away."""
    hubs = np.atleast_2d(np.asarray(rotor_hubs, dtype=float))
    radius = float(np.hypot(propeller_radius, propeller_thickness / 2.0))
    return Hitbox(hubs, np.full(len(hubs), radius))


def box_spheres(size: Sequence[float], divisions: Sequence[int]) -> Hitbox:
    """Spheres that cover a box centred on the centre of mass, aligned with the body
    axes: the box is cut into divisions = (nx, ny, nz) equal cells, and each cell
    is held by the sphere at its centre that passes through its corners (radius
    half the cell's diagonal). Every point of the box is in a cell, so in a sphere.
    """
    size = np.asarray(size, dtype=float)
    n = np.asarray(divisions, dtype=int)
    if size.shape != (3,) or not np.all(size > 0.0):
        raise ValueError("size must be three lengths > 0")
    if n.shape != (3,) or not np.all(n >= 1):
        raise ValueError("divisions must be three integers >= 1")
    cell = size / n
    axes = [(np.arange(k) - (k - 1) / 2.0) * c for k, c in zip(n, cell, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return Hitbox(grid, np.full(len(grid), float(np.linalg.norm(cell)) / 2.0))
