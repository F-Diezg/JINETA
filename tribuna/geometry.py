"""The course and the drone as meshes for the viewer, in NED axes [m].

A mesh is points (p, 3) and its cells in the flat layout VTK (and so PyVista)
uses: for each cell, the number of its points and then their indices, one after
the other. A square and a triangle sharing an edge are
`faces = [4, 0, 1, 2, 3,  3, 2, 1, 4]`; polylines (`lines`) are written the
same way. With that layout a mesh goes to PyVista as it is
(`pyvista.PolyData(points, faces=..., lines=...)`), and everything here is
plain NumPy that the tests check without PyVista.

The viewer draws in NED as the simulator computes (x North, y East, z down):
the camera is told that "up" is -z. Nothing is converted, so a point drawn is
a point of the simulation.

Shapes follow `jineta.campo` exactly: a box turned by its yaw, a column rising
from its base towards -z, a rectangular gate as four square bars around its
opening (outer rectangle = opening + thickness all round, depth = thickness), a
round gate as a torus whose tube (thickness across) is centred on a circle of
radius = opening radius + thickness / 2. The tests check that every point of a
mesh is on the surface the collision model uses (signed distance zero).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.campo.course import Course
from jineta.campo.gates import CircularGate, Gate, RectangularGate
from jineta.campo.indoor import Box, Cylinder, Room
from jineta.core import quaternion
from jineta.picadero.recording import DroneShape


def _cells(cells: Sequence[Sequence[int]]) -> np.ndarray:
    """Cells (lists of point indices) in VTK's flat layout."""
    flat: list[int] = []
    for cell in cells:
        flat.append(len(cell))
        flat.extend(int(i) for i in cell)
    return np.array(flat, dtype=np.int64)


def cell_lists(flat: npt.ArrayLike) -> list[list[int]]:
    """VTK's flat layout back to lists of point indices (for tests and checks)."""
    flat = np.asarray(flat, dtype=np.int64)
    cells, i = [], 0
    while i < len(flat):
        size = int(flat[i])
        cells.append([int(j) for j in flat[i + 1 : i + 1 + size]])
        i += 1 + size
    return cells


@dataclass(frozen=True)
class Mesh:
    """Points (p, 3) [m] with faces and polylines in VTK's flat layout."""

    points: np.ndarray
    faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    lines: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (p, 3)")
        object.__setattr__(self, "points", points)
        for name in ("faces", "lines"):
            flat = np.array(getattr(self, name), dtype=np.int64)
            for cell in cell_lists(flat):
                if any(i < 0 or i >= len(points) for i in cell):
                    raise ValueError(f"{name}: a cell uses a point that is not there")
            object.__setattr__(self, name, flat)


def merge(meshes: Sequence[Mesh]) -> Mesh:
    """One mesh with every point and cell of several (their indices moved)."""
    points, faces, lines = [], [], []
    offset = 0
    for mesh in meshes:
        points.append(mesh.points)
        faces.extend([[i + offset for i in c] for c in cell_lists(mesh.faces)])
        lines.extend([[i + offset for i in c] for c in cell_lists(mesh.lines)])
        offset += len(mesh.points)
    if not points:
        return Mesh(np.zeros((0, 3)))
    return Mesh(np.vstack(points), _cells(faces), _cells(lines))


# ---- solids --------------------------------------------------------------------

# A unit cube's corners (signs of the half sizes) and its six faces, each
# counter-clockwise seen from outside (right-handed axes), so their normals
# point out.
_CORNERS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)
_CUBE_FACES = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [0, 4, 7, 3],
]


def _yaw_matrix(yaw: float) -> np.ndarray:
    """A heading turn about the vertical: local x to NED (as campo's boxes)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def block(center: npt.ArrayLike, size: npt.ArrayLike, yaw: float = 0.0) -> Mesh:
    """A box: centre (NED) [m], size [length, width, height] [m] along its own
    axes, turned by a yaw [rad] (a heading, as `campo.indoor.Box`)."""
    half = np.asarray(size, dtype=float) / 2.0
    local = _CORNERS * half
    points = local @ _yaw_matrix(yaw).T + np.asarray(center, dtype=float)
    return Mesh(points, _cells(_CUBE_FACES))


def column(base: npt.ArrayLike, radius: float, height: float, sides: int = 32) -> Mesh:
    """A vertical column rising from `base` (centre of its bottom face, NED)
    towards -z: `sides` flat sides, the bottom and top faces."""
    angles = 2.0 * np.pi * np.arange(sides) / sides
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    base = np.asarray(base, dtype=float)
    bottom = np.column_stack([ring, np.zeros(sides)]) + base
    top = np.column_stack([ring, np.full(sides, -height)]) + base
    points = np.vstack([bottom, top])
    # Every face counter-clockwise seen from outside (z points down: the
    # bottom face, at the larger z, faces +z).
    around = [
        [i, sides + i, sides + (i + 1) % sides, (i + 1) % sides] for i in range(sides)
    ]
    faces = [*around, list(range(sides)), list(range(2 * sides - 1, sides - 1, -1))]
    return Mesh(points, _cells(faces))


def _gate_to_ned(gate: Gate, local: np.ndarray) -> np.ndarray:
    """Points in a gate's axes (n along its heading, u to its right, v down) to
    NED."""
    c, s = np.cos(gate.yaw), np.sin(gate.yaw)
    n, u, v = local[:, 0], local[:, 1], local[:, 2]
    return np.stack([c * n - s * u, s * n + c * u, v], axis=1) + gate.center


def rectangular_frame(gate: RectangularGate) -> Mesh:
    """The four square bars of a rectangular gate: top and bottom across the
    whole outer width, the sides between them."""
    t = gate.thickness
    w, h = gate.width / 2.0, gate.height / 2.0
    bars = [  # (u from, u to, v from, v to) in the gate's plane
        (-w - t, w + t, -h - t, -h),  # top (v down: -h is up)
        (-w - t, w + t, h, h + t),  # bottom
        (-w - t, -w, -h, h),  # left
        (w, w + t, -h, h),  # right
    ]
    meshes = []
    for u0, u1, v0, v1 in bars:
        local = np.array(
            [
                [(t / 2.0) * sn, (u0 + u1) / 2.0 + su * (u1 - u0) / 2.0,
                 (v0 + v1) / 2.0 + sv * (v1 - v0) / 2.0]
                for sn, su, sv in _CORNERS
            ]
        )  # fmt: skip
        # The corners in (n, u, v) have the cube's handedness, as (x, y, z).
        meshes.append(Mesh(_gate_to_ned(gate, local), _cells(_CUBE_FACES)))
    return merge(meshes)


def circular_frame(gate: CircularGate, around: int = 48, across: int = 12) -> Mesh:
    """The torus of a round gate: its tube's centre line a circle of radius
    opening + thickness / 2 in the gate's plane, the tube thickness across;
    `around` steps along the circle, `across` around the tube."""
    big = gate.radius + gate.thickness / 2.0
    small = gate.thickness / 2.0
    phi = 2.0 * np.pi * np.arange(around) / around  # along the circle
    psi = 2.0 * np.pi * np.arange(across) / across  # around the tube
    p, s = np.meshgrid(phi, psi, indexing="ij")
    radial = big + small * np.cos(s)
    local = np.stack(
        [small * np.sin(s), radial * np.cos(p), radial * np.sin(p)], axis=-1
    ).reshape(-1, 3)
    faces = []
    for i in range(around):
        for j in range(across):
            a = i * across + j
            b = ((i + 1) % around) * across + j
            c = ((i + 1) % around) * across + (j + 1) % across
            d = i * across + (j + 1) % across
            faces.append([a, b, c, d])
    return Mesh(_gate_to_ned(gate, local), _cells(faces))


def gate_frame(gate: Gate) -> Mesh:
    """A gate's frame, rectangular or round."""
    if isinstance(gate, RectangularGate):
        return rectangular_frame(gate)
    return circular_frame(gate)


def obstacle(shape: Box | Cylinder) -> Mesh:
    """A box or a column of a course."""
    if isinstance(shape, Box):
        return block(shape.center, shape.size, shape.yaw)
    return column(shape.base, shape.radius, shape.height)


# ---- the room --------------------------------------------------------------------


def room_edges(room: Room) -> Mesh:
    """The twelve edges of the room (floor corner at the origin, z up to
    -height), as lines."""
    length, width, height = room.size
    points = (_CORNERS + 1.0) / 2.0 * np.array([length, width, -height])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]  # fmt: skip
    return Mesh(points, lines=_cells(edges))


def floor_grid(room: Room, spacing: float = 1.0) -> Mesh:
    """Lines on the floor every `spacing` metres in both directions (and its
    edges), so distances can be read."""
    if not spacing > 0.0:
        raise ValueError("spacing must be > 0")
    length, width = room.size[0], room.size[1]
    norths = np.append(np.arange(0.0, length, spacing), length)
    easts = np.append(np.arange(0.0, width, spacing), width)
    points, lines = [], []
    for x in norths:
        lines.append([len(points), len(points) + 1])
        points += [[x, 0.0, 0.0], [x, width, 0.0]]
    for y in easts:
        lines.append([len(points), len(points) + 1])
        points += [[0.0, y, 0.0], [length, y, 0.0]]
    return Mesh(np.array(points), lines=_cells(lines))


@dataclass(frozen=True)
class CourseMeshes:
    """A course for the viewer: the room's edges and floor grid, each gate's
    frame (in order) and each obstacle, and the start."""

    edges: Mesh
    grid: Mesh
    gates: tuple[Mesh, ...]
    obstacles: tuple[Mesh, ...]
    start: np.ndarray


def course_meshes(course: Course, spacing: float = 1.0) -> CourseMeshes:
    """Every part of a course as a mesh."""
    return CourseMeshes(
        edges=room_edges(course.room),
        grid=floor_grid(course.room, spacing),
        gates=tuple(gate_frame(g) for g in course.gates),
        obstacles=tuple(obstacle(o) for o in course.obstacles),
        start=np.array([course.start[0], course.start[1], 0.0]),
    )


# ---- the drone -------------------------------------------------------------------


@dataclass(frozen=True)
class DroneMeshes:
    """A drone in its body axes (FRD [m]): the body and arms, and the rotor
    discs turning one way (spin +1, counter-clockwise from above) and the
    other; the viewer places them with `pose_matrix`."""

    frame: Mesh
    discs_ccw: Mesh
    discs_cw: Mesh


def disc(center: npt.ArrayLike, radius: float, sides: int = 24) -> Mesh:
    """A flat disc in the body's x-y plane (a rotor), as one polygon."""
    angles = 2.0 * np.pi * np.arange(sides) / sides
    points = np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(sides)]
    ) + np.asarray(center, dtype=float)
    return Mesh(points, _cells([list(range(sides))]))


def drone_meshes(shape: DroneShape, arm: float = 0.02) -> DroneMeshes:
    """The drone of a recording: a flat body (a box reaching a third of the way
    to the rotors), an arm (a thin box `arm` wide) from the centre to each
    rotor, and a disc of the propeller radius at each rotor."""
    rotors = shape.rotors
    reach = float(np.max(np.hypot(rotors[:, 0], rotors[:, 1])))
    body = block([0.0, 0.0, 0.0], [0.66 * reach, 0.66 * reach, 0.05])
    arms = []
    for rotor in rotors:
        length = float(np.hypot(rotor[0], rotor[1]))
        heading = float(np.arctan2(rotor[1], rotor[0]))
        middle = [rotor[0] / 2.0, rotor[1] / 2.0, rotor[2] / 2.0]
        arms.append(block(middle, [length, arm, arm], heading))
    radius = shape.propeller_radius if shape.propeller_radius > 0.0 else 0.4 * reach
    ccw = [disc(r, radius) for r, s in zip(rotors, shape.spins, strict=True) if s > 0]
    cw = [disc(r, radius) for r, s in zip(rotors, shape.spins, strict=True) if s < 0]
    return DroneMeshes(merge([body, *arms]), merge(ccw), merge(cw))


def pose_matrix(position: npt.ArrayLike, attitude: npt.ArrayLike) -> np.ndarray:
    """The 4 x 4 matrix that places a body-axes mesh in NED: [[R, p], [0, 1]],
    R the rotation BODY -> NED of the attitude quaternion (normalised first: a
    recording rounds it), p the position. A point b of the body goes to
    R b + p (what VTK's `user_matrix` does with it)."""
    matrix = np.eye(4)
    matrix[0:3, 0:3] = quaternion.to_matrix(quaternion.normalize(np.asarray(attitude)))
    matrix[0:3, 3] = np.asarray(position, dtype=float)
    return matrix


def path_lines(points: npt.ArrayLike) -> Mesh:
    """A path through points (k, 3), as one polyline (empty for fewer than two
    points)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) < 2:
        return Mesh(points)
    return Mesh(points, lines=_cells([list(range(len(points)))]))
