"""Courses: a room, its obstacles, the gates in order, the start and a time limit,
read from a YAML file.

A course file is strict: an unknown key, a missing key, a value of the wrong type
or size, a gate or obstacle outside the room, or a start inside an obstacle is an
error that says where it is (for example `gates[1].width`). A typo never passes
silently. Positions are NED like the rest of the simulator: z points down, so a
gate 1.5 m above the floor has z = -1.5. Angles are given in degrees (`yaw_deg`,
a heading: clockwise from North) and stored in radians.

The format (every key shown; `obstacles` and an obstacle's `yaw_deg` may be left
out):

    name: First course
    time_limit: 30                  # s
    room:
      size: [18.0, 10.0, 4.0]       # length North, width East, height [m]
    start:
      position: [1.5, 5.0]          # North, East, standing on the floor [m]
      yaw_deg: 0
    gates:                          # flown in this order
      - shape: rectangle
        center: [5.0, 5.0, -1.5]    # NED [m]
        yaw_deg: 0                  # the direction to fly through it
        width: 1.6
        height: 1.4
        thickness: 0.1
      - shape: circle
        center: [9.0, 3.0, -2.0]
        yaw_deg: 0
        radius: 0.8
        thickness: 0.1
    obstacles:
      - shape: box
        center: [12.0, 2.0, -1.0]
        size: [2.0, 1.0, 2.0]
        yaw_deg: 20
      - shape: cylinder
        base: [11.5, 6.6, 0.0]      # centre of its bottom face
        radius: 0.3
        height: 4.0
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml

from jineta.campo.collision import CollisionMonitor
from jineta.campo.gates import CircularGate, Gate, RectangularGate
from jineta.campo.indoor import Box, Cylinder, Room
from jineta.campo.referee import Referee
from jineta.core import quaternion

Obstacle = Box | Cylinder


class CourseError(ValueError):
    """A course that cannot be flown as written; the message says where."""


@dataclass(frozen=True)
class Course:
    """A course: the room, its obstacles, the gates in order, the start (North,
    East on the floor [m], and a heading [rad]) and the time limit of a lap [s]."""

    name: str
    room: Room
    start: np.ndarray
    start_yaw: float
    gates: tuple[Gate, ...]
    obstacles: tuple[Obstacle, ...]
    time_limit: float

    def start_state(self, rest_height: float) -> np.ndarray:
        """The 13-element state standing on the floor at the start: centre of mass
        `rest_height` [m] above the floor (see jinete.contact.resting_height),
        level, facing the start heading, still."""
        y = np.zeros(13)
        y[0:2] = self.start
        y[2] = -rest_height
        y[6:10] = quaternion.from_euler(0.0, 0.0, self.start_yaw)
        return y

    def monitor(
        self,
        sphere_centers: npt.ArrayLike,
        sphere_radii: npt.ArrayLike,
        floor_points: npt.ArrayLike,
    ) -> CollisionMonitor:
        """A collision monitor for this course and a vehicle's hitbox and gear."""
        return CollisionMonitor(
            self.room,
            self.obstacles,
            sphere_centers,
            sphere_radii,
            floor_points,
            gates=self.gates,
        )

    def referee(
        self,
        sphere_centers: npt.ArrayLike,
        sphere_radii: npt.ArrayLike,
        floor_points: npt.ArrayLike,
    ) -> Referee:
        """A referee for a lap of this course (with its own collision monitor)."""
        monitor = self.monitor(sphere_centers, sphere_radii, floor_points)
        return Referee(self.gates, monitor, self.time_limit)


def load_course(path: str | Path) -> Course:
    """Read a course file (YAML); a CourseError says what is wrong and where."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CourseError(f"{path}: not valid YAML: {error}") from None
    try:
        return course_from_dict(data)
    except CourseError as error:
        raise CourseError(f"{path}: {error}") from None


def course_from_dict(data: Any) -> Course:
    """Build a course from the contents of a course file (a dict)."""
    fields = _keys(
        data, "course", ["name", "time_limit", "room", "start", "gates"], ["obstacles"]
    )
    name = fields["name"]
    if not isinstance(name, str) or not name.strip():
        raise CourseError("name: must be a non-empty text")
    time_limit = _positive(fields["time_limit"], "time_limit")

    room_fields = _keys(fields["room"], "room", ["size"])
    room = _build(lambda: Room(_vector(room_fields["size"], 3, "room.size")), "room")

    start_fields = _keys(fields["start"], "start", ["position", "yaw_deg"])
    start = _vector(start_fields["position"], 2, "start.position")
    start_yaw = np.radians(_number(start_fields["yaw_deg"], "start.yaw_deg"))

    gates_data = fields["gates"]
    if not isinstance(gates_data, list) or not gates_data:
        raise CourseError("gates: must be a list of at least one gate")
    gates = tuple(_gate(g, f"gates[{i}]") for i, g in enumerate(gates_data))

    obstacles_data = fields.get("obstacles", [])
    if obstacles_data is None:
        obstacles_data = []
    if not isinstance(obstacles_data, list):
        raise CourseError("obstacles: must be a list")
    obstacles = tuple(
        _obstacle(o, f"obstacles[{i}]") for i, o in enumerate(obstacles_data)
    )

    course = Course(
        name.strip(), room, start, float(start_yaw), gates, obstacles, time_limit
    )
    _check_layout(course)
    return course


# ---- the parts of a file ------------------------------------------------------------


def _gate(data: Any, where: str) -> Gate:
    shape = _shape(data, where, ["rectangle", "circle"])
    if shape == "rectangle":
        f = _keys(
            data, where, ["shape", "center", "yaw_deg", "width", "height", "thickness"]
        )
        return _build(
            lambda: RectangularGate(
                _vector(f["center"], 3, f"{where}.center"),
                np.radians(_number(f["yaw_deg"], f"{where}.yaw_deg")),
                _positive(f["width"], f"{where}.width"),
                _positive(f["height"], f"{where}.height"),
                _positive(f["thickness"], f"{where}.thickness"),
            ),
            where,
        )
    f = _keys(data, where, ["shape", "center", "yaw_deg", "radius", "thickness"])
    return _build(
        lambda: CircularGate(
            _vector(f["center"], 3, f"{where}.center"),
            np.radians(_number(f["yaw_deg"], f"{where}.yaw_deg")),
            _positive(f["radius"], f"{where}.radius"),
            _positive(f["thickness"], f"{where}.thickness"),
        ),
        where,
    )


def _obstacle(data: Any, where: str) -> Obstacle:
    shape = _shape(data, where, ["box", "cylinder"])
    if shape == "box":
        f = _keys(data, where, ["shape", "center", "size"], ["yaw_deg"])
        yaw = _number(f.get("yaw_deg", 0.0), f"{where}.yaw_deg")
        return _build(
            lambda: Box(
                _vector(f["center"], 3, f"{where}.center"),
                _vector(f["size"], 3, f"{where}.size"),
                np.radians(yaw),
            ),
            where,
        )
    f = _keys(data, where, ["shape", "base", "radius", "height"])
    return _build(
        lambda: Cylinder(
            _vector(f["base"], 3, f"{where}.base"),
            _positive(f["radius"], f"{where}.radius"),
            _positive(f["height"], f"{where}.height"),
        ),
        where,
    )


def _check_layout(course: Course) -> None:
    """Everything inside the room, and the start clear of every obstacle.

    A gate's whole frame must fit: its centre at least as far from every wall, the
    floor and the ceiling as the frame's farthest point from its centre (a simple,
    slightly cautious rule that does not depend on the gate's heading).
    """
    room = course.room
    hint = " (NED: z points down, so a point 1.5 m above the floor has z = -1.5)"
    for i, gate in enumerate(course.gates):
        if isinstance(gate, RectangularGate):
            reach = np.hypot(
                gate.width / 2 + gate.thickness, gate.height / 2 + gate.thickness
            )
        else:
            reach = gate.radius + gate.thickness
        if room.clearance(gate.center) < reach:
            raise CourseError(f"gates[{i}]: the frame does not fit in the room{hint}")
    for i, obstacle in enumerate(course.obstacles):
        point = obstacle.center if isinstance(obstacle, Box) else obstacle.base
        if not room.contains(point):
            raise CourseError(f"obstacles[{i}]: outside the room{hint}")
    foot = np.array([course.start[0], course.start[1], -0.01])
    if not room.contains(foot):
        raise CourseError("start.position: outside the room")
    for i, obstacle in enumerate(course.obstacles):
        if obstacle.signed_distance(foot) <= 0.0:
            raise CourseError(f"start.position: inside obstacles[{i}]")


# ---- strict reading -----------------------------------------------------------------


def _keys(
    data: Any, where: str, required: Sequence[str], optional: Sequence[str] = ()
) -> dict:
    if not isinstance(data, dict):
        raise CourseError(f"{where}: must be a mapping of keys to values")
    unknown = [k for k in data if k not in required and k not in optional]
    if unknown:
        allowed = ", ".join([*required, *optional])
        raise CourseError(f"{where}: unknown key '{unknown[0]}' (expected: {allowed})")
    missing = [k for k in required if k not in data]
    if missing:
        raise CourseError(f"{where}: missing key '{missing[0]}'")
    return data


def _shape(data: Any, where: str, shapes: Sequence[str]) -> str:
    if not isinstance(data, dict) or "shape" not in data:
        raise CourseError(f"{where}: missing key 'shape' (one of {', '.join(shapes)})")
    shape = data["shape"]
    if shape not in shapes:
        raise CourseError(f"{where}.shape: '{shape}' is not one of {', '.join(shapes)}")
    return shape


def _number(value: Any, where: str) -> float:
    # bool is a kind of int in Python; "true" is not a number in a course file.
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CourseError(f"{where}: must be a number, not {value!r}")
    if not np.isfinite(value):
        raise CourseError(f"{where}: must be finite")
    return float(value)


def _positive(value: Any, where: str) -> float:
    number = _number(value, where)
    if not number > 0.0:
        raise CourseError(f"{where}: must be > 0")
    return number


def _vector(value: Any, size: int, where: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != size:
        raise CourseError(f"{where}: must be a list of {size} numbers")
    return np.array([_number(v, f"{where}[{i}]") for i, v in enumerate(value)])


def _build(make: Any, where: str) -> Any:
    """Call a constructor; its own ValueError becomes a CourseError at `where`."""
    try:
        return make()
    except CourseError:
        raise
    except ValueError as error:
        raise CourseError(f"{where}: {error}") from None
