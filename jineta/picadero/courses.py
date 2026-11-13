"""Random courses, and training on many courses at once.

A policy that has flown one course for millions of laps knows that course, not
how to fly. To learn to fly any course it trains on many, each new, and is
measured on courses it never saw.

`random_course(rng)` draws one: a room, gates one after the other (each a few
metres from the last, the direction of the legs turning a bounded angle, the
gate turned a little from the leg it ends, rectangles and circles of random
sizes, at random heights), and obstacles beside the path, close enough to matter
and far enough not to block it. A drawing that does not fit (a gate too close to
a wall or to another gate, the path through the gates passing near a frame, an
obstacle too close to the path) is thrown away and drawn again, so every course
returned can be flown: the path the classical baseline follows (`GatePath`)
keeps `path_clearance` from every obstacle and the frames it does not go through.
The course is built through `course_from_dict`, so it passes the checks of a
course file.

`CourseSetEnv` flies k courses at once, n drones on each (a `CourseEnv` per
course: the batch engine flies one course per batch), with the shape of
`CourseEnv` for n * k drones; every course is replaced by a newly drawn one every
`renew_every` steps (the courses in turn, not all at once), and the episodes of
its drones still flying are then truncated (they could have gone on).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from jineta.campo.course import Course, course_from_dict
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.escuadron.referee import FLYING
from jineta.picadero.baseline import GatePath
from jineta.picadero.env import CourseEnv
from jineta.picadero.spaces import BoxSpace

TRIES = 200  # drawings of one part before the whole course is drawn again


@dataclass(frozen=True)
class CourseRanges:
    """What a random course may be. Pairs are (low, high) [m, deg, count].

    room: length North, width East, height.
    gates: how many.
    leg: distance between consecutive gate centres (from the start for the
        first).
    turn_deg: largest change of direction from one leg to the next (the first
        leg within half of it from the start's heading, North).
    gate_turn_deg: largest angle between a gate's heading and its leg.
    height: a gate centre's height above the floor.
    rectangle, circle: width and height (rectangle), radius (circle) ranges.
    wall_margin: smallest distance from a gate centre to a wall or the ceiling.
    gate_spacing: smallest distance between two gate centres (and the start).
    obstacles: how many.
    obstacle_size: a box's sides and a column's diameter.
    path_clearance: smallest distance from an obstacle, or a frame the path does
        not go through, to the path through the gates.
    approach: entry and exit points of the path (as the baseline's).
    time_limit: of a lap [s].
    """

    room: tuple[tuple[float, float], ...] = ((16.0, 22.0), (9.0, 12.0), (3.5, 4.5))
    gates: tuple[int, int] = (3, 6)
    leg: tuple[float, float] = (3.5, 6.0)
    turn_deg: float = 70.0
    gate_turn_deg: float = 25.0
    height: tuple[float, float] = (1.0, 2.5)
    rectangle: tuple[tuple[float, float], tuple[float, float]] = (
        (1.3, 1.8),
        (1.2, 1.6),
    )
    circle: tuple[float, float] = (0.65, 0.9)
    wall_margin: float = 1.5
    gate_spacing: float = 3.0
    obstacles: tuple[int, int] = (1, 4)
    obstacle_size: tuple[float, float] = (0.3, 1.5)
    path_clearance: float = 1.0
    approach: float = 1.5
    time_limit: float = 30.0


def _gates(rng: np.random.Generator, r: CourseRanges, room, start) -> list | None:
    """Gate entries of the course dictionary, or None if they do not fit."""
    length, width, height = room
    count = int(rng.integers(r.gates[0], r.gates[1] + 1))
    centers = [np.array([start[0], start[1], 0.0])]
    heading = 0.0
    gates = []
    for i in range(count):
        gate: dict[str, Any] = {"thickness": 0.1}
        if rng.uniform() < 0.5:
            gate.update(
                shape="rectangle",
                width=float(rng.uniform(*r.rectangle[0])),
                height=float(rng.uniform(*r.rectangle[1])),
            )
            reach = np.hypot(gate["width"], gate["height"]) / 2.0 + 0.1
        else:
            gate.update(shape="circle", radius=float(rng.uniform(*r.circle)))
            reach = gate["radius"] + 0.1
        # The course file's check: the whole frame inside the room, judged by the
        # frame's reach from its centre (whatever its heading).
        low, high = reach + 0.05, height - max(reach, 0.6) - 0.05
        for _ in range(TRIES):
            limit = r.turn_deg / (2.0 if i == 0 else 1.0)
            leg_heading = heading + np.radians(rng.uniform(-limit, limit))
            distance = rng.uniform(*r.leg)
            up = float(np.clip(rng.uniform(*r.height), low, high))
            center = centers[-1] + distance * np.array(
                [np.cos(leg_heading), np.sin(leg_heading), 0.0]
            )
            center[2] = -up
            inside = (
                r.wall_margin <= center[0] <= length - r.wall_margin
                and r.wall_margin <= center[1] <= width - r.wall_margin
            )
            spaced = all(
                np.linalg.norm(center[:2] - c[:2]) >= r.gate_spacing for c in centers
            )
            if inside and spaced and low <= high:
                break
        else:
            return None
        heading = leg_heading
        yaw = leg_heading + np.radians(rng.uniform(-r.gate_turn_deg, r.gate_turn_deg))
        gate["center"] = [float(v) for v in center]
        gate["yaw_deg"] = float(np.degrees(yaw))
        gates.append(gate)
        centers.append(center)
    return gates


def _path_samples(course: Course, approach: float) -> np.ndarray:
    """Points every 0.1 m along the baseline's path, up to the last gate."""
    path = GatePath.through(course, approach, 0.5)
    end = path.distance[-2]  # the last gate's centre (its exit is not flown)
    return path.point_at(np.arange(0.0, end, 0.1))


def _frames_clear(course: Course, samples: np.ndarray, clearance: float) -> bool:
    """The path keeps `clearance` from every frame, except near the centre of the
    gate it goes through (within its opening's reach)."""
    for gate in course.gates:
        distance = gate.signed_distance(samples)
        near = np.linalg.norm(samples - gate.center, axis=1) < 1.5
        if np.any((distance < clearance) & ~near):
            return False
        if np.any(distance[near] < 0.3):
            return False
    return True


def _obstacle(
    rng: np.random.Generator, r: CourseRanges, room, samples: np.ndarray
) -> dict[str, Any]:
    """An obstacle entry beside a random point of the path."""
    point = samples[rng.integers(len(samples))]
    side = rng.uniform(0.0, 2.0 * np.pi)
    size = rng.uniform(*r.obstacle_size)
    offset = r.path_clearance + size / 2.0 + rng.uniform(0.0, 1.5)
    x = point[0] + offset * np.cos(side)
    y = point[1] + offset * np.sin(side)
    height = room[2]
    if rng.uniform() < 0.5:
        tall = height if rng.uniform() < 0.5 else rng.uniform(0.8, height)
        return {
            "shape": "cylinder",
            "base": [float(x), float(y), 0.0],
            "radius": float(size / 2.0),
            "height": float(tall),
        }
    sides = rng.uniform(*r.obstacle_size, 2)
    tall = height if rng.uniform() < 0.3 else rng.uniform(0.6, height - 0.5)
    return {
        "shape": "box",
        "center": [float(x), float(y), float(-tall / 2.0)],
        "size": [float(sides[0]), float(sides[1]), float(tall)],
        "yaw_deg": float(rng.uniform(0.0, 180.0)),
    }


def random_course(
    rng: np.random.Generator,
    ranges: CourseRanges | None = None,
    name: str = "Random course",
) -> Course:
    """A random course that can be flown (see the module)."""
    r = ranges or CourseRanges()
    while True:
        room = [float(rng.uniform(*span)) for span in r.room]
        start = [
            float(rng.uniform(1.2, 2.5)),
            float(rng.uniform(r.wall_margin, room[1] - r.wall_margin)),
        ]
        gates = _gates(rng, r, room, start)
        if gates is None:
            continue
        data: dict[str, Any] = {
            "name": name,
            "time_limit": r.time_limit,
            "room": {"size": room},
            "start": {"position": start, "yaw_deg": 0.0},
            "gates": gates,
            "obstacles": [],
        }
        try:
            course = course_from_dict(data)
        except ValueError:
            continue  # a frame that does not fit in the room
        samples = _path_samples(course, r.approach)
        if not _frames_clear(course, samples, r.path_clearance):
            continue
        if np.any(course.room.clearance(samples[5:]) < 0.6):
            continue  # the path gets too close to a wall or the ceiling
        wanted = int(rng.integers(r.obstacles[0], r.obstacles[1] + 1))
        start_point = np.array([start[0], start[1], -0.5])
        for _ in range(TRIES):
            if len(data["obstacles"]) == wanted:
                break
            entry = _obstacle(rng, r, room, samples)
            try:
                trial = course_from_dict(
                    {**data, "obstacles": [*data["obstacles"], entry]}
                )
            except ValueError:
                continue  # outside the room
            shape = trial.obstacles[-1]
            if np.min(shape.signed_distance(samples)) < r.path_clearance:
                continue
            if shape.signed_distance(start_point) < 1.0:
                continue
            if any(
                shape.signed_distance(g.center) < 1.6 for g in trial.gates
            ):  # away from every frame
                continue
            data["obstacles"].append(entry)
        return course_from_dict(data)


def gate_count(course: Course) -> tuple[int, int]:
    """How many rectangular and circular gates a course has."""
    rectangles = sum(isinstance(g, RectangularGate) for g in course.gates)
    circles = sum(isinstance(g, CircularGate) for g in course.gates)
    return rectangles, circles


class CourseSetEnv:
    """k courses flown at once, n drones each, renewed in turn (see the module).

    make_course: draws a course from a generator (e.g. `random_course`).
    k, n: courses at once and drones per course.
    renew_every: environment steps between two renewals of the same course.
    every other keyword goes to `CourseEnv` (each course's environment).
    """

    def __init__(
        self,
        make_course: Callable[[np.random.Generator], Course],
        k: int,
        n: int,
        renew_every: int = 1000,
        **env_kwargs: Any,
    ) -> None:
        if k < 1 or n < 1 or renew_every < 1:
            raise ValueError("k, n and renew_every must be >= 1")
        self.make_course, self.k, self.per_course = make_course, k, n
        self.n = k * n
        self.renew_every = renew_every
        self.env_kwargs = env_kwargs
        self.rng = np.random.default_rng()
        self.envs: list[CourseEnv] = []
        self.renewals = 0

    def _new_env(self) -> CourseEnv:
        return CourseEnv(
            self.make_course(self.rng), n=self.per_course, **self.env_kwargs
        )

    @property
    def observation_space(self) -> BoxSpace:
        return self._any_env().observation_space

    @property
    def action_space(self) -> BoxSpace:
        return self._any_env().action_space

    def _any_env(self) -> CourseEnv:
        if not self.envs:
            self.reset()
        return self.envs[0]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """New courses for every batch; returns the observations (n * k, size)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.envs = [self._new_env() for _ in range(self.k)]
        self.clock = 0
        return np.concatenate([env.reset()[0] for env in self.envs]), {}

    def close(self) -> None:
        """Nothing to release (the interface of `ParallelEnv`)."""

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """One step of every course's drones; returns what CourseEnv returns, for
        the n * k drones (course by course)."""
        action = np.asarray(action, dtype=float).reshape(self.k, self.per_course, -1)
        self.clock += 1
        parts = []
        for i, env in enumerate(self.envs):
            observations, rewards, terminated, truncated, info = env.step(action[i])
            due = (self.clock + i * self.renew_every // self.k) % self.renew_every
            if due == 0:
                info = dict(info)
                going_on = ~(terminated | truncated)
                truncated = truncated | going_on
                info["final_observation"] = np.where(
                    going_on[:, None], observations, info["final_observation"]
                )
                info["status"] = np.where(going_on, FLYING, info["status"])
                self.envs[i] = self._new_env()
                observations, _ = self.envs[i].reset()
                self.renewals += 1
            parts.append((observations, rewards, terminated, truncated, info))
        info = {key: np.concatenate([p[4][key] for p in parts]) for key in parts[0][4]}
        return (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
            np.concatenate([p[3] for p in parts]),
            info,
        )
