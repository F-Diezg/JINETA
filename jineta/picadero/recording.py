"""Recorded flights: whole laps written to a file, for the viewer (phase 13).

A recording is one JSON file: the course (or courses) the laps were flown on, the
shape of the drone (from its profile, to draw it), and the laps, each with its lap
clock, positions and attitudes at every environment step (50 Hz), how it ended,
when it crossed each gate, where it touched something and how close it came to
anything. It is what a viewer needs and nothing more: the agent that flew the laps
is kept as its own file (`Agent.save`), which is what flies them again exactly.

The numbers are rounded for the eye (positions to 0.1 mm, attitudes to 1e-5,
times to 0.1 ms): a recording is for looking at, not for checking the physics.
Non-finite numbers (a lap time that does not exist) are written as null.

A file is written whole or not at all (`write_recording`): it is written under a
temporary name and then renamed, and a rename inside one folder is atomic, so a
viewer reading the folder while a training writes to it never sees half a file.
`recording_files` lists only finished files.

The file format (version 1):

    {"format": "jineta-recording", "version": 1,
     "steps": training steps of the agent, "info": {...},
     "shape": {"rotors": [[x, y, z], ...], "spins": [1, -1, ...],
               "propeller_radius": r, "hitbox": {"centers": [...], "radii": [...]}},
     "courses": [course file contents (campo.course.course_to_dict), ...],
     "laps": [{"course": index, "label": text, "status": "finished" | "crashed" |
               "timeout", "lap_time": s | null, "gates": crossed,
               "pass_times": [s | null per gate], "clearance": m | null,
               "crash_time": s | null, "crash_point": [N, E, D] | null,
               "crash_kind": "room" | "obstacle" | "gate" | "floor" | "",
               "times": [s, ...], "positions": [[N, E, D], ...],
               "attitudes": [[w, x, y, z], ...]}, ...]}
"""

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jineta.campo.course import Course, course_from_dict, course_to_dict
from jineta.escuadron.referee import STATUS_NAMES
from jineta.jinete.profiles import MultirotorProfile

FORMAT = "jineta-recording"
VERSION = 1
SUFFIX = ".json"
TEMPORARY = ".tmp"  # a file being written; never read

# Decimals kept in a file (see the module's docstring).
POSITION_DECIMALS = 4
ATTITUDE_DECIMALS = 5
TIME_DECIMALS = 4


def _read_only(value: npt.ArrayLike, dtype: Any = float) -> np.ndarray:
    array = np.array(value, dtype=dtype)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class DroneShape:
    """What a viewer needs to draw a multirotor: its rotors (body axes, FRD [m])
    and their spins, the propeller radius [m] and the hitbox spheres."""

    rotors: np.ndarray
    spins: np.ndarray
    propeller_radius: float
    hitbox_centers: np.ndarray
    hitbox_radii: np.ndarray

    def __post_init__(self) -> None:
        rotors = _read_only(self.rotors)
        if rotors.ndim != 2 or rotors.shape[1] != 3 or len(rotors) < 1:
            raise ValueError("rotors must be (r, 3) with r >= 1")
        spins = _read_only(self.spins, int)
        if spins.shape != (len(rotors),) or not np.all(np.abs(spins) == 1):
            raise ValueError("spins must be +1 or -1, one per rotor")
        centers = _read_only(self.hitbox_centers)
        radii = _read_only(self.hitbox_radii)
        if centers.ndim != 2 or centers.shape[1] != 3 or radii.shape != (len(centers),):
            raise ValueError("the hitbox needs (k, 3) centres and k radii")
        if not self.propeller_radius >= 0.0:
            raise ValueError("propeller_radius must be >= 0")
        object.__setattr__(self, "rotors", rotors)
        object.__setattr__(self, "spins", spins)
        object.__setattr__(self, "hitbox_centers", centers)
        object.__setattr__(self, "hitbox_radii", radii)


def drone_shape(profile: MultirotorProfile) -> DroneShape:
    """The shape of a profile's drone."""
    params, _, _, hitbox = profile
    return DroneShape(
        rotors=np.array([rotor.position for rotor in params.rotors]),
        spins=np.array([rotor.spin for rotor in params.rotors]),
        propeller_radius=float(params.propeller_radius),
        hitbox_centers=hitbox.centers,
        hitbox_radii=hitbox.radii,
    )


@dataclass(frozen=True)
class Lap:
    """One recorded lap of one drone.

    course: the index of its course in the recording.
    label: what flew it ("mean action", "noisy", "drawn drone 3", ...).
    status: "finished", "crashed" or "timeout".
    lap_time: [s], nan unless finished.
    gates: gates crossed; pass_times (gates,) [s], nan where not crossed.
    clearance: the closest approach to anything [m] (nan if not known).
    crash_time, crash_point, crash_kind: where it touched something (nan and ""
        if it did not).
    times (k,), positions (k, 3), attitudes (k, 4): the lap clock [s], the centre
        of mass (NED [m]) and the attitude (quaternion BODY -> NED) at the start
        and after every environment step, until the lap ended.
    """

    course: int
    label: str
    status: str
    lap_time: float
    gates: int
    pass_times: np.ndarray
    clearance: float
    crash_time: float
    crash_point: np.ndarray
    crash_kind: str
    times: np.ndarray = field(repr=False)
    positions: np.ndarray = field(repr=False)
    attitudes: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.status not in ("finished", "crashed", "timeout"):
            raise ValueError(f"unknown lap status {self.status!r}")
        times = _read_only(self.times)
        positions = _read_only(self.positions)
        attitudes = _read_only(self.attitudes)
        k = len(times)
        if k < 1 or times.shape != (k,):
            raise ValueError("a lap needs at least one sample")
        if positions.shape != (k, 3) or attitudes.shape != (k, 4):
            raise ValueError("positions (k, 3) and attitudes (k, 4) must match times")
        if np.any(np.diff(times) < 0.0):
            raise ValueError("times must not go back")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "attitudes", attitudes)
        object.__setattr__(self, "pass_times", _read_only(self.pass_times))
        object.__setattr__(self, "crash_point", _read_only(self.crash_point))

    @property
    def duration(self) -> float:
        """The lap clock at its last sample [s]."""
        return float(self.times[-1])


@dataclass(frozen=True)
class Recording:
    """Laps flown by one agent: the training steps it had, the courses, the
    drone's shape, the laps and free information (what it was trained in, how
    the laps were flown)."""

    steps: int
    courses: tuple[Course, ...]
    shape: DroneShape
    laps: tuple[Lap, ...]
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be >= 0")
        if not self.courses:
            raise ValueError("a recording needs at least one course")
        for lap in self.laps:
            if not 0 <= lap.course < len(self.courses):
                raise ValueError(f"lap {lap.label!r}: no course {lap.course}")
            if lap.pass_times.shape != (len(self.courses[lap.course].gates),):
                raise ValueError(f"lap {lap.label!r}: one pass time per gate")
        object.__setattr__(self, "courses", tuple(self.courses))
        object.__setattr__(self, "laps", tuple(self.laps))


def laps_from_flight(
    result: dict[str, Any],
    drones: Sequence[int],
    labels: Sequence[str],
    course: int = 0,
) -> list[Lap]:
    """Laps of chosen drones from a lap flown by `training.fly(..., record=True)`
    (positions and attitudes are the first 3 and the 4 after the velocity of the
    state: the rigid body's layout)."""
    laps = []
    for i, label in zip(drones, labels, strict=True):
        flight = result["flights"][i]
        states = flight["states"]
        laps.append(
            Lap(
                course=course,
                label=label,
                status=STATUS_NAMES[int(result["status"][i])],
                lap_time=float(result["lap_time"][i]),
                gates=int(result["gates"][i]),
                pass_times=result["pass_times"][i],
                clearance=float(result["clearance"][i]),
                crash_time=float(result["crash_time"][i]),
                crash_point=result["crash_point"][i],
                crash_kind=str(result["crash_kind"][i]),
                times=flight["times"],
                positions=states[:, 0:3],
                attitudes=states[:, 6:10],
            )
        )
    return laps


# ---- the file -------------------------------------------------------------------


def _number(value: float, decimals: int | None = None) -> float | None:
    """A finite number (rounded), or None for nan and infinities."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, decimals) if decimals is not None else value


def _numbers(values: npt.ArrayLike, decimals: int | None = None) -> list:
    """Nested lists of numbers (rounded), None where not finite."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return _number(array, decimals)  # type: ignore[return-value]
    return [_numbers(row, decimals) for row in array]


def _float(value: Any) -> float:
    """A number from a file: None is nan."""
    return float("nan") if value is None else float(value)


def _floats(values: Any) -> np.ndarray:
    """Numbers from a file (nested lists): None is nan."""
    return np.array(_nan_for_none(values), dtype=float)


def _nan_for_none(values: Any) -> Any:
    if isinstance(values, list):
        return [_nan_for_none(v) for v in values]
    return np.nan if values is None else values


def recording_to_dict(recording: Recording) -> dict[str, Any]:
    """The contents of a recording file (JSON-ready)."""
    shape = recording.shape
    return {
        "format": FORMAT,
        "version": VERSION,
        "steps": int(recording.steps),
        "info": recording.info,
        "shape": {
            "rotors": _numbers(shape.rotors),
            "spins": [int(s) for s in shape.spins],
            "propeller_radius": float(shape.propeller_radius),
            "hitbox": {
                "centers": _numbers(shape.hitbox_centers),
                "radii": _numbers(shape.hitbox_radii),
            },
        },
        "courses": [course_to_dict(course) for course in recording.courses],
        "laps": [
            {
                "course": lap.course,
                "label": lap.label,
                "status": lap.status,
                "lap_time": _number(lap.lap_time, TIME_DECIMALS),
                "gates": lap.gates,
                "pass_times": _numbers(lap.pass_times, TIME_DECIMALS),
                "clearance": _number(lap.clearance, POSITION_DECIMALS),
                "crash_time": _number(lap.crash_time, TIME_DECIMALS),
                "crash_point": (
                    _numbers(lap.crash_point, POSITION_DECIMALS)
                    if np.all(np.isfinite(lap.crash_point))
                    else None
                ),
                "crash_kind": lap.crash_kind,
                "times": _numbers(lap.times, TIME_DECIMALS),
                "positions": _numbers(lap.positions, POSITION_DECIMALS),
                "attitudes": _numbers(lap.attitudes, ATTITUDE_DECIMALS),
            }
            for lap in recording.laps
        ],
    }


def recording_from_dict(data: Any) -> Recording:
    """A recording from the contents of its file; a ValueError says what is
    wrong."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f'not a recording (no "format": "{FORMAT}")')
    if data.get("version") != VERSION:
        raise ValueError(
            f"recording version {data.get('version')!r}; this code reads {VERSION}"
        )
    try:
        shape_data = data["shape"]
        shape = DroneShape(
            rotors=shape_data["rotors"],
            spins=shape_data["spins"],
            propeller_radius=float(shape_data["propeller_radius"]),
            hitbox_centers=shape_data["hitbox"]["centers"],
            hitbox_radii=shape_data["hitbox"]["radii"],
        )
        courses = tuple(course_from_dict(c) for c in data["courses"])
        laps = tuple(
            Lap(
                course=int(lap["course"]),
                label=str(lap["label"]),
                status=lap["status"],
                lap_time=_float(lap["lap_time"]),
                gates=int(lap["gates"]),
                pass_times=_floats(lap["pass_times"]),
                clearance=_float(lap["clearance"]),
                crash_time=_float(lap["crash_time"]),
                crash_point=(
                    _floats(lap["crash_point"])
                    if lap["crash_point"] is not None
                    else np.full(3, np.nan)
                ),
                crash_kind=str(lap["crash_kind"]),
                times=_floats(lap["times"]),
                positions=_floats(lap["positions"]),
                attitudes=_floats(lap["attitudes"]),
            )
            for lap in data["laps"]
        )
        return Recording(
            steps=int(data["steps"]),
            courses=courses,
            shape=shape,
            laps=laps,
            info=dict(data.get("info", {})),
        )
    except KeyError as error:
        raise ValueError(f"recording: missing key {error}") from None


def write_recording(folder: str | Path, name: str, recording: Recording) -> Path:
    """Write a recording as `folder/name.json`, whole or not at all (written
    under a temporary name, then renamed). Returns its path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}{SUFFIX}"
    temporary = folder / f"{name}{SUFFIX}{TEMPORARY}"
    text = json.dumps(recording_to_dict(recording), separators=(",", ":"))
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def read_recording(path: str | Path) -> Recording:
    """Read a recording file; a ValueError names the file and what is wrong."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return recording_from_dict(data)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{path}: {error}") from None


def recording_files(folder: str | Path) -> list[Path]:
    """The finished recordings in a folder, sorted by name (none if the folder
    does not exist yet)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix == SUFFIX)
