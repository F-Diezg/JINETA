"""What the viewer shows at each moment: the playback of recorded laps.

The player holds the recordings in order of training steps and plays them one
after the other, the way a film of the training would: every lap of the current
recording (on the course being shown) flies at once, on its own lap clock, and
once the last of them has ended and a short pause has passed, the next
recording starts. When there is no next one yet, the last recording is played
again until a new one arrives (a training still running adds them: `add`), so
the window always shows the latest flying. The keys of the viewer call its
methods: pause, speed, the next or previous recording, the first or the
newest, the next course.

Times: `t` is the lap clock of the recording being played [s]; `advance(dt)`
moves it by dt (the wall time since the last frame) times `speed`.

It knows nothing of PyVista: `frame()` says where every drone is, which path it
has flown and how it ended, and the scene draws that.

Seeing a policy learn (phase 13): the laps of the recordings before the current
one stay drawn as trails (`trails`), fading with their age, coloured by how they
ended, so the change from crashing to finishing shows. A trail is the first lap
of its recording on the course (the mean action's, the policy as it would fly):
trajectories are never averaged (the average of a path into the pillar and one
past it is a path nobody flew). What can be averaged (how many laps finished,
their lap time) goes in the charts beside (`tribuna.run`).
"""

from dataclasses import dataclass, field

import numpy as np

from jineta.campo.course import Course
from jineta.picadero.recording import Lap, Recording
from tribuna.motion import path_until, pose_at

SPEEDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # playback speeds offered
PAUSE = 1.0  # [s] of wall time between the end of a recording and the next
TRAILS = 10  # earlier recordings drawn as trails
NEWEST_OPACITY = 0.8  # the trail of the recording just before the current one
OLDEST_OPACITY = 0.1  # the trail TRAILS recordings back


def course_key(course: Course) -> tuple:
    """Enough of a course to tell two apart: its name, room and gates (the
    courses of different recordings are compared with it)."""
    return (
        course.name,
        tuple(float(x) for x in course.room.size),
        tuple(tuple(float(x) for x in g.center) for g in course.gates),
        len(course.obstacles),
    )


def trail_opacity(age: int, keep: int = TRAILS) -> float:
    """How visible the trail of the recording `age` places back is: from
    NEWEST_OPACITY (the one just before) down to OLDEST_OPACITY (keep back),
    in a straight line; 0 beyond."""
    if age < 1 or age > keep:
        return 0.0
    if keep == 1:
        return NEWEST_OPACITY
    fraction = (age - 1) / (keep - 1)
    return NEWEST_OPACITY + fraction * (OLDEST_OPACITY - NEWEST_OPACITY)


@dataclass
class Trail:
    """The first lap of an earlier recording on the course shown: its whole
    path, how it ended (and where it touched something), how visible."""

    steps: int
    status: str
    points: np.ndarray
    crash_point: np.ndarray | None
    opacity: float


@dataclass
class DroneFrame:
    """One recorded drone at the player's moment.

    position, attitude: where it is (NED [m], quaternion BODY -> NED).
    path: the path it has flown so far (k, 3).
    ended: whether its lap has ended by now; status: how it will end (or
        ended); lap_time: its lap time (nan unless it finished);
        crash_point: where it touched something, once it has (else None).
    """

    label: str
    status: str
    lap_time: float
    position: np.ndarray
    attitude: np.ndarray
    path: np.ndarray
    ended: bool
    crash_point: np.ndarray | None


@dataclass
class Frame:
    """Everything the scene draws at one moment: the recording, its course,
    the time, and every drone of the recording on that course."""

    index: int
    recording: Recording
    course: int
    t: float
    duration: float
    drones: list[DroneFrame]


@dataclass
class Player:
    """Plays recordings in order (see the module)."""

    recordings: list[Recording] = field(default_factory=list)
    index: int = 0
    course: int = 0
    t: float = 0.0
    speed: float = 1.0
    paused: bool = False
    pause: float = PAUSE
    _waited: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self.recordings = sorted(self.recordings, key=lambda r: r.steps)

    @property
    def current(self) -> Recording | None:
        return self.recordings[self.index] if self.recordings else None

    def laps(self) -> list[Lap]:
        """The current recording's laps on the course being shown."""
        recording = self.current
        if recording is None:
            return []
        return [lap for lap in recording.laps if lap.course == self.course]

    def duration(self) -> float:
        """How long the current recording plays: its longest lap on the course
        [s] (0 when there is none)."""
        laps = self.laps()
        return max((lap.duration for lap in laps), default=0.0)

    def add(self, recordings: list[Recording]) -> None:
        """New recordings (a training still running); kept in order of steps,
        the one being played stays the one being played."""
        if not recordings:
            return
        current = self.current
        self.recordings = sorted([*self.recordings, *recordings], key=lambda r: r.steps)
        if current is not None:
            self.index = next(i for i, r in enumerate(self.recordings) if r is current)

    def advance(self, dt: float) -> None:
        """Move the playback on by dt seconds of wall time."""
        if self.paused or self.current is None:
            return
        end = self.duration()
        if self.t < end:
            self.t = min(end, self.t + self.speed * dt)
            return
        self._waited += dt  # the pause after the end, in wall time
        if self._waited >= self.pause:
            if self.index + 1 < len(self.recordings):
                self.go_to(self.index + 1)
            else:
                self.restart()

    # ---- what the keys do ----------------------------------------------------------

    def go_to(self, index: int) -> None:
        """Play recording `index` (clipped to those there are) from its start;
        its course is kept if it has one of that number."""
        if not self.recordings:
            return
        self.index = int(np.clip(index, 0, len(self.recordings) - 1))
        if self.course >= len(self.recordings[self.index].courses):
            self.course = 0
        self.restart()

    def restart(self) -> None:
        self.t = 0.0
        self._waited = 0.0

    def next(self) -> None:
        self.go_to(self.index + 1)

    def previous(self) -> None:
        self.go_to(self.index - 1)

    def first(self) -> None:
        self.go_to(0)

    def newest(self) -> None:
        self.go_to(len(self.recordings) - 1)

    def next_course(self) -> None:
        """Show the laps on the recording's next course (random courses have
        one lap on each), from their start."""
        if self.current is not None:
            self.course = (self.course + 1) % len(self.current.courses)
            self.restart()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def faster(self) -> None:
        self.speed = _step_speed(self.speed, +1)

    def slower(self) -> None:
        self.speed = _step_speed(self.speed, -1)

    def seek(self, t: float) -> None:
        """Jump to lap clock t (clipped to the recording)."""
        self.t = float(np.clip(t, 0.0, self.duration()))
        self._waited = 0.0

    # ---- what the scene draws ------------------------------------------------------

    def trails(self, keep: int = TRAILS) -> list[Trail]:
        """The earlier recordings' laps on the course shown (the first lap of
        each, the most recent first), up to `keep` recordings back."""
        recording = self.current
        if recording is None:
            return []
        key = course_key(recording.courses[self.course])
        trails = []
        for age in range(1, keep + 1):
            j = self.index - age
            if j < 0:
                break
            earlier = self.recordings[j]
            matching = [
                i for i, c in enumerate(earlier.courses) if course_key(c) == key
            ]
            laps = [
                lap for lap in earlier.laps if matching and lap.course == matching[0]
            ]
            if not laps:
                continue
            lap = laps[0]
            crashed = lap.status == "crashed" and np.all(np.isfinite(lap.crash_point))
            trails.append(
                Trail(
                    steps=earlier.steps,
                    status=lap.status,
                    points=lap.positions,
                    crash_point=lap.crash_point.copy() if crashed else None,
                    opacity=trail_opacity(age, keep),
                )
            )
        return trails

    def frame(self) -> Frame | None:
        """Where everything is now (None before the first recording)."""
        recording = self.current
        if recording is None:
            return None
        drones = []
        for lap in self.laps():
            position, attitude = pose_at(lap, self.t)
            crashed = lap.status == "crashed" and np.all(np.isfinite(lap.crash_point))
            touched = crashed and self.t >= lap.crash_time
            drones.append(
                DroneFrame(
                    label=lap.label,
                    status=lap.status,
                    lap_time=lap.lap_time,
                    position=position,
                    attitude=attitude,
                    path=path_until(lap, self.t),
                    ended=self.t >= lap.duration,
                    crash_point=lap.crash_point.copy() if touched else None,
                )
            )
        return Frame(
            self.index, recording, self.course, self.t, self.duration(), drones
        )


def _step_speed(speed: float, step: int) -> float:
    """The next playback speed up or down the list (staying at its ends)."""
    i = int(np.argmin([abs(np.log(speed / s)) for s in SPEEDS]))
    return SPEEDS[int(np.clip(i + step, 0, len(SPEEDS) - 1))]


def describe(frame: Frame, speed: float, paused: bool, total: int) -> str:
    """The text in the corner of the window: which recording, the time, and
    how its laps ended."""
    recording = frame.recording
    lines = [
        f"recording {frame.index + 1} of {total}: "
        f"{recording.steps:,} steps"
        + (
            f" (iteration {recording.info['iteration']})"
            if "iteration" in recording.info
            else ""
        ),
        f"course {frame.course + 1} of {len(recording.courses)}: "
        f"{recording.courses[frame.course].name}",
        f"t = {frame.t:5.2f} s of {frame.duration:.2f} s, x{speed:g}"
        + ("  (paused)" if paused else ""),
    ]
    for drone in frame.drones:
        if not drone.ended:
            result = "..."
        elif np.isfinite(drone.lap_time):
            result = f"{drone.lap_time:.2f} s"
        else:
            result = drone.status
        lines.append(f"  {drone.label}: {result}")
    return "\n".join(lines)
