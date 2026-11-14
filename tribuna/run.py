"""A training's folder read as it grows: its recordings and its log.

A training writes `log.csv` (a row per update), `laps/<steps>.json` (the laps
of every evaluation, `jineta.picadero.recording`) and `agents/<steps>.npz`. A
two-stage training (`examples.train_course`, `examples.train_realism`) writes
two such folders side by side (`fly/`, `margin/`); a replay
(`examples.record_agent`) writes only `laps/` and `agents/`. `Run` takes any of
them, or a folder of recordings itself:

    run = Run("runs/realism")      # both stages
    new = run.poll()               # the recordings written since the last poll
    log = run.log()                # every stage's log, on one axis
    x = run.positions()            # where each recording is on that axis

`poll` can be called as often as wanted (the viewer calls it every second while
a training goes on): it reads only files it has not read, and a file is never
half-written (they are renamed into place whole). A file that cannot be read
is reported once in `errors` and skipped, so a viewer never stops on it.

Two clocks of a training. A recording's `steps` are the steps its agent was
trained for, those it brought from an earlier stage included; a stage's log
counts its own steps. A second stage starts from the first stage's *best*
agent, which may be older than its last, so in the agents' steps the stages
overlap (measured: a first stage of 3 M steps whose best agent had 2.6 M).
The viewer's axis is therefore the training's own clock: the steps run, stage
after stage (a stage's steps plus every step of the stages before it). The
stages are in the order they were run (the first recording of a later stage
has more steps than the first of an earlier one), and the recordings in the
order of that axis: stage after stage, then steps. A recording's place on it
is its steps minus those its agent brought, plus the stages before (the steps
brought are found from a recording and the log row of its iteration; a stage
without them is taken to have brought none).
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from jineta.picadero.recording import Recording, read_recording, recording_files

LAPS = "laps"
LOG = "log.csv"


def _stage_folders(folder: Path) -> list[Path]:
    """The folders of a run that hold recordings or a log: the folder itself,
    or its subfolders (stages), sorted by name."""
    if (folder / LAPS).is_dir() or (folder / LOG).is_file():
        return [folder]
    if recording_files(folder):  # a folder of recordings itself
        return [folder]
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_dir() and ((p / LAPS).is_dir() or (p / LOG).is_file())
    )


def _laps_folder(stage: Path) -> Path:
    return stage / LAPS if (stage / LAPS).is_dir() else stage


@dataclass
class LogSeries:
    """A training's log as arrays, one entry per update, on the run's axis of
    steps (see the module): the share of the episodes that ended in that update
    which finished, crashed or ran out of time, their mean lap time [s] (nan
    when none finished) and mean return."""

    steps: np.ndarray
    finished: np.ndarray
    crashed: np.ndarray
    timeout: np.ndarray
    mean_lap_time: np.ndarray
    mean_return: np.ndarray

    def __len__(self) -> int:
        return len(self.steps)


_COLUMNS = ("iteration", "steps", "episodes", "finished", "crashed", "timeout")


def read_log(path: str | Path) -> list[dict[str, float]]:
    """The complete rows of a training log (a row still being written, or a
    file not there yet, gives nothing), as numbers (nan where empty)."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="", encoding="ascii") as file:
        for row in csv.DictReader(file):
            if any(not row.get(key) for key in _COLUMNS):
                continue  # the last row, cut while it is written
            try:
                numbers = {
                    key: float(row[key]) if row.get(key) not in (None, "") else np.nan
                    for key in (*_COLUMNS, "mean_lap_time", "mean_return")
                }
            except ValueError:
                continue
            rows.append(numbers)
    return rows


@dataclass
class Run:
    """The recordings and the log of a training (or a replay) folder."""

    folder: Path
    recordings: list[Recording] = field(default_factory=list)
    errors: dict[Path, str] = field(default_factory=dict)
    _read: set[Path] = field(default_factory=set, repr=False)
    _stage_of: list[Path] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)

    def stages(self) -> list[Path]:
        """The stages in the order they were run: by the steps of their first
        recording (those without recordings last, by name)."""

        def first(stage: Path) -> tuple[float, str]:
            steps = [
                r.steps
                for r, s in zip(self.recordings, self._stage_of, strict=True)
                if s == stage
            ]
            return (min(steps) if steps else np.inf, stage.name)

        return sorted(_stage_folders(self.folder), key=first)

    def poll(self) -> list[Recording]:
        """Read the recordings written since the last call; returns them in the
        run's order (stage after stage, then steps), and keeps every recording,
        in that order, in `recordings`."""
        new: list[tuple[Recording, Path]] = []
        for stage in _stage_folders(self.folder):
            for path in recording_files(_laps_folder(stage)):
                if path in self._read:
                    continue
                self._read.add(path)
                try:
                    new.append((read_recording(path), stage))
                except (ValueError, OSError) as error:
                    self.errors[path] = str(error)
        pairs = [*zip(self.recordings, self._stage_of, strict=True), *new]
        self.recordings = [r for r, _ in pairs]
        self._stage_of = [s for _, s in pairs]
        rank = {stage: i for i, stage in enumerate(self.stages())}
        order = sorted(
            range(len(pairs)), key=lambda i: (rank[pairs[i][1]], pairs[i][0].steps)
        )
        self.recordings = [pairs[i][0] for i in order]
        self._stage_of = [pairs[i][1] for i in order]
        new_ids = {id(r) for r, _ in new}
        return [r for r in self.recordings if id(r) in new_ids]

    def stage(self, index: int) -> Path:
        """The folder of the stage that wrote recording `index`."""
        return self._stage_of[index]

    def _axis(self) -> dict[Path, tuple[float, float, list[dict[str, float]]]]:
        """For each stage: the steps its agent brought, the steps run by the
        stages before it, and its log's rows."""
        axis = {}
        before = 0.0
        for stage in self.stages():
            rows = read_log(stage / LOG)
            brought = self._brought(stage, rows)
            axis[stage] = (brought, before, rows)
            own = [
                r.steps - brought
                for r, s in zip(self.recordings, self._stage_of, strict=True)
                if s == stage
            ]
            before += max([*(r["steps"] for r in rows[-1:]), *own, 0.0])
        return axis

    def _brought(self, stage: Path, rows: list[dict[str, float]]) -> float:
        """Steps the stage's agent brought from before it (0 if unknown)."""
        by_iteration = {int(r["iteration"]): r["steps"] for r in rows}
        for recording, where in zip(self.recordings, self._stage_of, strict=True):
            iteration = recording.info.get("iteration")
            if where == stage and iteration in by_iteration:
                return recording.steps - by_iteration[iteration]
        return 0.0

    def positions(self) -> np.ndarray:
        """Where each recording is on the run's axis of steps (see the
        module)."""
        axis = self._axis()
        return np.array(
            [
                axis[s][1] + r.steps - axis[s][0]
                for r, s in zip(self.recordings, self._stage_of, strict=True)
            ],
            dtype=float,
        )

    def log(self) -> LogSeries:
        """Every stage's log, joined on the run's axis of steps."""
        rows: list[dict[str, float]] = []
        for _, before, stage_rows in self._axis().values():
            rows.extend({**r, "steps": r["steps"] + before} for r in stage_rows)

        def column(key: str) -> np.ndarray:
            return np.array([r[key] for r in rows], dtype=float)

        episodes = np.maximum(column("episodes"), 1.0)
        return LogSeries(
            steps=column("steps"),
            finished=column("finished") / episodes,
            crashed=column("crashed") / episodes,
            timeout=column("timeout") / episodes,
            mean_lap_time=column("mean_lap_time"),
            mean_return=column("mean_return"),
        )


def lap_numbers(recording: Recording, course: int | None = None) -> dict[str, float]:
    """What can be averaged over a recording's laps (on one course, or all):
    how many, the share that finished and that crashed, the mean lap time of
    those that finished (nan if none)."""
    laps = [lap for lap in recording.laps if course is None or lap.course == course]
    if not laps:
        return {"laps": 0, "finished": np.nan, "crashed": np.nan, "lap_time": np.nan}
    finished = [lap for lap in laps if lap.status == "finished"]
    return {
        "laps": len(laps),
        "finished": len(finished) / len(laps),
        "crashed": sum(lap.status == "crashed" for lap in laps) / len(laps),
        "lap_time": (
            float(np.mean([lap.lap_time for lap in finished])) if finished else np.nan
        ),
    }


@dataclass
class Progress:
    """The recordings' numbers on one axis of steps (one entry per recording
    that has laps on the course): the share of its laps that finished, their
    mean lap time [s] (nan if none), and the first lap's (the mean action's)
    lap time [s] (nan unless it finished)."""

    steps: np.ndarray
    finished: np.ndarray
    lap_time: np.ndarray
    first_lap_time: np.ndarray


def progress(
    recordings: list[Recording], course: int = 0, positions: np.ndarray | None = None
) -> Progress:
    """What the recordings say about learning, on one course; each at its
    position on the axis (by default its steps)."""
    if positions is None:
        positions = np.array([r.steps for r in recordings], dtype=float)
    steps, finished, lap_time, first = [], [], [], []
    for recording, x in zip(recordings, positions, strict=True):
        numbers = lap_numbers(recording, course)
        if numbers["laps"] == 0:
            continue
        laps = [lap for lap in recording.laps if lap.course == course]
        steps.append(x)
        finished.append(numbers["finished"])
        lap_time.append(numbers["lap_time"])
        first.append(laps[0].lap_time)
    return Progress(
        np.array(steps, dtype=float),
        np.array(finished, dtype=float),
        np.array(lap_time, dtype=float),
        np.array(first, dtype=float),
    )
