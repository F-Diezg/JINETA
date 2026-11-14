"""A training's folder read as it grows: its recordings and its log.

A training writes `log.csv` (a row per update), `laps/<steps>.json` (the laps
of every evaluation, `jineta.picadero.recording`) and `agents/<steps>.npz`. A
two-stage training (`examples.train_course`, `examples.train_realism`) writes
two such folders side by side (`fly/`, `margin/`); a replay
(`examples.record_agent`) writes only `laps/` and `agents/`. `Run` takes any of
them, or a folder of recordings itself:

    run = Run("runs/realism")      # both stages
    new = run.poll()               # the recordings written since the last poll
    log = run.log()                # every stage's log, on one axis of steps

`poll` can be called as often as wanted (the viewer calls it every second while
a training goes on): it reads only files it has not read, and a file is never
half-written (they are renamed into place whole). A file that cannot be read
is reported once in `errors` and skipped, so a viewer never stops on it.

The log's "steps" count each stage's own steps, while a recording's count every
step its agent was trained for (a second stage's agent brought the first
stage's): a stage's log is moved by the difference, found from a recording of
that stage (a stage without recordings is left as it is).
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
    """A training's log as arrays, one entry per update, on the recordings'
    axis of steps: the share of the episodes that ended in that update which
    finished, crashed or ran out of time, their mean lap time [s] (nan when none
    finished) and mean return."""

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

    def poll(self) -> list[Recording]:
        """Read the recordings written since the last call; returns them in
        order of steps (and keeps every recording, in that order, in
        `recordings`)."""
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
        new.sort(key=lambda pair: pair[0].steps)
        pairs = sorted(
            [*zip(self.recordings, self._stage_of, strict=True), *new],
            key=lambda pair: pair[0].steps,
        )
        self.recordings = [r for r, _ in pairs]
        self._stage_of = [s for _, s in pairs]
        return [r for r, _ in new]

    def stage(self, index: int) -> Path:
        """The folder of the stage that wrote recording `index`."""
        return self._stage_of[index]

    def log(self) -> LogSeries:
        """Every stage's log, joined on the recordings' axis of steps."""
        rows: list[dict[str, float]] = []
        for stage in _stage_folders(self.folder):
            stage_rows = read_log(stage / LOG)
            offset = self._offset(stage, stage_rows)
            rows.extend({**r, "steps": r["steps"] + offset} for r in stage_rows)
        rows.sort(key=lambda r: r["steps"])

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

    def _offset(self, stage: Path, rows: list[dict[str, float]]) -> float:
        """Steps the stage's agent brought from before it (0 if unknown)."""
        by_iteration = {int(r["iteration"]): r["steps"] for r in rows}
        for recording, where in zip(self.recordings, self._stage_of, strict=True):
            iteration = recording.info.get("iteration")
            if where == stage and iteration in by_iteration:
                return recording.steps - by_iteration[iteration]
        return 0.0


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
