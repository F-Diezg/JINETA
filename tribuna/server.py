"""The viewer's server: a run folder served to the web page (phase 13).

The viewer is a web page (`tribuna/web/`, drawn with three.js) shown in a
desktop window (pywebview) or a browser. A page cannot read files from the disk
by itself, so a small HTTP server from Python's standard library runs on this
computer only (127.0.0.1) and answers it:

    /                          the page and its files (tribuna/web/)
    /api/poll?known=k          the summaries (see `summary`) of the recordings
                               after the first k in the run's order, and for
                               every recording its key (stage and steps; the
                               page checks its first k are still the same),
                               place on the run's axis of steps and stage; the
                               files that could not be read
    /api/recording?index=i     recording i whole, as its file holds it
    /api/course?index=i&course=c
                               that course as surfaces and lines for three.js
                               (`geometry`: the shapes the collision model
                               uses) and the camera that sees the whole room
    /api/drone?index=i         the drone of recording i, in its body axes
    /api/log                   the training's log on the run's axis

The page asks /api/poll every second: a training that is still running adds
recordings, and the page plays them when it gets there. A summary is small (the
laps' endings and the first lap's path on each course, for the trails); the
whole recording (every lap, every 0.02 s) is asked for only when it is played:
a long training has hundreds of recordings, and sending all of them every
second would be tens of megabytes. The server only reads; a training never
notices it.

Every number goes out as JSON; nan and infinities as null. The page draws in
NED, as the simulator computes (its camera's up is -z).
"""

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from jineta.campo.course import Course, course_to_dict
from jineta.picadero.recording import (
    DroneShape,
    Recording,
    read_recording,
    recording_to_dict,
)
from tribuna.geometry import (
    course_meshes,
    drone_meshes,
    hitbox_mesh,
    lines_json,
    overview_camera,
    surface_json,
)
from tribuna.run import Run

WEB = Path(__file__).resolve().parent / "web"

# What each file is, said explicitly: on Windows Python's guess comes from the
# registry, which can say text/plain for .js, and a browser refuses to run a
# module script served as text/plain.
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".txt": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}

TRAIL_DECIMALS = 3  # [m] to the millimetre: plenty for a line on the screen


def _digest(data: Any) -> str:
    """A short name for some JSON data, the same for the same contents."""
    text = json.dumps(data, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def course_id(course: Course) -> str:
    """A course's name for the page: the same for the same course (the page
    keeps the courses it has by it, and draws trails of earlier recordings
    only on the course they flew)."""
    return _digest(course_to_dict(course))


def shape_id(shape: DroneShape) -> str:
    """A drone's name for the page (it keeps the drones it has by it)."""
    return _digest(
        [
            shape.rotors.tolist(),
            shape.spins.tolist(),
            shape.propeller_radius,
            shape.hitbox_centers.tolist(),
            shape.hitbox_radii.tolist(),
        ]
    )


def plain(values: Any) -> Any:
    """Numbers for JSON: arrays as (nested) lists, nan and infinities as None."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        value = float(array)
        return value if np.isfinite(value) else None
    return [plain(v) for v in array]


def summary(recording: Recording) -> dict[str, Any]:
    """What the page needs of a recording before it plays it: its steps and
    information, its courses (names and ids) and drone (id), how each lap
    ended and how long it lasts, and each course's first lap's path (the
    trail it leaves while later recordings play)."""
    trails = []
    for c in range(len(recording.courses)):
        laps = [lap for lap in recording.laps if lap.course == c]
        trails.append(
            np.round(laps[0].positions, TRAIL_DECIMALS).tolist() if laps else None
        )
    return {
        "steps": recording.steps,
        "info": json.loads(json.dumps(recording_to_dict(recording)["info"])),
        "courses": [{"id": course_id(c), "name": c.name} for c in recording.courses],
        "shape": shape_id(recording.shape),
        "laps": [
            {
                "course": lap.course,
                "label": lap.label,
                "status": lap.status,
                "lap_time": plain(lap.lap_time),
                "duration": lap.duration,
                "crash_time": plain(lap.crash_time),
                "crash_point": plain(lap.crash_point),
            }
            for lap in recording.laps
        ],
        "trails": trails,
    }


def course_json(course: Course) -> dict[str, Any]:
    """A course as the page draws it."""
    meshes = course_meshes(course)
    position, focus, up = overview_camera(course.room)
    return {
        "id": course_id(course),
        "name": course.name,
        "room": plain(course.room.size),
        "edges": lines_json(meshes.edges),
        "grid": lines_json(meshes.grid),
        "gates": [
            {"surface": surface_json(m), "round": not hasattr(g, "width")}
            for m, g in zip(meshes.gates, course.gates, strict=True)
        ],
        "obstacles": [surface_json(m) for m in meshes.obstacles],
        "start": plain(meshes.start),
        "camera": {"position": plain(position), "focus": plain(focus), "up": plain(up)},
    }


def drone_json(shape: DroneShape) -> dict[str, Any]:
    """A drone in its body axes (FRD [m]): body and arms, the rotor discs that
    turn each way, the collision spheres."""
    meshes = drone_meshes(shape)
    return {
        "id": shape_id(shape),
        "frame": surface_json(meshes.frame),
        "ccw": surface_json(meshes.discs_ccw),
        "cw": surface_json(meshes.discs_cw),
        "hitbox": surface_json(hitbox_mesh(shape)),
    }


LOG_KEYS = ("steps", "finished", "crashed", "timeout", "mean_lap_time")


class Feed:
    """What the page asks for, from a run folder or one recording file.

    The server answers each request in its own thread, so a lock keeps the
    run consistent while it reads new files. The JSON of each recording is
    made once and kept (a recording never changes once written)."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        self.lock = threading.Lock()
        self.run: Run | None = None
        self.single: list[Recording] = []
        if path.is_file():
            self.single = [read_recording(path)]
        else:
            self.run = Run(path)
        self._summaries: dict[int, dict[str, Any]] = {}

    def recordings(self) -> list[Recording]:
        return self.run.recordings if self.run is not None else self.single

    def _summary(self, recording: Recording) -> dict[str, Any]:
        key = id(recording)
        if key not in self._summaries:
            self._summaries[key] = summary(recording)
        return self._summaries[key]

    def poll(self, known: int = 0) -> dict[str, Any]:
        """New files read, the summaries of the recordings after the first
        `known` in the run's order, and every recording's key, place on the
        axis and stage; the unreadable files."""
        with self.lock:
            if self.run is not None:
                self.run.poll()
                positions = plain(self.run.positions())
                stages = [
                    self.run.stage(i).name for i in range(len(self.run.recordings))
                ]
                errors = [f"{p}: {e}" for p, e in self.run.errors.items()]
            else:
                positions = [float(r.steps) for r in self.single]
                stages, errors = ["" for _ in self.single], []
            recordings = self.recordings()
            return {
                "known": max(known, 0),
                "keys": [
                    f"{s}:{r.steps}" for s, r in zip(stages, recordings, strict=True)
                ],
                "recordings": [self._summary(r) for r in recordings[max(known, 0) :]],
                "positions": positions,
                "stages": stages,
                "errors": errors,
            }

    def _get(self, index: int) -> Recording:
        """Recording `index` (the files are read first if it is not known yet);
        IndexError when there is none."""
        if index < 0:
            raise IndexError(index)
        if self.run is not None and index >= len(self.run.recordings):
            self.run.poll()
        return self.recordings()[index]

    def recording(self, index: int) -> dict[str, Any]:
        with self.lock:
            return recording_to_dict(self._get(index))

    def course(self, index: int, course: int) -> dict[str, Any]:
        with self.lock:
            courses = self._get(index).courses
            if course < 0:
                raise IndexError(course)
            return course_json(courses[course])

    def drone(self, index: int) -> dict[str, Any]:
        with self.lock:
            return drone_json(self._get(index).shape)

    def log(self) -> dict[str, Any]:
        """The training's log on the run's axis (empty lists for a recording
        file)."""
        with self.lock:
            if self.run is None:
                return {key: [] for key in LOG_KEYS}
            log = self.run.log()
            return {key: plain(getattr(log, key)) for key in LOG_KEYS}


def handler_for(feed: Feed, web: Path = WEB) -> type[BaseHTTPRequestHandler]:
    """The request handler of a server that serves `feed` and the page."""
    web = web.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass  # quiet: the page asks every second

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data: Any) -> None:
            body = json.dumps(data, separators=(",", ":"), allow_nan=False)
            self._send(200, body.encode("utf-8"), CONTENT_TYPES[".json"])

        def do_GET(self) -> None:  # the name http.server calls
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/api/poll":
                    self._json(feed.poll(int(query.get("known", 0))))
                elif url.path == "/api/recording":
                    self._json(feed.recording(int(query["index"])))
                elif url.path == "/api/course":
                    index, course = int(query["index"]), int(query["course"])
                    self._json(feed.course(index, course))
                elif url.path == "/api/drone":
                    self._json(feed.drone(int(query["index"])))
                elif url.path == "/api/log":
                    self._json(feed.log())
                else:
                    self._file(url.path)
            except (KeyError, IndexError, ValueError) as error:
                self._send(400, f"bad request: {error!r}".encode(), "text/plain")

        def _file(self, path: str) -> None:
            """A file of the page; nothing outside its folder."""
            name = "index.html" if path in ("", "/") else path.lstrip("/")
            target = (web / name).resolve()
            if (
                web not in target.parents
                or not target.is_file()
                or target.suffix not in CONTENT_TYPES
            ):
                self._send(404, b"not found", "text/plain")
                return
            self._send(200, target.read_bytes(), CONTENT_TYPES[target.suffix])

    return Handler


def start(path: str | Path, port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Serve a run folder (or a recording) on 127.0.0.1 from a background
    thread; port 0 takes a free one. Returns the server (`shutdown()` stops
    it) and the page's address."""
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(Feed(path)))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, chosen = server.server_address[:2]
    return server, f"http://{host}:{chosen}/"
