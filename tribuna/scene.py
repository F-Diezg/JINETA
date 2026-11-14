"""The viewer's window: PyVista draws what the player says (phase 13).

PyVista is a Python layer over VTK, the Visualization Toolkit (the library many
scientific programs draw with). A `pyvista.Plotter` is a window with a scene:
meshes are added to it as *actors* (a mesh with its colour, its opacity and its
place). Moving a drone does not rebuild its mesh: its actor gets a new
`user_matrix`, the 4 x 4 matrix that puts its body axes where the recording
says (`geometry.pose_matrix`). A path that grows gets its points replaced
(`copy_from`). The viewer runs its own loop (`play`), about 30 times a second:
it lets the window handle the keys and the mouse, moves the player on by the
time that passed (`Viewer.tick`), and draws. Not a VTK timer: on Windows
PyVista runs its own loop instead of VTK's, and a timer made before the window
exists never reaches it (found on the author's computer: the first frame drawn,
then nothing moved). The loop is PyVista's own for Windows, with the tick
added.

Seeing a policy learn: the earlier recordings' laps stay as fading trails
(`Player.trails`), and two charts beside the scene show what can be averaged:
how the training's episodes ended (the share that finished and that crashed,
from its log) and the lap time (the log's mean, and each recording's first
lap), with a dashed line at the recording being played. The slider at the top
left moves through the recordings (the training's time); the one at the top
right through the lap.

The scene draws in NED, as the simulator computes; the camera's "up" is -z.

Keys (also shown in the window):

    space          pause / play
    Right, Left    next / previous recording (later / earlier in the training)
    Home, End      the first / the newest recording
    l, k           faster / slower
    c              the next course (random courses: one lap on each)
    g              follow the first drone / back to the whole room
    h              show / hide the first drone's collision spheres
    q              close

Only this module imports PyVista, and only when a scene is made: everything
else of the viewer is tested without it.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from jineta.campo.course import Course
from jineta.picadero.recording import DroneShape
from tribuna.geometry import (
    Mesh,
    course_meshes,
    drone_meshes,
    hitbox_mesh,
    overview_camera,
    path_lines,
    pose_matrix,
    sphere,
)
from tribuna.player import TRAILS, Frame, Player, Trail, course_key, describe
from tribuna.run import LogSeries, Progress, progress

# Colours (names PyVista knows, or hex).
STATUS_COLORS = {"finished": "#2ca02c", "crashed": "#d62728", "timeout": "#7f7f7f"}
FLYING_COLOR = "#1f77b4"
GATE_COLOR = "#ff9f1c"
OBSTACLE_COLOR = "#8da9c4"
ROOM_COLOR = "#555555"
GRID_COLOR = "#cfcfcf"
BACKGROUND = "white"

HELP = (
    "space pause | Right/Left recording | Home/End first/newest | "
    "l/k faster/slower | c course | g follow | h hitbox | q close"
)


def polydata(mesh: Mesh) -> Any:
    """A mesh as a pyvista.PolyData (its faces and lines in VTK's layout)."""
    import pyvista as pv

    faces = mesh.faces if len(mesh.faces) else None
    lines = mesh.lines if len(mesh.lines) else None
    return pv.PolyData(mesh.points, faces=faces, lines=lines)


def _line_points(path: np.ndarray) -> np.ndarray:
    """At least two points for a polyline (a path that has not moved yet is a
    line of length zero)."""
    return path if len(path) >= 2 else np.vstack([path[:1], path[:1]])


def _at(point: np.ndarray) -> np.ndarray:
    """The matrix that moves a marker made at the origin to a point."""
    return pose_matrix(point, [1.0, 0.0, 0.0, 0.0])


@dataclass
class _Drone:
    """The actors of one drone: its frame and discs (placed by a matrix), its
    path so far, and a marker where it touched something."""

    parts: list[Any]
    path: Any
    path_actor: Any
    crash: Any


class Scene:
    """The course, the drones of a recording and the trails of earlier ones,
    drawn in a plotter."""

    def __init__(self, plotter: Any, shape: DroneShape, drones: int) -> None:
        self.plotter = plotter
        self.shape = shape
        self.course_actors: list[str] = []
        self.shown: tuple | None = None
        self.follow = False
        self.show_hitbox = False
        plotter.set_background(BACKGROUND)
        meshes = drone_meshes(shape)
        frame, ccw, cw = (
            polydata(m) for m in (meshes.frame, meshes.discs_ccw, meshes.discs_cw)
        )
        marker = polydata(sphere([0.0, 0.0, 0.0], 0.08))

        def add(mesh: Any, name: str, **style: Any) -> Any:
            return plotter.add_mesh(mesh, name=name, reset_camera=False, **style)

        self.drones: list[_Drone] = []
        for i in range(drones):
            parts = [
                add(frame, f"drone-{i}-frame", color="#333333"),
                add(ccw, f"drone-{i}-ccw", color="#e4572e", opacity=0.75),
                add(cw, f"drone-{i}-cw", color="#2e86ab", opacity=0.75),
            ]
            path = polydata(path_lines(np.zeros((2, 3))))
            main = i == 0  # the first lap (the mean action's) stands out
            path_actor = add(
                path, f"drone-{i}-path", color=FLYING_COLOR,
                line_width=4 if main else 2, opacity=1.0 if main else 0.6,
            )  # fmt: skip
            crash = add(marker, f"drone-{i}-crash", color=STATUS_COLORS["crashed"])
            self.drones.append(_Drone(parts, path, path_actor, crash))
        self.hitbox = add(
            polydata(hitbox_mesh(shape)), "drone-0-hitbox", color="#f3c623",
            opacity=0.3,
        )  # fmt: skip
        self.hitbox.visibility = False
        self.trail_actors: list[tuple[Any, Any, Any]] = []
        for i in range(TRAILS):
            line = polydata(path_lines(np.zeros((2, 3))))
            actor = add(line, f"trail-{i}", color=FLYING_COLOR, line_width=2)
            end = add(marker, f"trail-{i}-crash", color=STATUS_COLORS["crashed"])
            actor.visibility = False
            end.visibility = False
            self.trail_actors.append((line, actor, end))
        self.text = plotter.add_text(
            "", position="upper_left", font_size=10, color="black", name="status"
        )
        plotter.add_text(
            HELP, position="lower_edge", font_size=8, color="black", name="help"
        )

    def show_course(self, course: Course) -> None:
        """Draw a course (once: nothing is redrawn while it stays the same)."""
        key = course_key(course)
        if key == self.shown:
            return
        for name in self.course_actors:
            self.plotter.remove_actor(name)
        meshes = course_meshes(course)
        parts = [
            ("course-grid", meshes.grid, {"color": GRID_COLOR, "line_width": 1}),
            ("course-edges", meshes.edges, {"color": ROOM_COLOR, "line_width": 2}),
        ]
        parts += [
            (f"course-gate-{i}", m, {"color": GATE_COLOR, "smooth_shading": True})
            for i, m in enumerate(meshes.gates)
        ]
        parts += [
            (f"course-obstacle-{i}", m, {"color": OBSTACLE_COLOR, "opacity": 0.7})
            for i, m in enumerate(meshes.obstacles)
        ]
        start = sphere(meshes.start, 0.06)
        parts.append(("course-start", start, {"color": "black"}))
        for name, mesh, style in parts:
            self.plotter.add_mesh(
                polydata(mesh), name=name, reset_camera=False, **style
            )
        self.course_actors = [name for name, _, _ in parts]
        self.shown = key
        self.overview(course)

    def overview(self, course: Course) -> None:
        """The camera over the whole room, from above its south-east side
        (`geometry.overview_camera`)."""
        position, focus, up = overview_camera(course.room)
        camera = self.plotter.camera
        camera.focal_point = tuple(focus)
        camera.position = tuple(position)
        camera.up = tuple(up)

    def chase(self, position: np.ndarray, attitude_matrix: np.ndarray) -> None:
        """The camera behind and above a drone, looking where it is heading."""
        forward = attitude_matrix[:, 0].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        forward = forward / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        camera = self.plotter.camera
        camera.focal_point = tuple(position + 1.5 * forward)
        camera.position = tuple(position - 2.5 * forward + np.array([0.0, 0.0, -1.2]))
        camera.up = (0.0, 0.0, -1.0)

    def draw_trails(self, trails: list[Trail]) -> None:
        """The earlier recordings' laps, fading with their age."""
        for i, (line, actor, end) in enumerate(self.trail_actors):
            if i >= len(trails):
                actor.visibility = False
                end.visibility = False
                continue
            trail = trails[i]
            line.copy_from(polydata(path_lines(_line_points(trail.points))))
            actor.prop.color = STATUS_COLORS[trail.status]
            actor.prop.opacity = trail.opacity
            actor.visibility = True
            end.visibility = trail.crash_point is not None
            if trail.crash_point is not None:
                end.user_matrix = _at(trail.crash_point)
                end.prop.opacity = trail.opacity

    def draw(self, frame: Frame, text: str, trails: list[Trail] | None = None) -> None:
        """Put every actor where the frame says (and the trails, when given)."""
        course = frame.recording.courses[frame.course]
        self.show_course(course)
        if trails is not None:
            self.draw_trails(trails)
        for i, drone in enumerate(self.drones):
            visible = i < len(frame.drones)
            for actor in drone.parts:
                actor.visibility = visible
            drone.path_actor.visibility = visible
            if not visible:
                drone.crash.visibility = False
                continue
            state = frame.drones[i]
            matrix = pose_matrix(state.position, state.attitude)
            for actor in drone.parts:
                actor.user_matrix = matrix
            drone.path.copy_from(polydata(path_lines(_line_points(state.path))))
            color = STATUS_COLORS[state.status] if state.ended else FLYING_COLOR
            drone.path_actor.prop.color = color
            drone.crash.visibility = state.crash_point is not None
            if state.crash_point is not None:
                drone.crash.user_matrix = _at(state.crash_point)
            if i == 0:
                self.hitbox.visibility = self.show_hitbox
                self.hitbox.user_matrix = matrix
                if self.follow:
                    self.chase(state.position, matrix[0:3, 0:3])
        if not frame.drones:
            self.hitbox.visibility = False
        self.text.set_text("upper_left", text)


def _finite(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The points of a series that have a value (a chart draws no nan)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _upper(*series: np.ndarray, floor: float = 1.0) -> float:
    """A round top for an axis: a tenth above every finite value (at least
    `floor`), rounded up to a whole number."""
    tops = [floor]
    for values in series:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            tops.append(float(np.max(values)))
    return float(np.ceil(max(tops) * 1.1))


class Charts:
    """Two charts beside the scene (see the module): how the episodes ended,
    and the lap time, against training steps."""

    def __init__(self, plotter: Any) -> None:
        import pyvista as pv

        ends = pv.Chart2D(
            size=(0.3, 0.27), loc=(0.69, 0.40), x_label="steps", y_label="share"
        )
        ends.title = "how the episodes of each update ended"
        times = pv.Chart2D(
            size=(0.3, 0.27), loc=(0.69, 0.09), x_label="steps", y_label="s"
        )
        times.title = "lap time"
        self.ends, self.times = ends, times
        zero = ([0.0, 0.0], [0.0, 0.0])
        green, red = STATUS_COLORS["finished"], STATUS_COLORS["crashed"]
        self.finished = ends.line(*zero, color=green, width=2, label="finished")
        self.crashed = ends.line(*zero, color=red, width=2, label="crashed")
        self.ends_now = ends.line(*zero, color="black", width=1, style="--")
        self.mean_time = times.line(
            *zero, color=FLYING_COLOR, width=2, label="training (mean)"
        )
        self.first_time = times.scatter(
            *zero, color="black", size=6, label="recorded (first lap)"
        )
        self.times_now = times.line(*zero, color="black", width=1, style="--")
        plotter.add_chart(ends, times)

    def update(self, log: LogSeries | None, numbers: Progress, steps: float) -> None:
        """New data (the log and the recordings so far), and the line at the
        recording being played (`steps`)."""
        has_log = log is not None and len(log) > 0
        lasts = [1.0, float(steps), *numbers.steps[-1:]]
        if has_log:
            lasts.append(float(log.steps[-1]))
            self.finished.update(log.steps, log.finished)
            self.crashed.update(log.steps, log.crashed)
            self.mean_time.update(*_finite(log.steps, log.mean_lap_time))
        right = max(lasts) * 1.02
        top = _upper(numbers.first_lap_time, log.mean_lap_time if has_log else [])
        self.first_time.update(*_finite(numbers.steps, numbers.first_lap_time))
        self.ends_now.update([steps, steps], [0.0, 1.0])
        self.times_now.update([steps, steps], [0.0, top])
        self.ends.x_range = [0.0, right]
        self.ends.y_range = [0.0, 1.0]
        self.times.x_range = [0.0, right]
        self.times.y_range = [0.0, top]


def play(
    process_events: Callable[[], None],
    done: Callable[[], bool],
    step: Callable[[float], None],
    render: Callable[[], None],
    frame: float = 1.0 / 30.0,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """The viewer's loop: handle the window's events (keys, mouse, closing),
    stop once it is done, move on by the time since the last frame (at most
    0.2 s: a stall, such as a window dragged, is not a jump) and draw, then
    wait for the rest of the frame. Returns the frames drawn."""
    frames = 0
    last = clock()
    while True:
        start = clock()
        process_events()
        if done():
            return frames
        step(min(start - last, 0.2))
        last = start
        render()
        frames += 1
        rest = frame - (clock() - start)
        if rest > 0.0:
            sleep(rest)


class Viewer:
    """A window playing recordings: the player, the scene, the charts, the
    keys, the sliders and a timer. `source()` (optional) is asked for new
    recordings every `poll_every` seconds (a training still writing them),
    `log()` (optional) for the training's log, and `positions()` (optional)
    for where the recordings are on the log's axis of steps (by default their
    steps; `tribuna.run`)."""

    def __init__(
        self,
        player: Player,
        shape: DroneShape,
        source: Callable[[], list] | None = None,
        plotter: Any = None,
        poll_every: float = 1.0,
        drones: int = 9,
        log: Callable[[], LogSeries] | None = None,
        positions: Callable[[], np.ndarray] | None = None,
    ) -> None:
        import pyvista as pv

        self.player = player
        self.source = source
        self.log = log
        self.positions = positions
        self.poll_every = poll_every
        self.plotter = plotter if plotter is not None else pv.Plotter(
            window_size=[1400, 860], title="Jineta - tribuna"
        )  # fmt: skip
        self.scene = Scene(self.plotter, shape, drones)
        self.charts = Charts(self.plotter)
        self._polled = 0.0
        self._shown: tuple | None = None  # what the trails and charts are of
        self.slider: Any = None
        self.timeline: Any = None
        self.refresh()

    def refresh(self) -> None:
        """The charts with the latest data."""
        recordings = self.player.recordings
        x = np.array([r.steps for r in recordings], dtype=float)
        if self.positions is not None:
            given = np.asarray(self.positions(), dtype=float)
            if len(given) == len(recordings):
                x = given
        now = float(x[self.player.index]) if len(recordings) else 0.0
        log = self.log() if self.log is not None else None
        numbers = progress(recordings, self.player.course, x)
        self.charts.update(log, numbers, now)

    def tick(self, dt: float) -> None:
        """One frame: new recordings and the log (now and then), the player
        moved on by dt, the scene drawn."""
        self._polled += dt
        if self.source is not None and self._polled >= self.poll_every:
            self._polled = 0.0
            self.player.add(self.source())
            self.refresh()
        self.player.advance(dt)
        frame = self.player.frame()
        if frame is None:
            self.scene.text.set_text(
                "upper_left", "waiting for the first recording ..."
            )
            return
        text = describe(
            frame, self.player.speed, self.player.paused, len(self.player.recordings)
        )
        trails = None
        shown = (id(frame.recording), frame.course, len(self.player.recordings))
        if shown != self._shown:  # another recording: its trails, the charts' line
            self._shown = shown
            trails = self.player.trails()
            self.refresh()
        self.scene.draw(frame, text, trails)
        if self.slider is not None:
            representation = self.slider.GetRepresentation()
            representation.SetMaximumValue(max(frame.duration, 1e-3))
            representation.SetValue(frame.t)
        if self.timeline is not None:
            representation = self.timeline.GetRepresentation()
            representation.SetMaximumValue(max(len(self.player.recordings) - 1, 1))
            representation.SetValue(self.player.index)

    def keys(self) -> dict[str, Callable[[], None]]:
        """What each key does."""
        player, scene = self.player, self.scene

        def follow() -> None:
            scene.follow = not scene.follow
            frame = player.frame()
            if not scene.follow and frame is not None:
                scene.overview(frame.recording.courses[frame.course])

        def hitbox() -> None:
            scene.show_hitbox = not scene.show_hitbox

        return {
            "space": player.toggle_pause,
            "Right": player.next,
            "Left": player.previous,
            "Home": player.first,
            "End": player.newest,
            "l": player.faster,
            "k": player.slower,
            "c": player.next_course,
            "g": follow,
            "h": hitbox,
        }

    def go_to(self, value: float) -> None:
        """The recordings slider: play the recording it points at."""
        self.player.go_to(round(value))

    def show(self, frame_ms: int = 33) -> None:
        """Open the window and play until it is closed."""
        for key, action in self.keys().items():
            self.plotter.add_key_event(key, action)
        style = {"style": "modern", "color": "black", "title_height": 0.02}
        self.timeline = self.plotter.add_slider_widget(
            self.go_to, [0.0, max(len(self.player.recordings) - 1, 1)],
            value=float(self.player.index), title="recording",
            pointa=(0.30, 0.92), pointb=(0.52, 0.92), interaction_event="end",
            fmt="%.0f", **style,
        )  # fmt: skip
        self.slider = self.plotter.add_slider_widget(
            self.player.seek, [0.0, 1.0], value=0.0, title="lap clock [s]",
            pointa=(0.58, 0.92), pointb=(0.95, 0.92), interaction_event="always",
            fmt="%.2f", **style,
        )  # fmt: skip
        self.tick(0.0)
        # The window opens and show() returns at once (interactive_update):
        # from then on the loop below handles its events and draws it.
        self.plotter.show(interactive_update=True, auto_close=False)
        interactor = self.plotter.iren.interactor

        def done() -> bool:
            return bool(interactor.GetDone()) or self.plotter.render_window is None

        play(
            interactor.ProcessEvents,
            done,
            self.tick,
            self.plotter.render,
            frame_ms / 1000,
        )
        self.plotter.close()
