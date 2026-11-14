"""The viewer's window: PyVista draws what the player says (phase 13).

PyVista is a Python layer over VTK, the Visualization Toolkit (the library many
scientific programs draw with). A `pyvista.Plotter` is a window with a scene:
meshes are added to it as *actors* (a mesh with its colour, its opacity and its
place). Moving a drone does not rebuild its mesh: its actor gets a new
`user_matrix`, the 4 x 4 matrix that puts its body axes where the recording
says (`geometry.pose_matrix`). A path that grows gets its points replaced
(`copy_from`). A timer calls `Viewer.tick` about 30 times a second: the player
moves on by the time that passed, and the actors follow.

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
from tribuna.player import Frame, Player, describe

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


def course_key(course: Course) -> tuple:
    """Enough of a course to tell two apart (to redraw it only when it
    changes)."""
    return (
        course.name,
        tuple(course.room.size),
        tuple(tuple(g.center) for g in course.gates),
        tuple(type(o).__name__ for o in course.obstacles),
    )


@dataclass
class _Drone:
    """The actors of one drone: its frame and discs (placed by a matrix), its
    path so far, and a marker where it touched something."""

    parts: list[Any]
    path: Any
    path_actor: Any
    crash: Any


class Scene:
    """The course and the drones of a recording, drawn in a plotter."""

    def __init__(self, plotter: Any, shape: DroneShape, drones: int) -> None:
        self.plotter = plotter
        self.shape = shape
        self.course_actors: list[str] = []
        self.shown: tuple | None = None
        plotter.set_background(BACKGROUND)
        meshes = drone_meshes(shape)
        frame, ccw, cw = (
            polydata(m) for m in (meshes.frame, meshes.discs_ccw, meshes.discs_cw)
        )
        marker = polydata(sphere([0.0, 0.0, 0.0], 0.08))
        self.drones: list[_Drone] = []
        for i in range(drones):
            parts = [
                plotter.add_mesh(
                    mesh,
                    color=color,
                    opacity=opacity,
                    name=f"drone-{i}-{part}",
                    reset_camera=False,
                )
                for part, mesh, color, opacity in (
                    ("frame", frame, "#333333", 1.0),
                    ("ccw", ccw, "#e4572e", 0.75),
                    ("cw", cw, "#2e86ab", 0.75),
                )
            ]
            path = polydata(path_lines(np.zeros((2, 3))))
            main = i == 0  # the first lap (the mean action's) stands out
            path_actor = plotter.add_mesh(
                path, color=FLYING_COLOR, line_width=4 if main else 2,
                opacity=1.0 if main else 0.6, name=f"drone-{i}-path",
                reset_camera=False,
            )  # fmt: skip
            crash = plotter.add_mesh(
                marker, color=STATUS_COLORS["crashed"], name=f"drone-{i}-crash",
                reset_camera=False,
            )  # fmt: skip
            self.drones.append(_Drone(parts, path, path_actor, crash))
        self.hitbox = plotter.add_mesh(
            polydata(hitbox_mesh(shape)), color="#f3c623", opacity=0.3,
            name="drone-0-hitbox", reset_camera=False,
        )  # fmt: skip
        self.hitbox.visibility = False
        self.text = plotter.add_text(
            "", position="upper_left", font_size=10, color="black", name="status"
        )
        plotter.add_text(
            HELP, position="lower_edge", font_size=8, color="black", name="help"
        )
        self.follow = False
        self.show_hitbox = False

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

    def draw(self, frame: Frame, text: str) -> None:
        """Put every actor where the frame says."""
        course = frame.recording.courses[frame.course]
        self.show_course(course)
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
                drone.crash.user_matrix = pose_matrix(
                    state.crash_point, [1.0, 0.0, 0.0, 0.0]
                )
            if i == 0:
                self.hitbox.visibility = self.show_hitbox
                self.hitbox.user_matrix = matrix
                if self.follow:
                    self.chase(state.position, matrix[0:3, 0:3])
        if not frame.drones:
            self.hitbox.visibility = False
        self.text.set_text("upper_left", text)


class Viewer:
    """A window playing recordings: the player, the scene, the keys and a
    timer. `source()` (optional) is asked for new recordings every
    `poll_every` seconds (a training still writing them)."""

    def __init__(
        self,
        player: Player,
        shape: DroneShape,
        source: Callable[[], list] | None = None,
        plotter: Any = None,
        poll_every: float = 1.0,
        drones: int = 9,
    ) -> None:
        import pyvista as pv

        self.player = player
        self.source = source
        self.poll_every = poll_every
        self.plotter = plotter if plotter is not None else pv.Plotter(
            window_size=[1400, 860], title="Jineta - tribuna"
        )  # fmt: skip
        self.scene = Scene(self.plotter, shape, drones)
        self._last = time.perf_counter()
        self._polled = 0.0
        self.slider: Any = None

    def tick(self, dt: float) -> None:
        """One frame: new recordings (now and then), the player moved on by dt,
        the scene drawn."""
        self._polled += dt
        if self.source is not None and self._polled >= self.poll_every:
            self._polled = 0.0
            self.player.add(self.source())
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
        self.scene.draw(frame, text)
        if self.slider is not None:
            representation = self.slider.GetRepresentation()
            representation.SetMaximumValue(max(frame.duration, 1e-3))
            representation.SetValue(frame.t)

    def _on_timer(self, _step: int) -> None:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        self.tick(min(dt, 0.2))  # a stall (a window dragged) is not a jump

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

    def show(self, frame_ms: int = 33) -> None:
        """Open the window and play until it is closed."""
        for key, action in self.keys().items():
            self.plotter.add_key_event(key, action)
        self.slider = self.plotter.add_slider_widget(
            self.player.seek, [0.0, 1.0], value=0.0, title="lap clock [s]",
            pointa=(0.55, 0.92), pointb=(0.95, 0.92), interaction_event="always",
            style="modern", fmt="%.2f", color="black",
        )  # fmt: skip
        self.tick(0.0)
        self._last = time.perf_counter()
        self.plotter.add_timer_event(
            max_steps=10**9, duration=frame_ms, callback=self._on_timer
        )
        self.plotter.show()
