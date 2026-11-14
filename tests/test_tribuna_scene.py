"""Tests for the viewer's window (phase 13, `tribuna.scene`), drawn off screen.
Skipped without PyVista (pip install pyvista): Claude's workspace cannot
install it, so these run on the author's computer."""

import numpy as np
import pytest

from jineta.doma.policy import Agent
from jineta.picadero.env import CourseEnv
from jineta.picadero.recording import Recording, laps_from_flight
from jineta.picadero.training import fly
from tests.test_recording import COURSE, POLICY
from tests.test_training import ConstantAgent
from tests.test_tribuna_player import SHAPE
from tribuna.geometry import overview_camera, pose_matrix
from tribuna.motion import pose_at
from tribuna.player import Player

pv = pytest.importorskip("pyvista")
from tribuna.scene import STATUS_COLORS, Scene, Viewer, course_key  # noqa: E402


@pytest.fixture(scope="module")
def recordings():
    lap = fly(Agent.load(POLICY), CourseEnv(COURSE), record=True)
    up = fly(ConstantAgent([1.0, 0.0, 0.0, 0.0]), CourseEnv(COURSE), record=True)
    laps = (
        *laps_from_flight(lap, [0], ["mean action"]),
        *laps_from_flight(up, [0], ["straight up"]),
    )
    first = Recording(1000, (COURSE,), SHAPE, laps, {"iteration": 10})
    second = Recording(2000, (COURSE,), SHAPE, laps[:1], {"iteration": 20})
    return first, second


@pytest.fixture
def plotter():
    plotter = pv.Plotter(off_screen=True, window_size=[640, 400])
    yield plotter
    plotter.close()


def test_a_frame_is_drawn_where_the_player_says(plotter, recordings):
    first, _ = recordings
    scene = Scene(plotter, SHAPE, drones=4)
    player = Player([first])
    player.seek(1.0)
    scene.draw(player.frame(), "text")
    mean = first.laps[0]
    expected = pose_matrix(*pose_at(mean, 1.0))
    for actor in scene.drones[0].parts:
        np.testing.assert_allclose(actor.user_matrix, expected, atol=1e-12)
    # Two laps in the recording: two drones shown, the other two hidden.
    assert [d.parts[0].visibility for d in scene.drones] == [True, True, False, False]
    assert scene.drones[0].path.n_points == int(np.sum(mean.times < 1.0)) + 1
    up = scene.drones[1]
    assert not up.crash.visibility  # still climbing
    assert up.path_actor.prop.color == pv.Color("#1f77b4")
    # The straight-up lap has hit the ceiling by 1.5 s: its path is red, its
    # crash marked where it touched.
    assert first.laps[1].crash_time < 1.5 < mean.duration
    player.seek(1.5)
    scene.draw(player.frame(), "text")
    assert up.crash.visibility
    np.testing.assert_allclose(up.crash.user_matrix[0:3, 3], first.laps[1].crash_point)
    assert up.path_actor.prop.color == pv.Color(STATUS_COLORS["crashed"])
    assert not scene.hitbox.visibility
    image = plotter.screenshot()
    assert image.ndim == 3 and image.shape[0:2] == (400, 640)
    assert image.std() > 0.0  # something was drawn


def test_the_course_is_drawn_once(plotter, recordings):
    first, _ = recordings
    scene = Scene(plotter, SHAPE, drones=2)
    player = Player([first])
    scene.draw(player.frame(), "")
    names = list(scene.course_actors)
    assert len(names) == 2 + 4 + 3 + 1  # grid, edges, gates, obstacles, start
    assert scene.shown == course_key(COURSE)
    camera = plotter.camera
    assert tuple(camera.up) == pytest.approx((0.0, 0.0, -1.0))
    np.testing.assert_allclose(camera.focal_point, [9.0, 5.0, -4.0 / 3.0])
    position, _, _ = overview_camera(COURSE.room)
    np.testing.assert_allclose(camera.position, position)
    gate = plotter.actors["course-gate-0"]
    scene.draw(player.frame(), "")
    assert scene.course_actors == names
    assert plotter.actors["course-gate-0"] is gate  # not drawn again


def test_the_viewer_reads_new_recordings_and_follows_a_drone(plotter, recordings):
    first, second = recordings
    pending = [[], [second]]
    viewer = Viewer(Player([first]), SHAPE, lambda: pending.pop(0), plotter, 0.5)
    viewer.tick(0.3)
    assert len(viewer.player.recordings) == 1
    viewer.tick(0.3)  # 0.6 s: asked once
    assert len(viewer.player.recordings) == 1
    viewer.tick(0.5)  # asked again: the new one
    assert [r.steps for r in viewer.player.recordings] == [1000, 2000]
    keys = viewer.keys()
    assert {"space", "Right", "Left", "Home", "End", "l", "k", "c", "g", "h"} <= set(
        keys
    )
    keys["g"]()
    keys["h"]()
    viewer.tick(0.1)
    position = viewer.player.frame().drones[0].position
    assert np.linalg.norm(np.array(plotter.camera.focal_point) - position) < 2.0
    assert viewer.scene.hitbox.visibility
    keys["g"]()  # back to the whole room
    assert plotter.camera.focal_point == pytest.approx((9.0, 5.0, -4.0 / 3.0))
    keys["End"]()
    assert viewer.player.current is second
