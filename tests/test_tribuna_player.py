"""Tests for the viewer's playback (phase 13, `tribuna.player`): recordings
played in order of steps, new ones added while playing, the keys, and where
every drone is at the player's moment."""

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.jinete.profiles import MULTIROTOR_PROFILES
from jineta.picadero.recording import Recording, drone_shape
from tests.test_recording import COURSE, lap
from tribuna.geometry import sphere
from tribuna.motion import pose_at
from tribuna.player import PAUSE, SPEEDS, Player, describe

SHAPE = drone_shape(MULTIROTOR_PROFILES["iris"])


def flown(duration, status="timeout", label="mean action", course=0, **change):
    """A straight lap North at 1 m/s, sampled every 0.02 s."""
    times = np.arange(0.0, duration + 1e-9, 0.02)
    positions = np.column_stack(
        [times, np.full_like(times, 5.0), np.full_like(times, -1.0)]
    )
    attitudes = np.tile(quaternion.from_euler(0.0, 0.0, 0.0), (len(times), 1))
    values = {"lap_time": duration if status == "finished" else np.nan}
    values.update(change)
    return lap(
        times=times, positions=positions, attitudes=attitudes, status=status,
        label=label, course=course, **values,
    )  # fmt: skip


def recording(steps, *laps, courses=(COURSE,)):
    return Recording(steps, courses, SHAPE, laps, {"iteration": steps // 100})


def test_recordings_play_in_order_then_the_last_again():
    player = Player([recording(300, flown(1.0)), recording(100, flown(0.5))])
    assert [r.steps for r in player.recordings] == [100, 300]
    assert player.duration() == pytest.approx(0.5)
    player.advance(0.2)
    assert player.t == pytest.approx(0.2)
    player.advance(1.0)
    assert player.t == pytest.approx(0.5)  # held at the end of the laps
    player.advance(PAUSE - 0.01)
    assert player.index == 0  # still the pause after the end
    player.advance(0.02)
    assert (player.index, player.t) == (1, 0.0)
    player.advance(1.0)
    player.advance(PAUSE)
    assert (player.index, player.t) == (1, 0.0)  # the last one again


def test_the_pause_between_recordings_is_wall_time_at_any_speed():
    player = Player([recording(100, flown(0.4)), recording(200, flown(0.4))])
    for _ in range(3):
        player.faster()  # x8
    player.advance(0.05)  # 0.4 s of lap clock: the end
    assert player.t == pytest.approx(0.4)
    player.advance(PAUSE / 2)
    assert player.index == 0
    player.advance(PAUSE / 2 + 1e-9)
    assert player.index == 1
    # And after a jump (a key), a whole pause again at the end.
    player.advance(0.05)
    player.advance(PAUSE * 0.9)
    player.previous()
    player.advance(0.05)
    player.advance(PAUSE * 0.5)
    assert player.index == 0


def test_new_recordings_are_played_when_the_player_gets_there():
    player = Player([recording(100, flown(0.5))])
    player.advance(0.3)
    player.add([recording(200, flown(0.4)), recording(50, flown(0.2))])
    # The one being played stays the one being played (an earlier one arrived).
    assert [r.steps for r in player.recordings] == [50, 100, 200]
    assert player.index == 1 and player.t == pytest.approx(0.3)
    player.advance(0.2)
    player.advance(PAUSE)
    assert player.current.steps == 200
    player.add([])
    assert player.current.steps == 200


def test_the_keys():
    player = Player([recording(s, flown(0.5)) for s in (100, 200, 300)])
    player.advance(0.1)
    player.next()
    assert (player.index, player.t) == (1, 0.0)
    player.newest()
    player.next()  # nothing after the newest
    assert player.index == 2
    player.first()
    player.previous()
    assert player.index == 0
    player.toggle_pause()
    player.advance(0.3)
    assert player.t == 0.0 and player.paused
    player.toggle_pause()
    assert player.speed == 1.0
    player.faster()
    player.advance(0.1)
    assert (player.speed, player.t) == (2.0, pytest.approx(0.2))
    for _ in range(10):
        player.faster()
    assert player.speed == SPEEDS[-1]
    for _ in range(10):
        player.slower()
    assert player.speed == SPEEDS[0]
    player.seek(9.0)
    assert player.t == pytest.approx(0.5)
    player.seek(-1.0)
    assert player.t == 0.0
    empty = Player()
    empty.advance(1.0)
    empty.next()
    empty.next_course()
    assert empty.frame() is None and empty.duration() == 0.0


def test_one_course_at_a_time():
    second = COURSE.__class__(**{**COURSE.__dict__, "name": "Second"})
    third = COURSE.__class__(**{**COURSE.__dict__, "name": "Third"})
    both = recording(
        100, flown(0.5, label="First course"), flown(0.3, label="Second", course=1),
        flown(0.2, label="Third", course=2), courses=(COURSE, second, third),
    )  # fmt: skip
    one = recording(200, flown(0.4))
    player = Player([both, one])
    assert [lap.label for lap in player.laps()] == ["First course"]
    player.advance(0.2)
    player.next_course()
    assert player.course == 1 and player.t == 0.0
    assert [lap.label for lap in player.laps()] == ["Second"]
    assert player.duration() == pytest.approx(0.3)
    player.next_course()
    assert player.course == 2 and player.duration() == pytest.approx(0.2)
    player.next_course()
    assert player.course == 0
    player.next_course()
    player.next()  # the next recording has one course: back to the first
    assert player.course == 0 and player.current is one


def test_a_frame_says_where_every_drone_is():
    crash = flown(
        0.4, status="crashed", label="noisy 1", crash_time=0.33,
        crash_point=np.array([0.33, 5.0, -1.0]), crash_kind="obstacle",
    )  # fmt: skip
    mean = flown(0.6, status="finished")
    player = Player([recording(100, mean, crash)])
    player.seek(0.25)
    frame = player.frame()
    assert (frame.index, frame.course, frame.duration) == (0, 0, pytest.approx(0.6))
    assert [d.label for d in frame.drones] == ["mean action", "noisy 1"]
    for drone, flight in zip(frame.drones, (mean, crash), strict=True):
        position, attitude = pose_at(flight, 0.25)
        np.testing.assert_allclose(drone.position, position)
        np.testing.assert_allclose(drone.attitude, attitude)
        np.testing.assert_allclose(drone.path[-1], position)
        assert not drone.ended
    assert frame.drones[1].crash_point is None  # not there yet
    player.seek(0.35)
    frame = player.frame()
    np.testing.assert_allclose(frame.drones[1].crash_point, [0.33, 5.0, -1.0])
    player.seek(0.5)
    frame = player.frame()
    assert frame.drones[1].ended and not frame.drones[0].ended
    np.testing.assert_allclose(frame.drones[1].path, crash.positions)
    text = describe(frame, 2.0, True, 7)
    assert "recording 1 of 7: 100 steps (iteration 1)" in text
    assert "x2  (paused)" in text and "First course" in text
    assert "mean action: ..." in text and "noisy 1: crashed" in text
    player.seek(0.59)  # between its last two samples: not ended yet
    assert not player.frame().drones[0].ended
    player.seek(0.6)
    assert player.frame().drones[0].ended
    assert "mean action: 0.60 s" in describe(player.frame(), 1.0, False, 1)


def test_a_sphere_is_round_with_faces_outwards():
    center = np.array([1.0, -2.0, 0.5])
    mesh = sphere(center, 0.3, rings=6, segments=10)
    np.testing.assert_allclose(np.linalg.norm(mesh.points - center, axis=1), 0.3)
    from tests.test_tribuna import face_normals

    for centroid, normal in face_normals(mesh):
        assert np.dot(normal, centroid - center) > 0.0


def test_the_overview_camera_sees_the_whole_room():
    from jineta.campo.indoor import Room
    from tribuna.geometry import overview_camera

    for size in ([18.0, 10.0, 4.0], [6.0, 6.0, 3.0], [40.0, 8.0, 12.0]):
        room = Room(np.array(size))
        position, focus, up = overview_camera(room)
        np.testing.assert_allclose(focus, [size[0] / 2, size[1] / 2, -size[2] / 3])
        np.testing.assert_array_equal(up, [0.0, 0.0, -1.0])
        assert position[2] < -size[2]  # above the ceiling
        sight = (focus - position) / np.linalg.norm(focus - position)
        corners = np.array(
            [
                [x, y, z]
                for x in (0, size[0])
                for y in (0, size[1])
                for z in (0, -size[2])
            ]
        )
        rays = corners - position
        angles = np.degrees(np.arccos(rays @ sight / np.linalg.norm(rays, axis=1)))
        assert np.max(angles) == pytest.approx(0.9 * 15.0, abs=1e-9)  # the farthest


def test_trails_fade_with_their_age():
    from tribuna.player import NEWEST_OPACITY, OLDEST_OPACITY, trail_opacity

    opacities = [trail_opacity(age, 10) for age in range(1, 11)]
    assert opacities[0] == NEWEST_OPACITY and opacities[-1] == pytest.approx(
        OLDEST_OPACITY
    )
    assert np.all(np.diff(opacities) < 0.0)
    assert trail_opacity(0, 10) == trail_opacity(11, 10) == 0.0
    assert trail_opacity(1, 1) == NEWEST_OPACITY


def test_the_trails_are_the_earlier_recordings_first_laps():
    from tribuna.player import trail_opacity

    crash = flown(
        0.2, status="crashed", crash_time=0.15, crash_point=np.array([0.15, 5.0, -1.0])
    )
    recordings = [
        recording(100, crash, flown(0.3, label="noisy 1")),
        recording(200, flown(0.4)),
        recording(300, flown(0.5, status="finished")),
        recording(400, flown(0.6, status="finished")),
    ]
    player = Player(recordings)
    assert player.trails() == []  # nothing before the first
    player.newest()
    trails = player.trails(keep=2)
    assert [t.steps for t in trails] == [300, 200]  # the most recent first
    assert [t.opacity for t in trails] == [trail_opacity(1, 2), trail_opacity(2, 2)]
    assert [t.status for t in trails] == ["finished", "timeout"]
    np.testing.assert_array_equal(trails[0].points, recordings[2].laps[0].positions)
    oldest = player.trails(keep=10)[-1]
    assert oldest.steps == 100 and oldest.status == "crashed"
    np.testing.assert_allclose(oldest.crash_point, [0.15, 5.0, -1.0])
    assert all(t.crash_point is None for t in player.trails(keep=10)[:-1])


def test_trails_only_of_the_same_course():
    from tribuna.player import course_key

    moved = COURSE.__class__(
        **{**COURSE.__dict__, "gates": (COURSE.gates[0].__class__(
            **{**COURSE.gates[0].__dict__, "center": np.array([6.0, 5.0, -1.5])}
        ), *COURSE.gates[1:])}
    )  # fmt: skip
    assert course_key(moved) != course_key(COURSE)
    other = COURSE.__class__(**{**COURSE.__dict__, "name": "Other"})
    # Random courses: the same validation courses, but listed in another order.
    first = recording(
        100, flown(0.3, label="Other", course=0), flown(0.2, label="First", course=1),
        courses=(other, COURSE),
    )  # fmt: skip
    elsewhere = recording(200, flown(0.4), courses=(moved,))
    now = recording(300, flown(0.5), courses=(COURSE,))
    player = Player([first, elsewhere, now])
    player.newest()
    trails = player.trails()
    assert [t.steps for t in trails] == [100]  # not the moved course's lap
    np.testing.assert_array_equal(trails[0].points, first.laps[1].positions)
