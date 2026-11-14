"""Tests for the viewer's parts that need no window (phase 13, `tribuna`): the
course and the drone as meshes (every point on the surface the collision model
uses, faces turned outwards), where a recorded drone is between samples, and a
training's folder read as it grows."""

import dataclasses
import shutil

import numpy as np
import pytest

from jineta.campo.gates import CircularGate, RectangularGate
from jineta.campo.indoor import Box, Cylinder
from jineta.core import quaternion
from jineta.doma.policy import Agent
from jineta.doma.ppo import PPOSettings
from jineta.jinete.profiles import IRIS, MULTIROTOR_PROFILES
from jineta.picadero.recording import Lap, Recording, drone_shape, recording_files
from jineta.picadero.training import LAPS_FOLDER, save_recording, train
from tests.test_recording import COURSE, lap
from tests.test_training import SHORT
from tests.test_training_record import short_training
from tribuna.geometry import (
    Mesh,
    block,
    cell_lists,
    circular_frame,
    column,
    course_meshes,
    drone_meshes,
    floor_grid,
    merge,
    path_lines,
    pose_matrix,
    rectangular_frame,
    room_edges,
)
from tribuna.motion import path_until, pose_at, slerp
from tribuna.run import Run, lap_numbers, read_log

SHAPE = drone_shape(MULTIROTOR_PROFILES["iris"])


def face_normals(mesh: Mesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """(centroid, normal) of every face, the normal by the right-hand rule."""
    result = []
    for cell in cell_lists(mesh.faces):
        p = mesh.points[cell]
        normal = np.cross(p[1] - p[0], p[2] - p[0])
        result.append((p.mean(axis=0), normal / np.linalg.norm(normal)))
    return result


def assert_outwards(mesh: Mesh, inside_point):
    """Every face of a convex piece turns away from a point inside it."""
    for centroid, normal in face_normals(mesh):
        assert np.dot(normal, centroid - inside_point) > 0.0


# ---- the course as meshes --------------------------------------------------------


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0, -125.0])
def test_a_box_is_on_its_surface_with_faces_outwards(yaw_deg):
    box = Box(
        np.array([7.0, 7.5, -0.75]), np.array([1.5, 1.0, 2.0]), np.radians(yaw_deg)
    )
    mesh = block(box.center, box.size, box.yaw)
    assert mesh.points.shape == (8, 3) and len(cell_lists(mesh.faces)) == 6
    np.testing.assert_allclose(box.signed_distance(mesh.points), 0.0, atol=1e-12)
    assert_outwards(mesh, box.center)
    # Each face is flat: its four corners in one plane.
    for cell in cell_lists(mesh.faces):
        p = mesh.points[cell]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        assert abs(np.dot(n, p[3] - p[0])) < 1e-12


def test_a_column_is_on_its_surface_with_faces_outwards():
    pillar = Cylinder(np.array([11.5, 6.6, 0.0]), 0.3, 4.0)
    mesh = column(pillar.base, pillar.radius, pillar.height, sides=16)
    np.testing.assert_allclose(pillar.signed_distance(mesh.points), 0.0, atol=1e-12)
    assert np.min(mesh.points[:, 2]) == pytest.approx(-4.0)  # up to the ceiling
    assert np.max(mesh.points[:, 2]) == pytest.approx(0.0)
    assert_outwards(mesh, pillar.base + np.array([0.0, 0.0, -2.0]))
    assert len(cell_lists(mesh.faces)) == 16 + 2


@pytest.mark.parametrize("index", [0, 2])  # straight, and turned 30 degrees
def test_a_rectangular_frame_is_on_the_frame_and_leaves_the_opening_clear(index):
    gate = COURSE.gates[index]
    assert isinstance(gate, RectangularGate)
    mesh = rectangular_frame(gate)
    assert mesh.points.shape == (4 * 8, 3)
    np.testing.assert_allclose(gate.signed_distance(mesh.points), 0.0, atol=1e-12)
    for bar in range(4):  # each bar's middle is inside the frame
        points = mesh.points[8 * bar : 8 * bar + 8]
        middle = points.mean(axis=0)
        assert gate.signed_distance(middle) < 0.0
        bar_mesh = Mesh(points, mesh.faces[: 6 * 5])
        assert_outwards(bar_mesh, middle)
    # The bars reach the frame's eight outer corners (no gap at the corners).
    t, w, h = (
        gate.thickness,
        gate.width / 2 + gate.thickness,
        gate.height / 2 + gate.thickness,
    )
    local = mesh.points - gate.center
    c, s = np.cos(gate.yaw), np.sin(gate.yaw)
    nuv = np.stack(
        [
            c * local[:, 0] + s * local[:, 1],
            -s * local[:, 0] + c * local[:, 1],
            local[:, 2],
        ],
        axis=1,
    )
    for corner in [
        (a * t / 2, b * w, d * h) for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)
    ]:
        assert np.min(np.linalg.norm(nuv - corner, axis=1)) < 1e-12
    # The opening (its plane, every 5 cm, edges excluded) is clear of the bars.
    u, v = np.meshgrid(
        np.linspace(-0.98, 0.98, 33) * gate.width / 2,
        np.linspace(-0.98, 0.98, 29) * gate.height / 2,
    )
    c, s = np.cos(gate.yaw), np.sin(gate.yaw)
    opening = gate.center + np.stack([-s * u, c * u, v], axis=-1).reshape(-1, 3)
    assert np.all(gate.signed_distance(opening) > 0.0)


def test_a_round_frame_is_on_the_torus_with_faces_outwards():
    gate = COURSE.gates[3]  # round, turned -40 degrees
    assert isinstance(gate, CircularGate)
    mesh = circular_frame(gate, around=40, across=10)
    assert mesh.points.shape == (400, 3)
    np.testing.assert_allclose(gate.signed_distance(mesh.points), 0.0, atol=1e-12)
    # Outwards: away from the tube's centre line (the nearest point of the circle
    # of radius opening + thickness / 2).
    big = gate.radius + gate.thickness / 2
    for centroid, normal in face_normals(mesh):
        local = gate.to_local(centroid)
        uv = local[1:3] / np.hypot(*local[1:3]) * big
        c, s = np.cos(gate.yaw), np.sin(gate.yaw)
        nearest = gate.center + np.array([-s * uv[0], c * uv[0], uv[1]])
        assert np.dot(normal, centroid - nearest) > 0.0


def test_the_room_and_its_floor_grid():
    room = COURSE.room  # 18 x 10 x 4 m
    edges = room_edges(room)
    assert len(cell_lists(edges.lines)) == 12
    np.testing.assert_allclose(np.min(edges.points, axis=0), [0.0, 0.0, -4.0])
    np.testing.assert_allclose(np.max(edges.points, axis=0), [18.0, 10.0, 0.0])
    lengths = [
        np.linalg.norm(edges.points[a] - edges.points[b])
        for a, b in cell_lists(edges.lines)
    ]
    assert sorted(lengths) == pytest.approx([4.0] * 4 + [10.0] * 4 + [18.0] * 4)
    grid = floor_grid(room, spacing=1.0)
    assert len(cell_lists(grid.lines)) == 19 + 11  # every metre, both edges
    assert np.all(grid.points[:, 2] == 0.0)
    assert len(cell_lists(floor_grid(room, spacing=4.0).lines)) == 6 + 4
    with pytest.raises(ValueError):
        floor_grid(room, spacing=0.0)


def test_every_part_of_the_first_course():
    meshes = course_meshes(COURSE)
    assert len(meshes.gates) == 4 and len(meshes.obstacles) == 3
    np.testing.assert_array_equal(meshes.start, [1.5, 5.0, 0.0])
    for mesh, shape in zip(meshes.obstacles, COURSE.obstacles, strict=True):
        np.testing.assert_allclose(shape.signed_distance(mesh.points), 0.0, atol=1e-12)


def test_merge_moves_the_indices_and_bad_cells_are_rejected():
    a = path_lines([[0, 0, 0], [1, 0, 0]])
    b = block([0, 0, 0], [1, 1, 1])
    both = merge([a, b])
    assert len(both.points) == 10
    assert cell_lists(both.lines) == [[0, 1]]
    assert min(min(c) for c in cell_lists(both.faces)) == 2
    assert len(path_lines([[0, 0, 0]]).lines) == 0
    assert len(merge([]).points) == 0
    with pytest.raises(ValueError, match="not there"):
        Mesh(np.zeros((3, 3)), faces=[3, 0, 1, 3])
    with pytest.raises(ValueError):
        Mesh(np.zeros((3, 2)))


# ---- the drone ---------------------------------------------------------------------


def test_the_drone_has_its_rotors_where_the_profile_has_them():
    meshes = drone_meshes(SHAPE)
    ccw, cw = cell_lists(meshes.discs_ccw.faces), cell_lists(meshes.discs_cw.faces)
    assert len(ccw) == sum(SHAPE.spins > 0) == 2 and len(cw) == 2
    centres = [meshes.discs_ccw.points[c].mean(axis=0) for c in ccw] + [
        meshes.discs_cw.points[c].mean(axis=0) for c in cw
    ]
    for rotor in SHAPE.rotors:
        found = [c for c in centres if np.allclose(c, rotor, atol=1e-12)]
        assert len(found) == 1
    # Every rim point a propeller radius from its rotor, in the rotor's plane.
    for discs in (meshes.discs_ccw, meshes.discs_cw):
        for cell in cell_lists(discs.faces):
            rim = discs.points[cell]
            centre = rim.mean(axis=0)
            np.testing.assert_allclose(
                np.linalg.norm(rim - centre, axis=1), SHAPE.propeller_radius, atol=1e-12
            )
            np.testing.assert_allclose(rim[:, 2], centre[2], atol=1e-15)
    # Arms reach from the centre to each rotor (in the body's plane).
    reach = np.max(np.hypot(meshes.frame.points[:, 0], meshes.frame.points[:, 1]))
    assert reach == pytest.approx(np.max(np.hypot(*IRIS_ROTORS_XY)), abs=0.02)


IRIS_ROTORS_XY = np.array([rotor.position[0:2] for rotor in IRIS.rotors]).T


def test_a_pose_places_the_body_in_ned():
    q = quaternion.from_euler(0.2, -0.3, np.radians(90.0))
    p = np.array([5.0, 4.0, -1.5])
    matrix = pose_matrix(p, q)
    body = np.vstack([SHAPE.rotors, [[1.0, 0.0, 0.0]]])
    placed = (matrix[0:3, 0:3] @ body.T).T + matrix[0:3, 3]
    expected = (quaternion.to_matrix(q) @ body.T).T + p
    np.testing.assert_allclose(placed, expected, atol=1e-15)
    np.testing.assert_array_equal(matrix[3], [0.0, 0.0, 0.0, 1.0])
    # Nose East at a heading of 90 degrees, level.
    level = pose_matrix(p, quaternion.from_euler(0.0, 0.0, np.radians(90.0)))
    np.testing.assert_allclose(
        level[0:3, 0:3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-15
    )
    # A rounded quaternion (as in a recording) still gives a rotation.
    rounded = pose_matrix(p, np.round(q, 3))
    np.testing.assert_allclose(
        rounded[0:3, 0:3] @ rounded[0:3, 0:3].T, np.eye(3), atol=1e-12
    )


# ---- where a recorded drone is -----------------------------------------------------


def turned(yaw_deg):
    return quaternion.from_euler(0.0, 0.0, np.radians(yaw_deg))


def heading(q):
    return np.degrees(quaternion.to_euler(q)[2])


def test_slerp_turns_steadily_about_one_axis():
    q0, q1 = turned(0.0), turned(90.0)
    for fraction in (0.0, 0.25, 0.5, 0.9, 1.0):
        q = slerp(q0, q1, fraction)
        assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-15)
        assert heading(q) == pytest.approx(90.0 * fraction, abs=1e-9)
    # -q1 is the same attitude: the same way round, not 270 degrees back.
    np.testing.assert_allclose(slerp(q0, -q1, 0.5), slerp(q0, q1, 0.5), atol=1e-15)
    # A tilt and a turn together: half-way is half the rotation between them.
    a = quaternion.from_euler(0.3, 0.1, 0.0)
    b = quaternion.from_euler(-0.2, 0.4, 1.0)
    half = slerp(a, b, 0.5)
    step = quaternion.multiply(quaternion.conjugate(a), half)
    whole = quaternion.multiply(quaternion.conjugate(a), b)
    np.testing.assert_allclose(
        2.0 * quaternion.to_rotation_vector(step),
        quaternion.to_rotation_vector(whole),
        atol=1e-12,
    )
    near = slerp(q0, turned(1e-3), 0.5)  # almost equal: still a unit quaternion
    assert np.linalg.norm(near) == pytest.approx(1.0, abs=1e-15)
    assert heading(near) == pytest.approx(5e-4, abs=1e-9)


def three_samples():
    return lap(
        times=[0.0, 0.02, 0.04],
        positions=[[0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [0.2, 0.4, -1.2]],
        attitudes=[turned(0.0), turned(10.0), turned(30.0)],
    )


def test_a_pose_between_samples():
    flight = three_samples()
    for k in range(3):  # at a sample, the sample
        position, attitude = pose_at(flight, flight.times[k])
        np.testing.assert_allclose(position, flight.positions[k], atol=1e-15)
        assert heading(attitude) == pytest.approx([0.0, 10.0, 30.0][k], abs=1e-9)
    position, attitude = pose_at(flight, 0.03)
    np.testing.assert_allclose(position, [0.2, 0.2, -1.1], atol=1e-12)
    assert heading(attitude) == pytest.approx(20.0, abs=1e-9)
    position, attitude = pose_at(flight, 0.005)
    np.testing.assert_allclose(position, [0.05, 0.0, -1.0], atol=1e-12)
    assert heading(attitude) == pytest.approx(2.5, abs=1e-9)
    # Before the start and after the end: where it stood and where it ended.
    np.testing.assert_allclose(pose_at(flight, -1.0)[0], flight.positions[0])
    np.testing.assert_allclose(pose_at(flight, 9.0)[0], flight.positions[-1])
    # Rounded attitudes come back as unit quaternions.
    rough = dataclasses.replace(flight, attitudes=np.round(flight.attitudes, 2))
    assert np.linalg.norm(pose_at(rough, 0.01)[1]) == pytest.approx(1.0, abs=1e-15)


def test_the_path_flown_so_far():
    flight = three_samples()
    np.testing.assert_allclose(path_until(flight, 0.0), [[0.0, 0.0, -1.0]])
    np.testing.assert_allclose(
        path_until(flight, 0.03), [[0.0, 0.0, -1.0], [0.2, 0.0, -1.0], [0.2, 0.2, -1.1]]
    )
    np.testing.assert_allclose(path_until(flight, 0.04), flight.positions)
    np.testing.assert_allclose(path_until(flight, 5.0), flight.positions)


# ---- a training's folder read as it grows ------------------------------------------


def test_a_run_read_as_it_grows(tmp_path):
    run = Run(tmp_path / "not yet")
    assert run.poll() == [] and len(run.log()) == 0
    short_training(tmp_path / "run")
    run = Run(tmp_path / "run")
    first = run.poll()
    assert [r.steps for r in first] == [256, 512, 768]
    assert run.poll() == []  # nothing new
    # A new recording, written while the viewer watches.
    agent = Agent.load(tmp_path / "run" / "agents" / "000000000512.npz")
    agent.info = {**agent.info, "steps": 640}
    save_recording(tmp_path / "run", agent, (SHORT,), first[0].laps, SHAPE)
    new = run.poll()
    assert [r.steps for r in new] == [640]
    assert [r.steps for r in run.recordings] == [256, 512, 640, 768]
    log = run.log()
    np.testing.assert_array_equal(log.steps, [256, 512, 768])
    assert np.all((log.finished >= 0) & (log.crashed <= 1))
    np.testing.assert_allclose(log.finished + log.crashed + log.timeout, 1.0)


def test_a_broken_file_is_reported_once_and_skipped(tmp_path):
    short_training(tmp_path)
    files = recording_files(tmp_path / LAPS_FOLDER)
    files[1].write_text("{ not json", encoding="utf-8")
    run = Run(tmp_path)
    assert [r.steps for r in run.poll()] == [256, 768]
    assert list(run.errors) == [files[1]] and "not json" not in run.errors[files[1]]
    assert run.poll() == []  # not read again


def test_a_row_being_written_is_left_out(tmp_path):
    short_training(tmp_path)
    log = tmp_path / "log.csv"
    text = log.read_text(encoding="ascii")
    log.write_text(text + "4,1024,", encoding="ascii")  # cut while written
    assert [r["steps"] for r in read_log(log)] == [256.0, 512.0, 768.0]
    assert read_log(tmp_path / "nothing.csv") == []


def test_two_stages_on_one_axis_of_steps(tmp_path):
    settings = PPOSettings(rollout_steps=16, minibatch_size=64, epochs=1)
    train(
        SHORT, 2 * 16 * 4, tmp_path / "fly", n=4, settings=settings,
        evaluate_every=1, margin=0.0,
    )  # fmt: skip
    first = Agent.load(tmp_path / "fly" / "last.npz")  # trained 128 steps
    train(
        SHORT, 2 * 16 * 4, tmp_path / "margin", n=4, settings=settings,
        evaluate_every=1, agent=first,
    )  # fmt: skip
    run = Run(tmp_path)
    run.poll()
    assert [r.steps for r in run.recordings] == [64, 128, 192, 256]
    assert [run.stage(i).name for i in range(4)] == ["fly", "fly", "margin", "margin"]
    # The second stage's log counted its own steps (64, 128): it is moved by the
    # 128 its agent brought.
    np.testing.assert_array_equal(run.log().steps, [64, 128, 192, 256])
    # A replay folder holds only laps (and agents); a folder of recordings too.
    replay = tmp_path / "replay"
    shutil.copytree(tmp_path / "fly" / LAPS_FOLDER, replay / LAPS_FOLDER)
    assert len(Run(replay).poll()) == 2 and len(Run(replay).log()) == 0
    assert len(Run(replay / LAPS_FOLDER).poll()) == 2


def test_what_can_be_averaged_over_a_recording():
    laps = (
        lap(status="finished", lap_time=3.0, label="a"),
        lap(status="finished", lap_time=4.0, label="b"),
        lap(status="crashed", label="c"),
        lap(status="timeout", label="d", course=1),
    )
    second = dataclasses.replace(COURSE, name="Second")
    recording = Recording(0, (COURSE, second), SHAPE, laps)
    assert lap_numbers(recording) == {
        "laps": 4, "finished": 0.5, "crashed": 0.25, "lap_time": 3.5,
    }  # fmt: skip
    one = lap_numbers(recording, course=1)
    assert one["laps"] == 1 and one["finished"] == 0.0 and np.isnan(one["lap_time"])
    assert lap_numbers(recording, course=5)["laps"] == 0
    assert isinstance(laps[0], Lap)


def test_what_the_recordings_say_about_learning():
    from tribuna.run import progress

    def rec(steps, *laps):
        return Recording(steps, (COURSE,), SHAPE, laps)

    recordings = [
        rec(100, lap(status="crashed", label="a"), lap(label="b")),
        rec(200, lap(status="finished", lap_time=5.0, label="a"), lap(label="b")),
        rec(300, lap(status="finished", lap_time=4.0, label="a"),
            lap(status="finished", lap_time=4.5, label="b")),
    ]  # fmt: skip
    numbers = progress(recordings)
    np.testing.assert_array_equal(numbers.steps, [100, 200, 300])
    np.testing.assert_array_equal(numbers.finished, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(numbers.lap_time[1:], [5.0, 4.25])
    assert np.isnan(numbers.lap_time[0]) and np.isnan(numbers.first_lap_time[0])
    np.testing.assert_array_equal(numbers.first_lap_time[1:], [5.0, 4.0])
    assert len(progress(recordings, course=1).steps) == 0
    # On a set of courses, each course's own first lap.
    second = dataclasses.replace(COURSE, name="Second")
    both = Recording(
        400, (COURSE, second), SHAPE,
        (lap(status="finished", lap_time=7.0, label="x", course=1),
         lap(status="finished", lap_time=6.0, label="y", course=0)),
    )  # fmt: skip
    np.testing.assert_array_equal(progress([both], course=0).first_lap_time, [6.0])
    np.testing.assert_array_equal(progress([both], course=1).first_lap_time, [7.0])
