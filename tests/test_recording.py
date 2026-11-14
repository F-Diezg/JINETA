"""Tests for recorded flights (phase 13): a course written as a file's contents
and read back, every drone's lap recorded by `fly` (times, states, gate passes,
where it touched something), and recordings written whole and read back."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.iris_course import COURSE_FILE
from jineta.campo.course import course_from_dict, course_to_dict, load_course
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.campo.indoor import Box
from jineta.doma.policy import Agent
from jineta.escuadron.referee import CRASHED, FINISHED
from jineta.jinete.profiles import IRIS_HITBOX, MULTIROTOR_PROFILES
from jineta.picadero import recording as recording_module
from jineta.picadero.courses import random_course
from jineta.picadero.env import CourseEnv
from jineta.picadero.recording import (
    DroneShape,
    Lap,
    Recording,
    drone_shape,
    laps_from_flight,
    read_recording,
    recording_files,
    write_recording,
)
from jineta.picadero.training import fly
from tests.test_training import ConstantAgent

COURSE = load_course(COURSE_FILE)
POLICY = Path(__file__).resolve().parents[1] / "policies" / "first_course.npz"


def assert_same_course(a, b):
    """Two courses equal: every number, the yaws to rounding (degrees and back)."""
    assert a.name == b.name and a.time_limit == b.time_limit
    np.testing.assert_array_equal(a.room.size, b.room.size)
    np.testing.assert_array_equal(a.start, b.start)
    assert a.start_yaw == pytest.approx(b.start_yaw, abs=1e-12)
    assert len(a.gates) == len(b.gates) and len(a.obstacles) == len(b.obstacles)
    for g, h in zip(a.gates, b.gates, strict=True):
        assert type(g) is type(h)
        np.testing.assert_array_equal(g.center, h.center)
        assert g.yaw == pytest.approx(h.yaw, abs=1e-12)
        assert g.thickness == h.thickness
        if isinstance(g, RectangularGate):
            assert (g.width, g.height) == (h.width, h.height)
        else:
            assert isinstance(g, CircularGate) and g.radius == h.radius
    for o, p in zip(a.obstacles, b.obstacles, strict=True):
        assert type(o) is type(p)
        if isinstance(o, Box):
            np.testing.assert_array_equal(o.center, p.center)
            np.testing.assert_array_equal(o.size, p.size)
            assert o.yaw == pytest.approx(p.yaw, abs=1e-12)
        else:
            np.testing.assert_array_equal(o.base, p.base)
            assert (o.radius, o.height) == (p.radius, p.height)


# ---- a course as a file's contents ------------------------------------------------


def test_the_first_course_written_and_read_back(tmp_path):
    data = course_to_dict(COURSE)
    assert_same_course(course_from_dict(data), COURSE)
    # As a YAML file, like the ones in courses/ (degrees, NED, every key).
    path = tmp_path / "copy.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert_same_course(load_course(path), COURSE)
    assert data["gates"][2]["yaw_deg"] == pytest.approx(30.0, abs=1e-12)
    assert data["obstacles"][1]["yaw_deg"] == pytest.approx(20.0, abs=1e-12)


def test_random_courses_written_and_read_back():
    rng = np.random.default_rng(13)
    for i in range(30):
        course = random_course(rng, name=f"Random {i}")
        data = json.loads(json.dumps(course_to_dict(course)))
        assert_same_course(course_from_dict(data), course)


# ---- every drone's lap recorded by fly --------------------------------------------


def gate_side(gate, point):
    """Where a point is along a gate's heading: < 0 before it, >= 0 after."""
    return float(gate.to_local(point)[0])


def test_the_phase_11_lap_is_recorded_whole():
    agent = Agent.load(POLICY)
    env = CourseEnv(COURSE, n=1)
    result = fly(agent, env, record=True)
    assert result["status"][0] == FINISHED
    flight = result["flights"][0]
    # The same states as drone 0's record, with a lap clock from 0 every
    # environment step (0.02 s), the last one the step that ended the lap.
    np.testing.assert_array_equal(flight["states"], result["states"])
    times = flight["times"]
    k = len(times)
    np.testing.assert_allclose(times, 0.02 * np.arange(k), atol=1e-12)
    lap_time = result["lap_time"][0]
    assert times[-2] < lap_time <= times[-1]
    # The four gates crossed in order, the last one at the lap time; each moment
    # between the two samples where the centre of mass goes through its plane.
    passes = result["pass_times"][0]
    assert np.all(np.diff(passes) > 0.0)
    assert passes[-1] == lap_time
    positions = flight["states"][:, 0:3]
    for gate, moment in zip(COURSE.gates, passes, strict=True):
        j = int(np.searchsorted(times, moment))  # times[j - 1] < moment <= times[j]
        assert gate_side(gate, positions[j - 1]) < 0.0 <= gate_side(gate, positions[j])
    assert np.isnan(result["crash_time"][0]) and result["crash_kind"][0] == ""
    assert np.all(np.isnan(result["crash_point"][0]))


def test_a_crash_into_the_ceiling_is_recorded_where_it_touched():
    env = CourseEnv(COURSE, n=1)
    result = fly(ConstantAgent([1.0, 0.0, 0.0, 0.0]), env, record=True)
    assert result["status"][0] == CRASHED
    assert result["crash_kind"][0] == "room"
    # Level all the way up: the centre of mass is where the hitbox's top reaches
    # the ceiling (4 m), between the last two samples.
    up = IRIS_HITBOX.extent([0.0, 0.0, -1.0])
    point = result["crash_point"][0]
    assert point[2] == pytest.approx(-COURSE.room.size[2] + up, abs=1e-5)
    np.testing.assert_allclose(point[0:2], COURSE.start, atol=1e-6)
    flight = result["flights"][0]
    times, heights = flight["times"], -flight["states"][:, 2]
    assert times[-2] < result["crash_time"][0] <= times[-1]
    assert heights[-2] < -point[2] <= heights[-1]
    assert np.all(np.isnan(result["pass_times"][0]))


def test_every_drone_of_a_batch_is_recorded_to_its_own_end():
    agent = Agent.load(POLICY)
    n = 6
    env = CourseEnv(COURSE, n=n)
    result = fly(agent, env, np.random.default_rng(3), record=True)
    lengths = [len(f["times"]) for f in result["flights"]]
    for i, flight in enumerate(result["flights"]):
        np.testing.assert_array_equal(flight["states"][0], env.start)
        assert flight["times"][0] == 0.0
        if result["status"][i] == FINISHED:
            assert flight["times"][-2] < result["lap_time"][i] <= flight["times"][-1]
            assert result["pass_times"][i][-1] == result["lap_time"][i]
    # The drones of a batch fly as if alone: the same commands give the same
    # tracks bit for bit. A constant agent gives them: the network's products
    # can round differently from row to row on another processor (phase 11's
    # decision log), so the policy's mean action is not compared bit for bit.
    commands = ConstantAgent([0.8, 0.1, -0.05, 0.0])
    same = fly(commands, CourseEnv(COURSE, n=3), record=True)
    for flight in same["flights"]:
        np.testing.assert_array_equal(flight["states"], same["flights"][0]["states"])
    assert len(set(lengths)) > 1  # noisy laps end at different moments


# ---- recordings as files ----------------------------------------------------------


def a_recording():
    agent = Agent.load(POLICY)
    result = fly(agent, CourseEnv(COURSE, n=1), record=True)
    finished = laps_from_flight(result, [0], ["mean action"])
    crash = fly(
        ConstantAgent([1.0, 0.0, 0.0, 0.0]), CourseEnv(COURSE, n=1), record=True
    )
    crashed = laps_from_flight(crash, [0], ["straight up"])
    return Recording(
        steps=17_694_720,
        courses=(COURSE,),
        shape=drone_shape(MULTIROTOR_PROFILES["iris"]),
        laps=(*finished, *crashed),
        info={"iteration": 270, "realism": "ideal"},
    )


def test_laps_from_a_flight():
    recording = a_recording()
    lap, crash = recording.laps
    assert (lap.label, lap.status, lap.gates) == ("mean action", "finished", 4)
    assert lap.lap_time == pytest.approx(3.235, abs=0.05)
    np.testing.assert_allclose(np.linalg.norm(lap.attitudes, axis=1), 1.0, atol=1e-12)
    assert lap.positions[0, 2] < 0.0 and lap.duration == lap.times[-1]
    assert (crash.status, crash.crash_kind, crash.gates) == ("crashed", "room", 0)
    assert np.isnan(crash.lap_time)
    shape = recording.shape
    assert shape.rotors.shape == (4, 3) and sorted(shape.spins) == [-1, -1, 1, 1]
    assert shape.propeller_radius == pytest.approx(0.128)
    assert len(shape.hitbox_radii) == 13


def test_a_recording_written_and_read_back(tmp_path):
    recording = a_recording()
    path = write_recording(tmp_path / "laps", "000017694720", recording)
    assert path.name == "000017694720.json"
    assert recording_files(tmp_path / "laps") == [path]
    # Small: a few tens of kB for two laps.
    assert path.stat().st_size < 40_000
    back = read_recording(path)
    assert back.steps == recording.steps and back.info == recording.info
    assert_same_course(back.courses[0], COURSE)
    np.testing.assert_array_equal(back.shape.spins, recording.shape.spins)
    np.testing.assert_allclose(back.shape.rotors, recording.shape.rotors, atol=1e-12)
    for got, lap in zip(back.laps, recording.laps, strict=True):
        assert (got.label, got.status, got.gates, got.crash_kind) == (
            lap.label,
            lap.status,
            lap.gates,
            lap.crash_kind,
        )
        # Rounded for the eye: 0.1 mm, 1e-5, 0.1 ms.
        np.testing.assert_allclose(got.times, lap.times, atol=5e-5)
        np.testing.assert_allclose(got.positions, lap.positions, atol=5e-5)
        np.testing.assert_allclose(got.attitudes, lap.attitudes, atol=5e-6)
        np.testing.assert_allclose(got.pass_times, lap.pass_times, atol=5e-5)
        np.testing.assert_allclose(got.crash_point, lap.crash_point, atol=5e-5)
        for a, b in [(got.lap_time, lap.lap_time), (got.crash_time, lap.crash_time)]:
            assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, abs=5e-5)
        assert got.clearance == pytest.approx(lap.clearance, abs=5e-5)
    # Strict JSON: no NaN or Infinity in the text (null instead).
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text, parse_constant=lambda name: pytest.fail(name))


def test_a_recording_is_written_whole_or_not_at_all(tmp_path, monkeypatch):
    recording = a_recording()

    def broken_rename(source, target):
        raise OSError("disk full")

    monkeypatch.setattr(recording_module.os, "replace", broken_rename)
    with pytest.raises(OSError):
        write_recording(tmp_path, "000000000001", recording)
    # The file being written is there, but it is not a recording.
    assert recording_files(tmp_path) == []
    assert [p.name for p in tmp_path.iterdir()] == ["000000000001.json.tmp"]
    monkeypatch.undo()
    path = write_recording(tmp_path, "000000000001", recording)
    assert recording_files(tmp_path) == [path]
    assert not (tmp_path / "000000000001.json.tmp").exists()


def test_recording_files_are_the_finished_ones_in_order(tmp_path):
    assert recording_files(tmp_path / "not yet") == []
    for name in ["000000000300.json", "000000000020.json", "000000000100.json"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "000000000400.json.tmp").write_text("{", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("", encoding="utf-8")
    (tmp_path / "sub.json").mkdir()
    names = [p.name for p in recording_files(tmp_path)]
    assert names == ["000000000020.json", "000000000100.json", "000000000300.json"]


def test_a_broken_file_names_itself(tmp_path):
    recording = a_recording()
    path = write_recording(tmp_path, "000000000001", recording)
    text = path.read_text(encoding="utf-8")
    path.write_text(text[: len(text) // 2], encoding="utf-8")  # cut in half
    with pytest.raises(ValueError, match=r"000000000001\.json"):
        read_recording(path)
    data = json.loads(text)
    for change, message in [
        ({"format": "something else"}, "not a recording"),
        ({"version": 2}, "version 2"),
    ]:
        path.write_text(json.dumps({**data, **change}), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_recording(path)
    del data["laps"][0]["times"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing key 'times'"):
        read_recording(path)


def lap(**change):
    values = {
        "course": 0,
        "label": "a lap",
        "status": "timeout",
        "lap_time": np.nan,
        "gates": 0,
        "pass_times": np.full(4, np.nan),
        "clearance": 1.0,
        "crash_time": np.nan,
        "crash_point": np.full(3, np.nan),
        "crash_kind": "",
        "times": [0.0, 0.02],
        "positions": np.zeros((2, 3)),
        "attitudes": [[1.0, 0.0, 0.0, 0.0]] * 2,
    }
    return Lap(**{**values, **change})


def test_bad_input_is_rejected():
    shape = drone_shape(MULTIROTOR_PROFILES["iris"])
    with pytest.raises(ValueError, match="status"):
        lap(status="landed")
    with pytest.raises(ValueError, match="at least one sample"):
        lap(times=[], positions=np.zeros((0, 3)), attitudes=np.zeros((0, 4)))
    with pytest.raises(ValueError, match="must match"):
        lap(positions=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="go back"):
        lap(times=[0.02, 0.0])
    with pytest.raises(ValueError, match="no course 1"):
        Recording(0, (COURSE,), shape, (lap(course=1),))
    with pytest.raises(ValueError, match="one pass time per gate"):
        Recording(0, (COURSE,), shape, (lap(pass_times=[np.nan]),))
    with pytest.raises(ValueError, match="at least one course"):
        Recording(0, (), shape, ())
    with pytest.raises(ValueError, match="steps"):
        Recording(-1, (COURSE,), shape, ())
    with pytest.raises(ValueError, match="spins"):
        DroneShape(np.zeros((2, 3)), [1, 0], 0.1, np.zeros((1, 3)), [0.1])
    with pytest.raises(ValueError, match="hitbox"):
        DroneShape(np.zeros((2, 3)), [1, -1], 0.1, np.zeros((2, 3)), [0.1])
    with pytest.raises(ValueError):
        lap().times[0] = 1.0  # read-only
