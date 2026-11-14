"""Tests for the laps a training records (phase 13): every evaluation's laps
written with the agent that flew them, the same training with and without
recording, and the laps kept by each kind of evaluation."""

import csv

import numpy as np
import pytest

from jineta.doma.policy import Agent
from jineta.doma.ppo import PPOSettings
from jineta.picadero.courses import CourseRanges
from jineta.picadero.env import CourseEnv
from jineta.picadero.realism import RANDOMISED_LAB
from jineta.picadero.recording import read_recording, recording_files
from jineta.picadero.training import (
    AGENTS_FOLDER,
    LAPS_FOLDER,
    RECORDED_LAPS,
    evaluate_courses,
    evaluate_drawn,
    fly,
    train,
    train_random,
)
from tests.test_recording import COURSE, POLICY, assert_same_course
from tests.test_training import SHORT

SETTINGS = PPOSettings(rollout_steps=32, minibatch_size=128, epochs=2)


def short_training(out, record=True):
    rows = []
    train(
        SHORT, 3 * 32 * 8, out, n=8, seed=2, settings=SETTINGS, evaluate_every=1,
        margin=0.0, report=rows.append, record=record,
    )  # fmt: skip
    return rows


def test_a_training_records_every_evaluation_with_its_agent(tmp_path):
    short_training(tmp_path)
    names = ["000000000256", "000000000512", "000000000768"]
    files = recording_files(tmp_path / LAPS_FOLDER)
    assert [p.stem for p in files] == names
    agents = sorted(p.stem for p in (tmp_path / AGENTS_FOLDER).iterdir())
    assert agents == names
    with open(tmp_path / "log.csv", encoding="ascii") as file:
        log = list(csv.DictReader(file))
    for path, row in zip(files, log, strict=True):
        recording = read_recording(path)
        assert recording.steps == int(row["steps"]) == int(path.stem)
        assert_same_course(recording.courses[0], SHORT)
        labels = [lap.label for lap in recording.laps]
        assert labels == ["mean action"] + [f"noisy {i + 1}" for i in range(8)]
        # What the log says of the evaluation is what was recorded.
        assert recording.info["iteration"] == int(row["iteration"])
        assert recording.info["eval_status"] == row["eval_status"]
        assert recording.laps[0].status == row["eval_status"]
        assert recording.laps[0].gates == int(row["eval_gates"])
        assert recording.info["course"] == SHORT.name
        # The agent kept with the recording flies its mean lap again.
        agent = Agent.load(tmp_path / AGENTS_FOLDER / f"{path.stem}.npz")
        assert agent.info["steps"] == recording.steps
        again = fly(agent, CourseEnv(SHORT, n=1), record=True)["flights"][0]
        lap = recording.laps[0]
        assert len(again["times"]) == len(lap.times)
        np.testing.assert_allclose(again["states"][:, 0:3], lap.positions, atol=5e-5)
        np.testing.assert_allclose(again["states"][:, 6:10], lap.attitudes, atol=5e-6)


def test_recording_does_not_change_the_training(tmp_path):
    recorded = short_training(tmp_path / "a")
    plain = short_training(tmp_path / "b", record=False)
    assert not (tmp_path / "b" / LAPS_FOLDER).exists()
    assert not (tmp_path / "b" / AGENTS_FOLDER).exists()
    for a, b in zip(recorded, plain, strict=True):
        assert set(a) == set(b)
        for key in set(a) - {"seconds"}:
            assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))
    for name in ("best.npz", "last.npz"):
        one, other = (
            Agent.load(tmp_path / "a" / name),
            Agent.load(tmp_path / "b" / name),
        )
        observations = np.random.default_rng(0).normal(size=(5, 43))
        np.testing.assert_array_equal(one.act(observations), other.act(observations))


def test_drawn_drones_are_recorded_as_the_evaluation_flew_them():
    agent = Agent.load(POLICY)
    n = 12
    plain = evaluate_drawn(agent, COURSE, n, seed=4, realism=RANDOMISED_LAB)
    result = evaluate_drawn(agent, COURSE, n, 4, True, realism=RANDOMISED_LAB)
    assert result.success_rate == plain.success_rate
    np.testing.assert_array_equal(result.gates, plain.gates)
    assert len(result.laps) == RECORDED_LAPS < n
    for i, lap in enumerate(result.laps):
        assert lap.label == f"drawn drone {i + 1}"
        assert lap.gates == result.gates[i]
        # Each drawn drone's closest approach is its own: the lap is drone i's.
        assert lap.clearance == result.clearances[i]
        assert np.isnan(lap.lap_time) == np.isnan(result.lap_times[i])
        assert (lap.status == "finished") == np.isfinite(result.lap_times[i])


def test_one_lap_per_course_on_a_set_of_courses():
    agent = Agent.load(POLICY)
    shorter = SHORT.__class__(
        **{**SHORT.__dict__, "name": "Shorter", "time_limit": 2.0}
    )
    result = evaluate_courses(agent, [SHORT, shorter], record=True)
    assert [lap.course for lap in result.laps] == [0, 1]
    assert [lap.label for lap in result.laps] == ["First course", "Shorter"]
    assert result.laps[1].duration <= 2.0 + 0.02
    assert list(result.gates) == [lap.gates for lap in result.laps]


def test_a_training_on_random_courses_records_every_validation_course(tmp_path):
    settings = PPOSettings(rollout_steps=16, minibatch_size=64, epochs=1)
    train_random(
        2 * 16 * 8, tmp_path, k=2, n=4, renew_every=10, settings=settings,
        ranges=CourseRanges(time_limit=3.0), validation=3, evaluate_every=1,
    )  # fmt: skip
    files = recording_files(tmp_path / LAPS_FOLDER)
    assert len(files) == 2
    recording = read_recording(files[-1])
    assert len(recording.courses) == 3
    assert [lap.course for lap in recording.laps] == [0, 1, 2]
    assert recording.info["course"] == "random" and recording.info["nearest"] == 4
    assert [lap.label for lap in recording.laps] == [c.name for c in recording.courses]


def test_the_iteration_and_agent_info_are_plain_json(tmp_path):
    short_training(tmp_path)
    recording = read_recording(recording_files(tmp_path / LAPS_FOLDER)[0])
    for key in ("iteration", "steps", "seed", "drones", "realism", "eval_status"):
        assert key in recording.info
    assert recording.info["realism"] == "ideal"
    lap_time = recording.info["eval_lap_time"]
    assert lap_time is None or lap_time == pytest.approx(recording.laps[0].lap_time)


def test_an_agent_flown_again_and_recorded(tmp_path, capsys):
    from examples.record_agent import default_out, main

    path = main([str(POLICY), "--out", str(tmp_path / "replay")])
    assert path == tmp_path / "replay" / LAPS_FOLDER / "000017694720.json"
    assert (tmp_path / "replay" / AGENTS_FOLDER / "000017694720.npz").is_file()
    recording = read_recording(path)
    assert recording.info["replay"] is True and recording.info["flown_in"] == "ideal"
    mean = recording.laps[0]
    assert (mean.label, mean.status) == ("mean action", "finished")
    assert mean.lap_time == pytest.approx(3.235, abs=0.05)
    assert len(recording.laps) == 1 + RECORDED_LAPS
    assert "mean action: finished 3.2" in capsys.readouterr().out
    # Where a replay goes by default.
    runs = tmp_path / "runs" / "realism" / "margin"
    assert default_out(runs / "best.npz").name == "margin_best"
    assert default_out(runs / "agents" / "000000163840.npz").name == (
        "margin_000000163840"
    )
    assert default_out(POLICY).name == "first_course"


def test_the_realism_of_an_agent_from_its_label():
    from jineta.picadero.finish import StopAfterFinish
    from jineta.picadero.realism import IDEAL, LAB
    from jineta.picadero.training import realism_from_label, realism_label

    for realism in (IDEAL, LAB, RANDOMISED_LAB):
        for stop in (None, StopAfterFinish()):
            assert realism_from_label(realism_label(realism, stop)) is realism
