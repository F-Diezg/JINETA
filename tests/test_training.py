"""Tests for training and measuring a policy on a course: a lap flown by an agent
(statuses, lap time, gates, clearance, the recorded states), the evaluation, the
order of evaluations, and a short training (its log and saved agents, repeated
by its seed)."""

import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.policy import Agent
from jineta.doma.ppo import PPOSettings
from jineta.escuadron.control import BatchAttitudeController, BatchVelocityController
from jineta.escuadron.referee import CRASHED, FINISHED, TIMEOUT
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.picadero.baseline import PathGuidance
from jineta.picadero.env import CourseEnv
from jineta.picadero.training import (
    LOG_COLUMNS,
    Evaluation,
    _better,
    evaluate,
    fly,
    train,
)
from tests.test_baseline import BEST, BEST_LAP
from tests.test_env import reference_episode

COURSE = load_course(COURSE_FILE)
SHORT = dataclasses.replace(COURSE, time_limit=3.0)


class BaselineAgent:
    """The classical baseline as an agent: it reads the states of `env`'s
    drones (not the observations) and gives the actions of its loops at 50 Hz."""

    def __init__(self, env: CourseEnv) -> None:
        self.env = env
        self.guidance = PathGuidance(COURSE, BEST, env.n)
        self.velocity = BatchVelocityController(
            IRIS, IRIS_GAINS.velocity, env.step_time, env.n
        )
        self.attitude = BatchAttitudeController(IRIS_GAINS.attitude)

    def act(self, observations, rng=None):
        states = self.env.flight.states
        velocity, yaw = self.guidance.commands(states)
        attitude, thrust = self.velocity.update(velocity, yaw, states)
        return self.env.limits.actions(thrust, self.attitude.update(attitude, states))


class ConstantAgent:
    """The same action for every drone, always."""

    def __init__(self, action) -> None:
        self.action = np.asarray(action, dtype=float)

    def act(self, observations, rng=None):
        return np.tile(self.action, (len(observations), 1))


def test_a_lap_of_the_baseline_through_fly():
    env = CourseEnv(COURSE, n=1)
    result = fly(BaselineAgent(env), env, record=True)
    assert result["status"][0] == FINISHED
    assert result["lap_time"][0] == pytest.approx(BEST_LAP, abs=0.2)
    assert result["gates"][0] == 4
    assert 0.02 < result["clearance"][0] < 0.5
    states = result["states"]
    # Recorded every environment step from the start to where the lap ended
    # (just past the last gate), not where the drone was put back.
    np.testing.assert_array_equal(states[0], env.start)
    last_gate = COURSE.gates[-1]
    assert np.linalg.norm(states[-1, 0:3] - last_gate.center) < 0.5
    assert len(states) == pytest.approx(BEST_LAP / env.step_time + 1, abs=12)


def test_laps_that_end_differently():
    """Drones standing on the floor time out with no gate and a clearance above
    the floor's (the floor does not count); drones climbing at full thrust hit
    the ceiling."""
    env = CourseEnv(COURSE, n=3)
    standing = fly(ConstantAgent([-1.0, 0.0, 0.0, 0.0]), env)
    assert np.all(standing["status"] == TIMEOUT)
    assert np.all(standing["gates"] == 0) and np.all(np.isnan(standing["lap_time"]))
    assert np.all(standing["clearance"] > 1.0)
    climbing = fly(ConstantAgent([1.0, 0.0, 0.0, 0.0]), env)
    assert np.all(climbing["status"] == CRASHED)
    assert np.all(climbing["clearance"] < 0.5)


def test_evaluate_an_agent():
    agent = Agent.create(43, 4, (8,), np.random.default_rng(0))
    result = evaluate(agent, COURSE, n=4, seed=1)
    assert result.status in ("crashed", "timeout")
    assert np.isnan(result.lap_time) and result.success_rate == 0.0
    assert len(result.lap_times) == 4 and not result.finished
    again = evaluate(agent, COURSE, n=4, seed=1)
    assert again.clearance == result.clearance and again.gates == result.gates


def test_which_evaluation_is_better():
    def ev(status, lap, gates):
        return Evaluation(status, lap, gates, 0.1, 0.0, np.nan, np.array([]))

    assert _better(ev("crashed", np.nan, 1), None)
    assert _better(ev("finished", 9.0, 4), ev("crashed", np.nan, 3))
    assert not _better(ev("crashed", np.nan, 3), ev("finished", 9.0, 4))
    assert _better(ev("finished", 7.0, 4), ev("finished", 7.5, 4))
    assert not _better(ev("finished", 7.5, 4), ev("finished", 7.0, 4))
    assert _better(ev("timeout", np.nan, 3), ev("crashed", np.nan, 2))
    # With a margin, a lap that keeps it beats a faster one that does not.
    safe = Evaluation("finished", 7.0, 4, 0.06, 0.0, np.nan, np.array([]))
    close = Evaluation("finished", 6.0, 4, 0.01, 0.0, np.nan, np.array([]))
    assert _better(safe, close, margin=0.05)
    assert not _better(safe, close, margin=0.0)
    assert _better(close, ev("crashed", np.nan, 4), margin=0.05)


def test_a_short_training_writes_its_log_and_agents(tmp_path):
    # The first course with a 3 s time limit, so that the evaluations are short.
    settings = PPOSettings(rollout_steps=32, minibatch_size=128, epochs=2)

    def short(out):
        rows = []
        agent, best = train(
            SHORT, 3 * 32 * 8, out, n=8, seed=2, settings=settings,
            evaluate_every=1, margin=0.0, report=rows.append,
        )  # fmt: skip
        return agent, best, rows

    _, best, rows = short(tmp_path / "a")
    assert len(rows) == 3
    with open(tmp_path / "a" / "log.csv", encoding="ascii") as file:
        log = list(csv.DictReader(file))
    assert tuple(log[0]) == LOG_COLUMNS and len(log) == 3
    assert [int(r["steps"]) for r in log] == [256, 512, 768]
    assert all(r["eval_status"] in ("crashed", "timeout") for r in log)
    for name in ("best.npz", "last.npz"):
        loaded = Agent.load(tmp_path / "a" / name)
        assert loaded.info["course"] == SHORT.name and loaded.info["seed"] == 2
    assert Agent.load(tmp_path / "a" / "last.npz").info["steps"] == 768
    assert isinstance(best, Evaluation)
    _, _, again = short(tmp_path / "b")
    for a, b in zip(rows, again, strict=True):  # the same numbers (nan as nan)
        assert set(a) == set(b)
        for key in set(a) - {"seconds"}:
            assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key]))


def test_the_margin_makes_a_near_pass_a_crash():
    """The baseline's lap passes a few centimetres from a frame (its clearance,
    measured without a margin): with a margin larger than that it crashes, with
    a smaller one it finishes."""
    plain = CourseEnv(COURSE, n=1)
    lap = fly(BaselineAgent(plain), plain)
    gap = lap["clearance"][0]
    assert 0.02 < gap < 0.2
    tight = CourseEnv(COURSE, n=1, margin=gap + 0.005)
    assert fly(BaselineAgent(tight), tight)["status"][0] == CRASHED
    loose = CourseEnv(COURSE, n=1, margin=gap - 0.005)
    assert fly(BaselineAgent(loose), loose)["status"][0] == FINISHED
    with pytest.raises(ValueError):
        CourseEnv(COURSE, margin=-0.01)


POLICY_FILE = Path(__file__).resolve().parent.parent / "policies" / "first_course.npz"


def test_the_trained_policy_beats_the_baseline():
    """The policy of phase 11 (policies/first_course.npz) finishes the first
    course faster than the baseline, keeping more than the 5 cm the baseline's
    robust laps keep; its actions, replayed in the reference engine (Multirotor,
    RatePilot, the course's referee), give the same lap. The lap time is a
    regression value: on another computer the network's products round
    differently in the last digits."""
    agent = Agent.load(POLICY_FILE)
    env = CourseEnv(COURSE, n=1)
    observations, _ = env.reset()
    actions = []
    clearance = np.inf
    for _ in range(1000):
        action = agent.act(observations)
        actions.append(action[0])
        observations, _, terminated, truncated, info = env.step(action)
        if terminated[0] or truncated[0]:
            break
        margins = env.flight.referee.monitor.margins(env.flight.states)
        clearance = min(clearance, margins[0, :-1].min())
    assert info["status"][0] == FINISHED
    lap = info["lap_time"][0]
    assert lap == pytest.approx(4.325, abs=0.05)
    assert lap < BEST_LAP - 2.0
    assert clearance > 0.05
    _, referee, _ = reference_episode(np.array(actions))
    assert referee.status == "finished"
    assert referee.lap_time == pytest.approx(lap, abs=1e-6)
