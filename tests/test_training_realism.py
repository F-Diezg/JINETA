"""Tests for training with realism (phase 12, step 8): an agent widened for new
inputs acts as before, the evaluation on drawn drones, the realism's label, and
a short training in the randomised lab with the stop after the finish."""

import dataclasses

import numpy as np
import pytest

from jineta.doma.policy import Agent
from jineta.doma.ppo import PPOSettings
from jineta.picadero.finish import StopAfterFinish
from jineta.picadero.realism import IDEAL, LAB, RANDOMISED_LAB, Realism
from jineta.picadero.training import (
    DrawnEvaluation,
    evaluate_drawn,
    realism_label,
    train,
)
from tests.test_env import COURSE
from tests.test_training import ConstantAgent

SHORT = dataclasses.replace(COURSE, time_limit=1.0)


def test_a_widened_agent_acts_as_before(tmp_path):
    """The phase 11 policy widened by one input: on 500 observations, whatever
    the new input says, the same actions and values to rounding; it saves and
    loads; the original is untouched."""
    agent = Agent.load("policies/first_course.npz")
    wide = agent.widened(1)
    assert wide.normalizer.size == agent.normalizer.size + 1 == 44
    assert wide.normalizer.count == agent.normalizer.count
    rng = np.random.default_rng(0)
    observations = agent.normalizer.mean + rng.normal(0.0, 1.0, (500, 43)) * np.sqrt(
        agent.normalizer.var
    )
    for flag in (0.0, 1.0, -7.0):
        extended = np.column_stack([observations, np.full(500, flag)])
        np.testing.assert_allclose(
            wide.act(extended), agent.act(observations), atol=1e-12
        )
        np.testing.assert_allclose(
            wide.values(extended), agent.values(observations), atol=1e-12
        )
    wide.save(tmp_path / "wide.npz")
    again = Agent.load(tmp_path / "wide.npz")
    np.testing.assert_array_equal(again.act(extended), wide.act(extended))
    assert agent.policy.network.weights[0].shape == (43, 64)
    with pytest.raises(ValueError):
        agent.widened(0)


def test_the_realism_label():
    assert realism_label(IDEAL) == "ideal"
    assert realism_label(LAB) == "motor lag, ground effect, sensors, latency 20 ms"
    assert realism_label(Realism(latency=0.03), StopAfterFinish()) == (
        "latency 30 ms, stop"
    )
    assert realism_label(RANDOMISED_LAB).endswith("randomised")


def test_evaluate_on_drawn_drones():
    """Standing on the floor nobody finishes; the phase 11 policy flies some of
    the drawn drones without the stop, none with it; the same seed flies the
    same drones."""
    standing = evaluate_drawn(
        ConstantAgent([-1.0, 0.0, 0.0, 0.0]), SHORT, 8, realism=RANDOMISED_LAB
    )
    assert isinstance(standing, DrawnEvaluation)
    assert standing.success_rate == 0.0 and np.isnan(standing.mean_lap_time)
    assert np.all(standing.gates == 0) and len(standing.lap_times) == 8
    agent = Agent.load("policies/first_course.npz")
    plain = evaluate_drawn(agent, COURSE, 64, seed=3, realism=RANDOMISED_LAB)
    again = evaluate_drawn(agent, COURSE, 64, seed=3, realism=RANDOMISED_LAB)
    np.testing.assert_array_equal(plain.lap_times, again.lap_times)
    assert 0.0 < plain.success_rate < 0.3


def test_a_short_training_in_the_randomised_lab(tmp_path):
    """Four drones, every effect, the stop after the finish, judged on 4 drawn
    drones: the log says how many it flew, and the agent knows what it was
    trained in."""
    rows = []
    settings = PPOSettings(rollout_steps=16, minibatch_size=64, epochs=1)
    agent, result = train(
        SHORT, 2 * 16 * 4, tmp_path, n=4, settings=settings, evaluate_every=1,
        realism=RANDOMISED_LAB, stop=StopAfterFinish(), judged_by=4,
        report=rows.append,
    )  # fmt: skip
    assert len(rows) == 2 and rows[-1]["eval_status"].endswith("/4")
    assert isinstance(result, DrawnEvaluation)
    assert agent.info["stop"] and agent.info["realism"].endswith("randomised, stop")
    assert agent.normalizer.size == 44  # the stop's flag


def test_the_baseline_through_the_interface():
    """The baseline as an agent of the environment: in the ideal simulator it
    finishes the first course in 6.76 s (phase 10: through the actions at 50 Hz),
    and a drone that restarts flies its next lap exactly as its first."""
    from jineta.picadero.baseline import BaselineAgent
    from jineta.picadero.env import CourseEnv
    from tests.test_baseline import BEST

    env = CourseEnv(COURSE)
    agent = BaselineAgent(env, BEST)
    observation, _ = env.reset()
    laps, observations = [], [[], []]
    for _ in range(800):
        observations[len(laps)].append(observation.copy())
        observation, _, terminated, truncated, info = env.step(agent.act(observation))
        if terminated[0] or truncated[0]:
            laps.append(info["lap_time"][0])
            if len(laps) == 2:
                break
    assert laps[0] == pytest.approx(6.76, abs=0.02) and laps[1] == laps[0]
    np.testing.assert_array_equal(observations[0], observations[1])
