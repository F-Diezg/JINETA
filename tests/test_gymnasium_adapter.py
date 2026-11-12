"""Tests for the adapter to Gymnasium: the environment passes Gymnasium's own
checker, and the one-drone and n-drone versions fly exactly as `CourseEnv`; the
n-drone one gives what Gymnasium's own vectoriser gives.

Skipped when Gymnasium is not installed (it is an optional dependency)."""

import warnings

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from gymnasium.utils.env_checker import check_env  # noqa: E402
from gymnasium.vector import AutoresetMode, SyncVectorEnv  # noqa: E402

from examples.iris_course import COURSE_FILE  # noqa: E402
from jineta.campo.course import load_course  # noqa: E402
from jineta.escuadron.referee import STATUS_NAMES  # noqa: E402
from jineta.picadero.env import CourseEnv  # noqa: E402
from jineta.picadero.gymnasium_adapter import (  # noqa: E402
    EPISODE_KEYS,
    CourseGymEnv,
    CourseVectorEnv,
    to_box,
)
from tests.test_env import script  # noqa: E402

COURSE = load_course(COURSE_FILE)


def scripts(drones: int, steps: int) -> np.ndarray:
    """Actions (steps, drones, 4) of the environment tests' scripted episodes."""
    return np.stack([script(i, steps) for i in range(drones)], axis=1)


def test_passes_gymnasiums_environment_checker():
    env = CourseGymEnv(COURSE)
    with warnings.catch_warnings():
        # Expected: the observation space has no limits (infinite bounds) and the
        # environment has no registered spec to try render modes with.
        warnings.simplefilter("ignore")
        check_env(env)


def test_spaces():
    env = CourseVectorEnv(COURSE, n=3)
    inner = env.course_env
    single = CourseGymEnv(COURSE)
    for box, space in (
        (env.single_observation_space, inner.observation_space),
        (env.single_action_space, inner.action_space),
        (single.observation_space, inner.observation_space),
        (single.action_space, inner.action_space),
    ):
        assert isinstance(box, gym.spaces.Box)
        assert box.dtype == np.float32
        np.testing.assert_array_equal(box.low, space.low)
        np.testing.assert_array_equal(box.high, space.high)
    assert env.observation_space.shape == (3, inner.observer.size)
    assert env.action_space.shape == (3, 4)
    assert env.metadata["autoreset_mode"] == AutoresetMode.SAME_STEP
    assert to_box(inner.action_space) == gym.spaces.Box(-1.0, 1.0, (4,), np.float32)


def test_one_drone_flies_as_the_environment():
    actions = scripts(1, 400)[:, 0]
    env = CourseGymEnv(COURSE)
    reference = CourseEnv(COURSE, n=1)
    observation, info = env.reset(seed=3)
    expected, _ = reference.reset(seed=3)
    np.testing.assert_array_equal(observation, expected[0])
    assert observation.shape == (reference.observer.size,) and info == {}
    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        expected, rewards, ended, cut, details = reference.step(action[None])
        assert isinstance(reward, float) and reward == rewards[0]
        assert (terminated, truncated) == (bool(ended[0]), bool(cut[0]))
        if terminated or truncated:
            # The last observation of the episode, not the first of the next.
            np.testing.assert_array_equal(observation, details["final_observation"][0])
            assert info["status"] == "crashed"
            assert info["episode_return"] == details["episode_return"][0]
            break
        np.testing.assert_array_equal(observation, expected[0])
        assert observation.dtype == np.float32
        assert info["status"] == "flying"
    else:
        pytest.fail("the scripted episode should end in a crash")


def test_many_drones_fly_as_the_environment():
    actions = scripts(4, 300)
    env = CourseVectorEnv(COURSE, n=4)
    reference = CourseEnv(COURSE, n=4)
    observations, _ = env.reset(seed=1)
    np.testing.assert_array_equal(observations, reference.reset(seed=1)[0])
    ended = 0
    for action in actions:
        observations, rewards, terminated, truncated, info = env.step(action)
        expected, expected_rewards, ended_, cut, details = reference.step(action)
        np.testing.assert_array_equal(observations, expected)
        np.testing.assert_array_equal(rewards, expected_rewards)
        np.testing.assert_array_equal(terminated, ended_)
        np.testing.assert_array_equal(truncated, cut)
        done = terminated | truncated
        if not np.any(done):
            assert info == {}
            continue
        ended += int(np.sum(done))
        np.testing.assert_array_equal(info["_final_obs"], done)
        for i in range(4):
            if done[i]:
                np.testing.assert_array_equal(
                    info["final_obs"][i], details["final_observation"][i]
                )
                assert (
                    info["final_info"]["status"][i]
                    == STATUS_NAMES[details["status"][i]]
                )
            else:
                assert info["final_obs"][i] is None
    assert ended >= 4  # every scripted drone crashed at least once


def test_gives_what_gymnasiums_vectoriser_gives():
    """CourseVectorEnv against Gymnasium's SyncVectorEnv running n copies of
    CourseGymEnv with the same-step autoreset: the same observations, rewards,
    ends, final observations and final infos (with their masks)."""
    n = 3
    actions = scripts(n, 250)
    ours = CourseVectorEnv(COURSE, n=n)
    theirs = SyncVectorEnv(
        [lambda: CourseGymEnv(COURSE)] * n, autoreset_mode=AutoresetMode.SAME_STEP
    )
    np.testing.assert_array_equal(ours.reset(seed=0)[0], theirs.reset(seed=0)[0])
    restarts = 0
    for action in actions:
        mine = ours.step(action)
        gym_out = theirs.step(action)
        for a, b in zip(mine[:4], gym_out[:4], strict=True):
            np.testing.assert_array_equal(a, b)
        info, gym_info = mine[4], gym_out[4]
        assert ("final_obs" in info) == ("final_obs" in gym_info)
        if "final_obs" not in info:
            continue
        restarts += 1
        np.testing.assert_array_equal(info["_final_obs"], gym_info["_final_obs"])
        np.testing.assert_array_equal(info["_final_info"], gym_info["_final_info"])
        for i in range(n):
            if info["_final_obs"][i]:
                np.testing.assert_array_equal(
                    info["final_obs"][i], gym_info["final_obs"][i]
                )
        assert set(info["final_info"]) == set(gym_info["final_info"])
        for key in EPISODE_KEYS:
            if key not in info["final_info"]:
                continue
            np.testing.assert_array_equal(
                info["final_info"][f"_{key}"], gym_info["final_info"][f"_{key}"]
            )
            mask = info["final_info"][f"_{key}"]
            np.testing.assert_array_equal(
                info["final_info"][key][mask], gym_info["final_info"][key][mask]
            )
    assert restarts >= 2


def test_a_finished_lap_gives_its_lap_time():
    """The classical baseline flown through the one-drone environment's actions
    (its loops at 50 Hz) finishes: the info has the lap time, as CourseEnv's."""
    from jineta.escuadron.control import (
        BatchAttitudeController,
        BatchVelocityController,
    )
    from jineta.jinete.profiles import IRIS, IRIS_GAINS
    from jineta.picadero.baseline import PathGuidance
    from tests.test_baseline import BEST, BEST_LAP

    env = CourseGymEnv(COURSE)
    env.reset(seed=0)
    inner = env.course_env
    guidance = PathGuidance(COURSE, BEST, 1)
    velocity_loop = BatchVelocityController(IRIS, IRIS_GAINS.velocity, 0.02, 1)
    attitude_loop = BatchAttitudeController(IRIS_GAINS.attitude)
    for _ in range(2000):
        states = inner.flight.states
        velocity, yaw = guidance.commands(states)
        attitude, thrust = velocity_loop.update(velocity, yaw, states)
        rates = attitude_loop.update(attitude, states)
        action = inner.limits.actions(thrust, rates)[0]
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert info["status"] == "finished"
    assert info["lap_time"] == pytest.approx(BEST_LAP, abs=0.2)
