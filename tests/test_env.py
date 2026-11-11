"""Tests for the training environment: its interface, the flight it produces
against the reference engine, each drone's episode its own, and the restarts."""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import course_from_dict, load_course
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.referee import CRASHED, FLYING, STATUS_NAMES, TIMEOUT
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import RatePilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_GAINS,
    IRIS_HITBOX,
    IRIS_LANDING_GEAR,
    IRIS_RATE_GAINS,
)
from jineta.picadero.env import CourseEnv
from jineta.picadero.reward import Reward
from jineta.picadero.spaces import Observer, autopilot_envelope

COURSE = load_course(COURSE_FILE)
LIMITS = autopilot_envelope(IRIS, IRIS_GAINS)
DT = 0.01
WEIGHT = IRIS.mass * GRAVITY


def script(drone: int, steps: int) -> np.ndarray:
    """A drone's actions (steps, 4): climb for 0.5 s, then wander near the hover
    thrust with small random rates changing every 0.2 s, until it hits something."""
    actions = np.zeros((steps, 4))
    for k in range(steps):
        rng = np.random.default_rng(1000 * drone + k // 10)
        actions[k, 0] = 0.25 if k < 25 else 0.07 + 0.05 * rng.normal()
        actions[k, 1:] = 0.15 * rng.normal(size=3)
    return actions


def reference_episode(actions: np.ndarray):
    """The same actions flown by the reference engine (Multirotor, RatePilot, the
    course's referee), each held for two steps, until the lap ends. Returns the
    states of every physics step, the referee, and the reward of every
    environment step."""
    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    pilot = RatePilot(vehicle, IRIS_RATE_GAINS, DT, landed=True)
    referee = COURSE.referee(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    targets, passes = [], []
    thrust, rates = LIMITS.commands(actions)

    def pre_step(t, y):
        k = len(targets)
        targets.append(-1 if referee.next_gate is None else referee.next_gate)
        passes.append(len(referee.passes))
        pilot.thrust_command, pilot.rate_command = thrust[k // 2], rates[k // 2]
        pilot(t, y)

    start = COURSE.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    _, states = run(
        vehicle.derivative,
        start,
        DT,
        2 * len(actions) * DT,
        stop=referee,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    passes.append(len(referee.passes))
    reward = Reward(COURSE.gates)
    steps = len(states) - 1
    physics = np.array(
        [
            reward(
                states[j : j + 1, 0:3],
                states[j + 1 : j + 2, 0:3],
                [targets[j]],
                [passes[j + 1] - passes[j]],
                [j == steps - 1 and referee.status == "crashed"],
                [True],
            )[0]
            for j in range(steps)
        ]
    )
    padded = np.zeros(2 * ((steps + 1) // 2))
    padded[:steps] = physics
    return states, referee, padded.reshape(-1, 2).sum(axis=1)


# ---- interface ---------------------------------------------------------------------


def test_reset_gives_the_start_observation_of_every_drone():
    env = CourseEnv(COURSE, n=3)
    observation, info = env.reset(seed=0)
    assert observation.shape == (3, 43) and observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert env.action_space.shape == (4,)
    assert info == {}
    expected = Observer(COURSE.gates)(
        np.tile(env.start, (3, 1)), np.zeros(3, int), np.zeros((3, 4))
    )
    np.testing.assert_array_equal(observation, expected.astype(np.float32))
    assert np.all(env.pilot.landed) and np.all(env.flight.clock == 0.0)
    assert env.step_time == pytest.approx(0.02)


def test_step_returns_one_row_per_drone():
    env = CourseEnv(COURSE, n=5)
    env.reset()
    observation, reward, terminated, truncated, info = env.step(np.zeros((5, 4)))
    assert observation.shape == (5, 43) and observation.dtype == np.float32
    assert reward.shape == terminated.shape == truncated.shape == (5,)
    assert terminated.dtype == truncated.dtype == bool
    for key in ("final_observation", "status", "lap_time", "episode_return"):
        assert len(info[key]) == 5
    np.testing.assert_array_equal(info["episode_steps"], 1)


def test_least_thrust_stays_on_the_floor_without_a_crash():
    """Action -1 on the thrust is 10 % of the maximum: the drone stays standing
    for the whole time limit and the episode is truncated, not a crash."""
    env = CourseEnv(COURSE)
    env.reset()
    action = np.array([[-1.0, 0.0, 0.0, 0.0]])
    steps = round(COURSE.time_limit / env.step_time)
    total = 0.0
    for k in range(steps):
        _, reward, terminated, truncated, info = env.step(action)
        total += reward[0]
        if k < steps - 1:
            assert not terminated[0] and not truncated[0]
    assert truncated[0] and not terminated[0]
    assert info["status"][0] == TIMEOUT and info["episode_steps"][0] == steps
    assert abs(total) < 1e-3  # standing still: no progress
    assert env.episode_steps[0] == 0  # restarted


def test_rejects_bad_input():
    env = CourseEnv(COURSE, n=2)
    env.reset()
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros(4))
    with pytest.raises(ValueError):
        CourseEnv(COURSE, n=0)
    with pytest.raises(ValueError):
        CourseEnv(COURSE, action_repeat=0)
    with pytest.raises(ValueError):
        CourseEnv(COURSE, dt=0.0)


# ---- against the reference -----------------------------------------------------------


@pytest.mark.parametrize("drone", [0, 2, 3])
def test_an_episode_is_the_reference_engine_flight(drone):
    """With n = 1, every state of the episode within 1e-9 of the reference engine
    flying the same actions, the same end, and the same rewards."""
    actions = script(drone, 400)
    states, referee, rewards = reference_episode(actions)
    env = CourseEnv(COURSE)
    env.reset()
    for k, action in enumerate(actions):
        _, reward, terminated, truncated, info = env.step(action[None])
        assert reward[0] == pytest.approx(rewards[k], abs=1e-9)
        if terminated[0] or truncated[0]:
            break
        assert np.max(np.abs(env.flight.states[0] - states[2 * (k + 1)])) < 1e-9
    assert terminated[0]
    assert STATUS_NAMES[info["status"][0]] == referee.status == "crashed"
    assert np.sum(rewards) == pytest.approx(info["episode_return"][0], abs=1e-9)
    final = Observer(COURSE.gates)(
        states[-1:], [len(referee.passes)], actions[k : k + 1]
    )
    np.testing.assert_allclose(info["final_observation"][0], final[0], atol=1e-5)
    assert info["episode_steps"][0] == k + 1 == (len(states)) // 2


def test_a_drone_gives_the_same_numbers_alone_or_in_a_batch():
    n, steps = 4, 300
    actions = np.stack([script(i, steps) for i in range(n)], axis=1)
    env = CourseEnv(COURSE, n=n)
    together = [env.reset()[0]]
    rewards, dones = [], []
    for k in range(steps):
        observation, reward, terminated, truncated, _ = env.step(actions[k])
        together.append(observation)
        rewards.append(reward)
        dones.append(terminated | truncated)
    assert np.sum(dones) >= 4  # restarts happen inside the batch
    for i in (1, 3):
        alone = CourseEnv(COURSE)
        observations = [alone.reset()[0]]
        for k in range(steps):
            observation, reward, terminated, truncated, _ = alone.step(
                actions[k, i : i + 1]
            )
            observations.append(observation)
            assert reward[0] == rewards[k][i]
            assert (terminated | truncated)[0] == dones[k][i]
        np.testing.assert_array_equal(
            np.array(observations)[:, 0], np.array(together)[:, i]
        )


def test_a_restarted_drone_flies_its_next_episode_as_a_fresh_one():
    """A drone that crashes starts again at once; its next episode is a fresh
    environment's first episode bit for bit (same actions), and the drone beside
    it is not disturbed."""
    steps = 400
    crash = script(3, steps)  # crashes after ~100 steps
    other = script(0, steps)
    env = CourseEnv(COURSE, n=2)
    env.reset()
    k_end = None
    for k in range(steps):
        observation, _, terminated, _, info = env.step(np.stack([crash[k], other[k]]))
        if terminated[0]:
            k_end = k
            assert info["status"][0] == CRASHED and info["status"][1] == FLYING
            assert not np.array_equal(observation[0], info["final_observation"][0])
            first = observation[0].copy()
            break
    assert k_end is not None
    fresh = CourseEnv(COURSE)
    np.testing.assert_array_equal(fresh.reset()[0][0], first)
    # Both go on: the restarted drone and a fresh environment, same actions.
    for j in range(60):
        a = np.stack([crash[j], other[k_end + 1 + j]])
        observation, reward, *_ = env.step(a)
        fresh_observation, fresh_reward, *_ = fresh.step(crash[j : j + 1])
        np.testing.assert_array_equal(observation[0], fresh_observation[0])
        assert reward[0] == fresh_reward[0]


def test_time_limit_truncates():
    data = {
        "name": "Short",
        "time_limit": 0.5,
        "room": {"size": [10.0, 10.0, 4.0]},
        "start": {"position": [2.0, 5.0], "yaw_deg": 0},
        "gates": [
            {
                "shape": "circle",
                "center": [6.0, 5.0, -1.5],
                "yaw_deg": 0,
                "radius": 0.8,
                "thickness": 0.1,
            }
        ],
    }
    env = CourseEnv(course_from_dict(data))
    env.reset()
    for k in range(25):
        _, _, terminated, truncated, info = env.step(np.zeros((1, 4)))
        assert truncated[0] == (k == 24) and not terminated[0]
    assert info["episode_steps"][0] == 25


def test_reset_with_a_seed_repeats_the_episode():
    actions = script(2, 150)
    runs = []
    env = CourseEnv(COURSE)
    for _ in range(2):
        observations = [env.reset(seed=7)[0]]
        for a in actions:
            observations.append(env.step(a[None])[0])
        runs.append(np.array(observations))
    np.testing.assert_array_equal(runs[0], runs[1])
    assert env.np_random.integers(1000) == np.random.default_rng(7).integers(1000)


def test_the_action_is_held_for_action_repeat_steps():
    """With action_repeat = 1 and twice the actions, the same flight."""
    actions = script(0, 120)
    slow = CourseEnv(COURSE)
    fast = CourseEnv(COURSE, action_repeat=1)
    slow.reset()
    fast.reset()
    for a in actions:
        _, reward, *_ = slow.step(a[None])
        r1 = fast.step(a[None])[1]
        r2 = fast.step(a[None])[1]
        np.testing.assert_array_equal(slow.flight.states, fast.flight.states)
        assert reward[0] == pytest.approx(r1[0] + r2[0], abs=1e-12)


def test_the_observation_holds_the_previous_action_clipped():
    env = CourseEnv(COURSE)
    env.reset()
    observation = env.step(np.array([[0.3, 2.0, -0.5, 0.1]]))[0]
    np.testing.assert_allclose(observation[0, -4:], [0.3, 1.0, -0.5, 0.1], atol=1e-7)


def test_a_restarted_drone_stands_landed_with_a_clean_loop():
    """After its crash the drone is put back on the floor with its pilot landed:
    turning there charges nothing (its last episode left the pilot flying)."""
    env = CourseEnv(COURSE)
    env.reset()
    for a in script(3, 400):
        terminated = env.step(a[None])[2]
        if terminated[0]:
            break
    assert terminated[0]
    assert env.pilot.landed[0]
    np.testing.assert_array_equal(env.flight.states[0], env.start)
    for _ in range(20):  # below the weight, asked to turn
        env.step(np.array([[-0.2, 0.5, -0.5, 0.3]]))
        assert np.all(env.pilot.rate_loop.integral == 0.0)
    assert env.pilot.landed[0] and env.flight.referee.status[0] == FLYING
