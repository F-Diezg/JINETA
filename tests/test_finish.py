"""Tests for the stop after the finish (phase 12, step 7; loose end 39): when a
drone counts as stopped, an episode that goes on past the last gate until the
drone stops (the same lap time, a bonus, the seconds paid), crashes or runs out
of time, the flight after the finish equal to the reference engine, the flag in
the observation, and restarts."""

import numpy as np
import pytest

from jineta.campo.course import course_from_dict
from jineta.core import quaternion
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.control import BatchAttitudeController, BatchVelocityController
from jineta.escuadron.referee import CRASHED, FINISHED, FLYING, TIMEOUT
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
from jineta.picadero.baseline import BaselineSettings, PathGuidance
from jineta.picadero.env import CourseEnv
from jineta.picadero.finish import StopAfterFinish
from jineta.picadero.reward import RewardWeights
from tests.test_env import COURSE, DT, LIMITS
from tests.test_training import BaselineAgent

WEIGHTS = RewardWeights(progress=1.0, gate=1.0, crash=5.0, time=0.25, stop=1.0)
# Two gates in a long hall: room to stop after the last one.
OPEN = course_from_dict(
    {
        "name": "Open hall",
        "time_limit": 20,
        "room": {"size": [22.0, 8.0, 4.0]},
        "start": {"position": [1.5, 4.0], "yaw_deg": 0},
        "gates": [
            {"shape": "rectangle", "center": [5.0, 4.0, -1.5], "yaw_deg": 0,
             "width": 1.6, "height": 1.4, "thickness": 0.1},
            {"shape": "circle", "center": [9.0, 4.5, -1.5], "yaw_deg": 10,
             "radius": 0.8, "thickness": 0.1},
        ],
    }
)  # fmt: skip


class Braking:
    """The baseline's path follower that, once every gate is crossed, asks the
    velocity loop to stop."""

    def __init__(self, env: CourseEnv) -> None:
        self.env = env
        settings = BaselineSettings(
            cruise=4.0, lateral_accel=5.0, braking_accel=4.0, lookahead=2.5,
            approach=1.5,
        )  # fmt: skip
        self.guidance = PathGuidance(env.course, settings, env.n)
        self.velocity = BatchVelocityController(
            IRIS, IRIS_GAINS.velocity, env.step_time, env.n
        )
        self.attitude = BatchAttitudeController(IRIS_GAINS.attitude)

    def act(self, observations, rng=None):
        states = self.env.flight.states
        velocity, yaw = self.guidance.commands(states)
        after = self.env.flight.referee.next_gate < 0
        velocity = np.where(after[:, None], 0.0, velocity)
        attitude, thrust = self.velocity.update(velocity, yaw, states)
        return self.env.limits.actions(thrust, self.attitude.update(attitude, states))


def episode(env: CourseEnv, agent, steps: int = 1000):
    """One episode of drone 0: (observations, rewards, the last step's
    terminated, truncated and info, the number of steps)."""
    observation, _ = env.reset()
    observations, rewards = [observation], []
    for k in range(steps):
        observation, reward, terminated, truncated, info = env.step(
            agent.act(observation)
        )
        observations.append(info["final_observation"])
        rewards.append(reward[0])
        if terminated[0] or truncated[0]:
            return observations, rewards, terminated[0], truncated[0], info, k + 1
    raise AssertionError("the episode did not end")


def level(speed=0.0, tilt_deg=0.0, rate=0.0) -> np.ndarray:
    state = np.zeros(13)
    state[3] = speed
    state[6:10] = quaternion.from_euler(np.radians(tilt_deg), 0.0, 0.0)
    state[10] = rate
    return state


def test_when_a_drone_is_stopped():
    rule = StopAfterFinish(speed=0.5, tilt=np.radians(15.0), rate=0.5)
    states = np.array(
        [level(), level(0.49, 14.9, 0.49), level(0.51), level(tilt_deg=15.1),
         level(rate=0.51), level(tilt_deg=180.0)]
    )  # fmt: skip
    np.testing.assert_array_equal(
        rule.stopped(states), [True, True, False, False, False, False]
    )
    for bad in ({"window": 0.0}, {"speed": -1.0}, {"tilt": 0.0}, {"rate": 0.0}):
        with pytest.raises(ValueError):
            StopAfterFinish(**bad)


def test_a_drone_that_stops_after_the_finish():
    """With room to stop: the episode does not end at the last gate but when
    the drone has stopped, a second later; the lap time is the crossing's, the
    same as without the rule; the return is the one without the rule plus the
    stop bonus minus the seconds spent stopping (0.25 per second)."""
    plain = CourseEnv(OPEN, weights=WEIGHTS)
    _, plain_rewards, done, _, plain_info, plain_steps = episode(plain, Braking(plain))
    assert done and plain_info["status"][0] == FINISHED
    env = CourseEnv(OPEN, weights=WEIGHTS, stop=StopAfterFinish())
    observations, rewards, done, cut, info, steps = episode(env, Braking(env))
    assert done and not cut and info["status"][0] == FINISHED
    assert info["lap_time"][0] == plain_info["lap_time"][0]
    assert steps > plain_steps + 10
    final = info["final_state"][0]
    assert StopAfterFinish().stopped(final[None])[0]
    # The rewards before the finish step are the same; after it, each physics
    # step costs 0.25 dt, and the stop pays 1.
    np.testing.assert_array_equal(rewards[: plain_steps - 1], plain_rewards[:-1])
    extra = info["episode_return"][0] - plain_info["episode_return"][0]
    physics_after = round((1.0 - extra) / (0.25 * DT))
    assert extra == pytest.approx(1.0 - 0.25 * DT * physics_after, abs=1e-9)
    assert 2 * (steps - plain_steps) - 1 <= physics_after <= 2 * (steps - plain_steps)
    # The flag in the observation: 0 until the last gate, 1 after it.
    flags = np.array([o[0, -1] for o in observations])
    assert observations[0].shape == (1, 44)
    assert np.all(flags[: plain_steps - 1] == 0.0) and np.all(
        flags[plain_steps:] == 1.0
    )


def test_a_drone_that_cannot_stop_runs_out_of_time():
    env = CourseEnv(OPEN, weights=WEIGHTS, stop=StopAfterFinish(window=0.2))
    _, _, done, cut, info, _ = episode(env, Braking(env))
    assert cut and not done
    assert info["status"][0] == TIMEOUT and np.isnan(info["lap_time"][0])


def reference_after_the_finish(actions: np.ndarray):
    """The actions flown by the reference engine on the first course with only
    the collision monitor as `stop` (it does not stop at the finish)."""
    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    pilot = RatePilot(vehicle, IRIS_RATE_GAINS, DT, landed=True)
    monitor = COURSE.monitor(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    thrust, rates = LIMITS.commands(actions)
    count = {"steps": 0}

    def pre_step(t, y):
        k = count["steps"] // 2
        count["steps"] += 1
        pilot.thrust_command, pilot.rate_command = thrust[k], rates[k]
        pilot(t, y)

    start = COURSE.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    _, states = run(
        vehicle.derivative, start, DT, 2 * len(actions) * DT,
        stop=monitor, pre_step=pre_step, post_step=normalize_attitude,
    )  # fmt: skip
    return states, monitor


def test_a_drone_that_crashes_after_the_finish():
    """The baseline flies on past the last gate of the first course into the
    wall 1.5 m behind it: without the rule a finish, with it a crash (no lap
    time, the crash penalty); the flight after the finish is the reference
    engine's, which only stops at the collision (every state within 1e-9, the
    same last sample)."""
    env = CourseEnv(COURSE, weights=WEIGHTS, stop=StopAfterFinish())
    agent = BaselineAgent(env)
    observation, _ = env.reset()
    actions, states = [], []
    for _ in range(1000):
        action = agent.act(observation)
        actions.append(action[0])
        observation, reward, terminated, truncated, info = env.step(action)
        states.append(info["final_state"][0])
        if terminated[0] or truncated[0]:
            break
    assert terminated[0] and info["status"][0] == CRASHED
    assert np.isnan(info["lap_time"][0]) and reward[0] < -4.0
    reference, monitor = reference_after_the_finish(np.array(actions))
    assert monitor.collision is not None and monitor.collision.kind == "room"
    for k, state in enumerate(states[:-1]):
        assert np.max(np.abs(state - reference[2 * (k + 1)])) < 1e-9
    # The crash step: the environment's last state is the reference's last.
    assert len(reference) in (2 * len(states), 2 * len(states) + 1)
    assert np.max(np.abs(states[-1] - reference[-1])) < 1e-9


def test_restarts_and_drones_beside_each_other():
    """Two drones: one stops after the finish and restarts, the other is held
    on the floor; the stopped drone's next episode starts clean."""
    env = CourseEnv(OPEN, n=2, weights=WEIGHTS, stop=StopAfterFinish())
    agent = Braking(env)
    observation, _ = env.reset()
    for _ in range(1000):
        action = agent.act(observation)
        action[1] = [-1.0, 0.0, 0.0, 0.0]
        observation, _, terminated, _, info = env.step(action)
        if terminated[0]:
            break
    assert info["status"][0] == FINISHED and info["status"][1] == FLYING
    assert not env.stopping[0] and env.outcome[0] == FLYING
    assert observation[0, -1] == 0.0  # the flag is back to 0


def test_without_the_rule_the_observation_is_as_before():
    assert CourseEnv(OPEN).observation_space.shape == (43,)
    assert CourseEnv(OPEN, stop=StopAfterFinish()).observation_space.shape == (44,)


def test_the_phase_11_policy_does_not_stop():
    """The phase 11 policy (its observation without the flag) crosses the last
    gate of the first course at 7.6 m/s, 2 m from the wall: it crashes after the
    finish (about 0.3 s after it, measured)."""
    from jineta.doma.policy import Agent

    agent = Agent.load("policies/first_course.npz")
    env = CourseEnv(COURSE, stop=StopAfterFinish())
    observation, _ = env.reset()
    for _ in range(400):
        observation, _, terminated, truncated, info = env.step(
            agent.act(observation[:, :43])
        )
        if terminated[0] or truncated[0]:
            break
    assert info["status"][0] == CRASHED
