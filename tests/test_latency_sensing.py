"""Tests for what the policy sees and when its command arrives (phase 12, step
5): the latency against the reference engine with delayed commands, restarts
that clear the commands on their way, the observation built from the onboard
estimate, and the phase 11 policy with each effect."""

import dataclasses

import numpy as np
import pytest

from jineta.atalaya.estimator import MOCAP_LAB, Sensors
from jineta.campo.course import load_course
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.referee import CRASHED, FINISHED
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import RatePilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_HITBOX,
    IRIS_LANDING_GEAR,
    IRIS_RATE_GAINS,
)
from jineta.picadero.env import CourseEnv
from jineta.picadero.realism import IDEAL, LAB, Realism
from tests.test_env import COURSE, DT, LIMITS, script

PERFECT = Sensors(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def delayed_reference(actions: np.ndarray, delay: int, motor_lag: bool = False):
    """The actions flown by the reference engine (each held for two physics
    steps), every command reaching the rate pilot `delay` physics steps late
    (zero thrust and rates before the first arrives). Returns the states of
    every physics step and the referee."""
    vehicle = Multirotor(
        IRIS, contact=GroundContact(IRIS_LANDING_GEAR), motor_lag=motor_lag
    )
    pilot = RatePilot(vehicle, IRIS_RATE_GAINS, DT, landed=True)
    referee = COURSE.referee(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    thrust, rates = LIMITS.commands(actions)
    sent = {"steps": 0}

    def pre_step(t, y):
        k = sent["steps"] - delay  # the physics step whose command arrives
        sent["steps"] += 1
        if k < 0:
            pilot.thrust_command, pilot.rate_command = 0.0, np.zeros(3)
        else:
            pilot.thrust_command, pilot.rate_command = thrust[k // 2], rates[k // 2]
        pilot(t, y)

    start = vehicle.initial_state(
        COURSE.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    )
    _, states = run(
        vehicle.derivative, start, DT, 2 * len(actions) * DT,
        stop=referee, pre_step=pre_step, post_step=normalize_attitude,
    )  # fmt: skip
    return states, referee


@pytest.mark.parametrize(("latency", "motor_lag"), [(0.03, False), (0.02, True)])
def test_the_latency_is_the_reference_engine_with_delayed_commands(latency, motor_lag):
    """30 ms (three physics steps: an action arrives in the middle of the next
    one's two steps) and 20 ms with the motor lag: every state within 1e-9 of the
    reference engine whose commands arrive that late, and the same end."""
    actions = script(2, 400)
    delay = round(latency / DT)
    states, referee = delayed_reference(actions, delay, motor_lag)
    env = CourseEnv(COURSE, realism=Realism(motor_lag=motor_lag, latency=latency))
    env.reset()
    assert env.delays[0] == delay
    for k, action in enumerate(actions):
        _, _, terminated, truncated, _ = env.step(action[None])
        if terminated[0] or truncated[0]:
            break
        assert np.max(np.abs(env.flight.states[0] - states[2 * (k + 1)])) < 1e-9
    assert terminated[0] and referee.status == "crashed"
    # The latency changed the flight: without it the crash comes elsewhere.
    plain, _ = delayed_reference(actions, 0, motor_lag)
    assert len(plain) != len(states) or np.max(np.abs(plain - states)) > 1e-3


def test_a_restart_clears_the_commands_on_their_way():
    """With 40 ms of latency a crashed drone restarts with no old command in
    flight: its next episode is a fresh environment's first, bit for bit, and
    its neighbour is not disturbed."""
    realism = Realism(latency=0.04, motor_lag=True)
    steps = 400
    crash, other = script(3, steps), script(0, steps)
    env = CourseEnv(COURSE, n=2, realism=realism)
    env.reset()
    k_end = None
    for k in range(steps):
        observation, _, terminated, _, info = env.step(np.stack([crash[k], other[k]]))
        if terminated[0]:
            k_end = k
            assert info["status"][0] == CRASHED
            break
    assert k_end is not None
    alone = CourseEnv(COURSE, n=1, realism=realism)
    alone.reset()
    for k in range(k_end + 1):  # the neighbour, flown alone until the crash
        alone.step(other[k : k + 1])
    fresh = CourseEnv(COURSE, realism=realism)
    np.testing.assert_array_equal(fresh.reset()[0][0], observation[0])
    for j in range(60):
        observation, reward, *_ = env.step(np.stack([crash[j], other[k_end + 1 + j]]))
        fresh_observation, fresh_reward, *_ = fresh.step(crash[j : j + 1])
        alone_observation, *_ = alone.step(other[k_end + 1 + j : k_end + 2 + j])
        np.testing.assert_array_equal(observation[0], fresh_observation[0])
        np.testing.assert_array_equal(observation[1], alone_observation[0])
        assert reward[0] == fresh_reward[0]


def test_perfect_sensors_see_the_true_state():
    """Sensors without noise or delay: the same flight bit for bit (the physics
    does not look at the sensors) and the same observations to float32
    rounding."""
    actions = script(1, 150)
    ideal = CourseEnv(COURSE)
    sensed = CourseEnv(COURSE, realism=Realism(sensors=PERFECT))
    a, _ = ideal.reset(seed=1)
    b, _ = sensed.reset(seed=1)
    np.testing.assert_array_equal(a, b)
    for action in actions:
        a, *_ = ideal.step(action[None])
        b, *_ = sensed.step(action[None])
        np.testing.assert_array_equal(ideal.flight.states, sensed.flight.states)
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=1e-6)


def test_the_observation_is_built_from_the_estimate():
    """With the lab's sensors the observation is the observer's view of the
    estimated state (not of the true one), the noise repeats with the seed,
    and a restarted drone's estimate starts at its start."""
    actions = script(3, 200)
    env = CourseEnv(COURSE, n=1, realism=Realism(sensors=MOCAP_LAB))
    env.reset(seed=5)
    seen = []
    for action in actions:
        observation, _, terminated, _, _ = env.step(action[None])
        if terminated[0]:
            np.testing.assert_array_equal(env.estimator.estimate[0], env.start)
            break
        expected = env.observer(
            env.estimator.estimate, env.flight.referee.next_gate, action[None]
        ).astype(np.float32)
        np.testing.assert_array_equal(observation, expected)
        true_view = env.observer(
            env.flight.states, env.flight.referee.next_gate, action[None]
        ).astype(np.float32)
        assert not np.array_equal(observation, true_view)
        assert np.max(np.abs(observation - true_view)) < 0.1
        seen.append(observation)
    assert terminated[0]
    again = CourseEnv(COURSE, n=1, realism=Realism(sensors=MOCAP_LAB))
    again.reset(seed=5)
    for k, action in enumerate(actions[: len(seen)]):
        np.testing.assert_array_equal(again.step(action[None])[0], seen[k])
    other = CourseEnv(COURSE, n=1, realism=Realism(sensors=MOCAP_LAB))
    other.reset(seed=6)
    assert not np.array_equal(other.step(actions[0][None])[0], seen[0])


@pytest.fixture(scope="module")
def phase_11_policy():
    from jineta.doma.policy import Agent

    return Agent.load("policies/first_course.npz")


@pytest.mark.parametrize(
    ("realism", "status", "lap"),
    [
        (Realism(sensors=MOCAP_LAB), FINISHED, 3.232),
        (Realism(latency=0.02), FINISHED, 3.394),
        (Realism(latency=0.04), CRASHED, None),
        (Realism(motor_lag=True, latency=0.02), CRASHED, None),
        (LAB, FINISHED, 3.442),
    ],
)
def test_the_phase_11_policy_with_each_effect(phase_11_policy, realism, status, lap):
    """Measured: the lab's sensors alone change nothing that matters (3.232 s);
    20 ms of latency slows it (3.394 s, 5.8 cm from a frame); 40 ms crashes, and
    so does 20 ms with the motor lag; every effect of the lab at once happens to
    finish (3.442 s): the lap is on a knife's edge, not robust."""
    from jineta.picadero.training import fly

    course = load_course("courses/first_course.yaml")
    result = fly(phase_11_policy, CourseEnv(course, realism=realism))
    assert result["status"][0] == status
    if lap is not None:
        assert result["lap_time"][0] == pytest.approx(lap, abs=0.05)


def test_the_realism_switches():
    assert IDEAL == Realism() and IDEAL.latency_steps(DT) == 0
    assert LAB.motor_lag and LAB.ground_effect and LAB.sensors == MOCAP_LAB
    assert LAB.latency_steps(DT) == 2
    assert dataclasses.replace(LAB, latency=0.026).latency_steps(DT) == 3
    with pytest.raises(ValueError):
        Realism(latency=-0.01)
