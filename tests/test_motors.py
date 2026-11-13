"""Tests for the motor lag (phase 12, step 2): each rotor's speed follows its
command as a first order (12.5 ms up, 25 ms down for the Iris), part of the
state. Checked against the exact discrete solution of RK4 and the continuous
exponential, hover as an equilibrium, the batch against the reference (one
derivative and a whole episode of the environment), the engine without the
lag unchanged, and the rate loop with the lag."""

import dataclasses

import numpy as np
import pytest

from jineta.campo.course import load_course
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.jinete.contact import GroundContact
from jineta.jinete.control import RatePilot
from jineta.jinete.multirotor import Multirotor, MultirotorParams, motor_derivative
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR, IRIS_RATE_GAINS
from jineta.picadero.env import CourseEnv
from jineta.picadero.realism import Realism
from tests.test_batch_multirotor import (
    random_commands,
    random_quaternions,
)
from tests.test_env import COURSE, script
from tests.test_env import reference_episode as env_reference_episode

DT = 0.01
HOVER = float(np.sqrt(IRIS.mass * GRAVITY / 4 / IRIS.thrust_coefficient))
UP, DOWN = IRIS.motor_time_constant_up, IRIS.motor_time_constant_down


def rk4_factor(z: float) -> float:
    """What one RK4 step multiplies the solution of x' = lambda x by, z =
    lambda dt (the exact discrete solution of a linear first order)."""
    return 1.0 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def airborne(speeds: float) -> np.ndarray:
    """A level Iris 5 m up, still, its rotors at `speeds`."""
    state = np.zeros(17)
    state[2], state[6] = -5.0, 1.0
    state[13:] = speeds
    return state


def test_the_iris_time_constants():
    # PX4-SITL_gazebo-classic, models/iris/iris.sdf.jinja, every motor plugin.
    assert (UP, DOWN) == (0.0125, 0.025)


def test_motor_derivative_by_hand():
    command = np.array([800.0, 200.0, 1500.0, -50.0])
    speeds = np.array([600.0, 400.0, 1000.0, 100.0])
    expected = [
        (800 - 600) / UP,  # speeding up
        (200 - 400) / DOWN,  # slowing down
        (1100 - 1000) / UP,  # the command clipped to the maximum
        (0 - 100) / DOWN,  # and to zero
    ]
    np.testing.assert_allclose(
        motor_derivative(IRIS, command, speeds), expected, rtol=1e-15
    )
    assert np.all(motor_derivative(IRIS, speeds, speeds) == 0.0)
    # Many vehicles at once, row by row.
    stacked = motor_derivative(IRIS, np.tile(command, (3, 1)), np.tile(speeds, (3, 1)))
    np.testing.assert_array_equal(stacked, np.tile(expected, (3, 1)))


@pytest.mark.parametrize(
    ("start", "command", "tau"), [(HOVER, 1000.0, UP), (HOVER, 500.0, DOWN)]
)
def test_rotor_speed_follows_the_exact_discrete_and_continuous_lag(start, command, tau):
    """A step of the command in free flight: every sample of the rotor speeds is
    the exact discrete solution of RK4, command + (start - command) R^k
    (1e-9 rad/s), and close to the continuous exponential (within 0.3 % of the
    step when speeding up, dt / tau = 0.8, and 0.02 % slowing down, 0.4)."""
    vehicle = Multirotor(IRIS, motor_lag=True)
    vehicle.rotor_speed_command = np.full(4, command)
    times, states = run(vehicle.derivative, airborne(start), DT, 0.2)
    k = np.arange(len(times))
    discrete = command + (start - command) * rk4_factor(-DT / tau) ** k
    for rotor in range(4):
        np.testing.assert_allclose(states[:, 13 + rotor], discrete, atol=1e-9)
    continuous = command + (start - command) * np.exp(-times / tau)
    worst = np.max(np.abs(states[:, 13] - continuous)) / abs(command - start)
    assert worst < (0.003 if tau == UP else 2e-4)
    assert worst > 0.0  # RK4 at 100 Hz is not exact for the lag


def test_speeding_up_is_twice_as_fast_as_slowing_down():
    """After one time constant up (12.5 ms, not a multiple of the step: checked
    at 0.1 s = 8 tau_up = 4 tau_down) the speed-up has gone further."""
    up = Multirotor(IRIS, motor_lag=True)
    up.rotor_speed_command = np.full(4, HOVER + 200.0)
    down = Multirotor(IRIS, motor_lag=True)
    down.rotor_speed_command = np.full(4, HOVER - 200.0)
    _, rising = run(up.derivative, airborne(HOVER), DT, 0.1)
    _, falling = run(down.derivative, airborne(HOVER), DT, 0.1)
    left_up = abs(rising[-1, 13] - (HOVER + 200.0)) / 200.0
    left_down = abs(falling[-1, 13] - (HOVER - 200.0)) / 200.0
    assert left_up == pytest.approx(rk4_factor(-0.8) ** 10, rel=1e-6)
    assert left_down == pytest.approx(rk4_factor(-0.4) ** 10, rel=1e-6)
    assert left_up < left_down / 10


def test_hover_is_an_equilibrium_with_the_lag():
    vehicle = Multirotor(IRIS, motor_lag=True)
    vehicle.rotor_speed_command = np.full(4, HOVER)
    state = airborne(HOVER)
    assert np.max(np.abs(vehicle.derivative(0.0, state))) < 1e-12
    _, states = run(vehicle.derivative, state, DT, 10.0)
    assert np.max(np.abs(states[-1] - state)) < 1e-9


def test_the_forces_use_the_rotor_speeds_of_the_state():
    """With the lag, the forces are those of the instant motors spinning at the
    state's speeds, bit for bit, whatever the command; the derivative of the
    body is the same too."""
    rng = np.random.default_rng(3)
    lagging = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR), motor_lag=True)
    instant = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    for _ in range(50):
        state = airborne(0.0)
        state[0:3] = rng.uniform(-1.0, 1.0, 3) + np.array([0.0, 0.0, -0.05])
        state[3:6] = rng.normal(0.0, 3.0, 3)
        state[6:10] = random_quaternions(rng, 1)[0]
        state[10:13] = rng.normal(0.0, 2.0, 3)
        state[13:] = rng.uniform(-50.0, 1200.0, 4)
        lagging.rotor_speed_command = rng.uniform(0.0, 1100.0, 4)
        instant.rotor_speed_command = state[13:].copy()
        for a, b in zip(
            lagging.force_moment(0.0, state),
            instant.force_moment(0.0, state[:13]),
            strict=True,
        ):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            lagging.derivative(0.0, state)[:13], instant.derivative(0.0, state[:13])
        )


def test_without_the_lag_the_state_and_forces_are_as_before():
    vehicle = Multirotor(IRIS)
    assert vehicle.state_size == 13
    body = airborne(0.0)[:13]
    np.testing.assert_array_equal(vehicle.initial_state(body), body)
    vehicle.rotor_speed_command = np.array([-5.0, 300.0, 900.0, 2000.0])
    np.testing.assert_array_equal(
        vehicle.rotor_speeds(body), [0.0, 300.0, 900.0, 1100.0]
    )
    lagging = Multirotor(IRIS, motor_lag=True)
    assert lagging.state_size == 17
    np.testing.assert_array_equal(lagging.initial_state(body)[13:], np.zeros(4))
    np.testing.assert_array_equal(
        lagging.initial_state(body, [1, 2, 3, 4])[13:], [1, 2, 3, 4]
    )


def test_the_batch_derivative_matches_the_reference_for_every_drone():
    """300 drones near the floor and in the air, any attitude, rotor speeds and
    commands inside and outside their range: every derivative (the body's 13 and
    the rotors' 4) within 1e-12 of the reference's (relative to its size)."""
    rng = np.random.default_rng(7)
    n = 300
    states = np.zeros((n, 17))
    states[:, 0:2] = rng.uniform(-2.0, 2.0, (n, 2))
    states[:, 2] = rng.uniform(-0.3, 0.1, n)
    states[:, 3:6] = rng.normal(0.0, 3.0, (n, 3))
    states[:, 6:10] = random_quaternions(rng, n)
    states[:, 10:13] = rng.normal(0.0, 2.0, (n, 3))
    states[:, 13:] = rng.uniform(-50.0, 1200.0, (n, 4))
    commands = random_commands(rng, n)
    batch = BatchMultirotor(
        IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR), motor_lag=True
    )
    batch.rotor_speed_command = commands
    derivatives = batch.derivative(0.0, states)
    assert derivatives.shape == (n, 17)
    reference = Multirotor(
        IRIS, contact=GroundContact(IRIS_LANDING_GEAR), motor_lag=True
    )
    for i in range(n):
        reference.rotor_speed_command = commands[i]
        expected = reference.derivative(0.0, states[i])
        scale = np.maximum(1.0, np.abs(expected))
        assert np.max(np.abs(derivatives[i] - expected) / scale) < 1e-12


def test_an_episode_with_the_lag_is_the_reference_engine_flight():
    """The environment with motor lag and the reference engine with it (the
    same actions through RatePilot): every state within 1e-9, rotor speeds
    included, the same end and rewards."""
    actions = script(2, 400)
    states, referee, rewards = env_reference_episode(actions, motor_lag=True)
    assert states.shape[1] == 17
    env = CourseEnv(COURSE, realism=Realism(motor_lag=True))
    env.reset()
    assert env.flight.states.shape == (1, 17)
    for k, action in enumerate(actions):
        _, reward, terminated, truncated, info = env.step(action[None])
        assert reward[0] == pytest.approx(rewards[k], abs=1e-9)
        if terminated[0] or truncated[0]:
            break
        assert np.max(np.abs(env.flight.states[0] - states[2 * (k + 1)])) < 1e-9
    assert terminated[0] and referee.status == "crashed"
    assert info["final_state"].shape == (1, 17)
    # It crashed later than without the lag, from the same actions: the motors
    # spin up more slowly (the take-off is delayed).
    without, _, _ = env_reference_episode(actions)
    assert len(states) != len(without)


def test_the_rate_loop_stays_stable_with_the_lag():
    """A roll-rate step of 2 rad/s at hover: with the lag the loop overshoots
    more (10 % against 5 %, measured), and both settle near the command."""
    overshoot = {}
    for lag in (False, True):
        vehicle = Multirotor(IRIS, motor_lag=lag)
        pilot = RatePilot(vehicle, IRIS_RATE_GAINS, DT)
        pilot.thrust_command = IRIS.mass * GRAVITY
        pilot.rate_command = np.array([2.0, 0.0, 0.0])
        start = vehicle.initial_state(airborne(HOVER)[:13], HOVER)
        _, states = run(
            vehicle.derivative, start, DT, 0.6,
            pre_step=pilot, post_step=normalize_attitude,
        )  # fmt: skip
        roll_rate = states[:, 10]
        overshoot[lag] = roll_rate.max() / 2.0 - 1.0
        assert abs(roll_rate[-1] - 2.0) < 0.05
    assert 0.03 < overshoot[False] < overshoot[True] < 0.15


def test_the_phase_11_policy_with_the_lag():
    """The policy of phase 11 (trained with instant motors) still finishes the
    first course with the lag, a little slower and closer to things (measured:
    3.276 s against 3.235 s, 8.7 cm against 13.0 cm)."""
    from jineta.doma.policy import Agent
    from jineta.picadero.training import fly

    agent = Agent.load("policies/first_course.npz")
    course = load_course("courses/first_course.yaml")
    lap = fly(agent, CourseEnv(course, realism=Realism(motor_lag=True)))
    assert lap["lap_time"][0] == pytest.approx(3.276, abs=0.05)
    assert 0.05 < lap["clearance"][0] < 0.13


def test_rejects_a_lag_without_time_constants():
    plain = MultirotorParams(
        mass=IRIS.mass,
        inertia=IRIS.inertia,
        rotors=IRIS.rotors,
        thrust_coefficient=IRIS.thrust_coefficient,
        moment_constant=IRIS.moment_constant,
        max_rotor_speed=IRIS.max_rotor_speed,
        rotor_drag_coefficient=IRIS.rotor_drag_coefficient,
        body_drag_area=IRIS.body_drag_area,
    )
    with pytest.raises(ValueError):
        Multirotor(plain, motor_lag=True)
    with pytest.raises(ValueError):
        BatchMultirotor(plain, 2, motor_lag=True)
    Multirotor(plain)  # without the lag it does not need them
    with pytest.raises(ValueError):
        dataclasses.replace(plain, motor_time_constant_up=-0.01)
