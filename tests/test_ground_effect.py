"""Tests for the ground effect (phase 12, step 3): Cheeseman and Bennett's
factor by hand, the height at which fixed rotor speeds hold the Iris up, a
tilted drone near the floor levelled by it, the batch against the reference
(one derivative and a whole episode), and the engine without it unchanged."""

import numpy as np
import pytest

from jineta.core.environment import GRAVITY
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.jinete.contact import GroundContact
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR
from jineta.jinete.proximity import GroundEffect, ground_effect_factor
from jineta.picadero.env import CourseEnv
from jineta.picadero.realism import Realism
from tests.test_batch_multirotor import random_quaternions
from tests.test_env import COURSE, script
from tests.test_env import reference_episode as env_reference_episode

R = IRIS.propeller_radius
RHO = 1.225
HUB_UP = 0.023  # the Iris's hubs above its centre of mass [m]


def hover_speed(fraction: float = 1.0) -> float:
    """The rotor speed at which the four rotors give `fraction` of the weight."""
    return float(np.sqrt(fraction * IRIS.mass * GRAVITY / 4 / IRIS.thrust_coefficient))


def level_state(com_height: float, speed: float = 0.0) -> np.ndarray:
    """The Iris level and still, its centre of mass `com_height` above the
    floor, moving North at `speed`."""
    state = np.zeros(13)
    state[2], state[3], state[6] = -com_height, speed, 1.0
    return state


def test_the_factor_by_hand():
    thrust = 3.7
    # Hovering: 1 / (1 - (R / 4z)^2).
    assert ground_effect_factor(R, R, thrust, 0.0, RHO) == pytest.approx(16 / 15)
    assert ground_effect_factor(R, 2 * R, thrust, 0.0, RHO) == pytest.approx(64 / 63)
    assert ground_effect_factor(R, 4 * R, thrust, 0.0, RHO) == pytest.approx(256 / 255)
    # Not below half a radius (4/3), whatever the height, even below the floor.
    for height in (R / 2, R / 4, 0.0, -0.1):
        assert ground_effect_factor(R, height, thrust, 0.0, RHO) == pytest.approx(4 / 3)
    assert ground_effect_factor(R, R / 4, thrust, 0.0, RHO, min_height=1.0) == (
        pytest.approx(16 / 15)
    )
    # Forward flight: at V = v_i the excess (R / 4z)^2 is halved.
    induced = np.sqrt(thrust / (2 * RHO * np.pi * R * R))
    assert ground_effect_factor(R, R, thrust, induced, RHO) == pytest.approx(32 / 31)
    assert ground_effect_factor(R, R, thrust, 3 * induced, RHO) == pytest.approx(
        1 / (1 - 1 / 160)
    )
    # No thrust, no wake, no effect; far away almost none.
    assert ground_effect_factor(R, R, 0.0, 0.0, RHO) == 1.0
    assert ground_effect_factor(R, 10.0, thrust, 0.0, RHO) - 1 < 1.1e-5
    # Many rotors at once, elementwise.
    heights = np.array([[R, 2 * R], [4 * R, R / 2]])
    np.testing.assert_allclose(
        ground_effect_factor(R, heights, thrust, 0.0, RHO),
        [[16 / 15, 64 / 63], [256 / 255, 4 / 3]],
    )
    with pytest.raises(ValueError):
        GroundEffect(min_height=0.25)


def test_fixed_rotor_speeds_hold_the_iris_at_the_height_the_formula_gives():
    """Rotors spinning to lift 90 % of the weight free of the floor hold the
    Iris up where their factor is 1 / 0.9: hubs at R / (4 sqrt(0.1)) above the
    floor (10.1 cm; its feet 2.3 cm up). Higher it sinks, lower it rises."""
    vehicle = Multirotor(IRIS, ground_effect=GroundEffect())
    vehicle.rotor_speed_command = np.full(4, hover_speed(0.9))
    hub_height = R / (4 * np.sqrt(0.1))
    equilibrium = hub_height - HUB_UP
    acceleration = vehicle.derivative(0.0, level_state(equilibrium))
    assert np.max(np.abs(acceleration[3:13])) < 1e-12
    assert vehicle.derivative(0.0, level_state(equilibrium + 0.01))[5] > 0.0
    assert vehicle.derivative(0.0, level_state(equilibrium - 0.01))[5] < 0.0
    # Without the effect the same speeds let it fall at 10 % of g.
    plain = Multirotor(IRIS)
    plain.rotor_speed_command = vehicle.rotor_speed_command
    assert plain.derivative(0.0, level_state(equilibrium))[5] == pytest.approx(
        0.1 * GRAVITY
    )


def test_moving_fast_the_floor_matters_less():
    """At hover speed, 10 cm above the floor: standing still the floor pushes
    up harder than flying at 6 m/s along it."""
    vehicle = Multirotor(IRIS, ground_effect=GroundEffect())
    vehicle.rotor_speed_command = np.full(4, hover_speed())
    still = vehicle.force_moment(0.0, level_state(0.1))[0][2]
    moving = vehicle.force_moment(0.0, level_state(0.1, speed=6.0))[0][2]
    plain = Multirotor(IRIS)
    plain.rotor_speed_command = vehicle.rotor_speed_command
    free = plain.force_moment(0.0, level_state(0.1))[0][2]
    assert still < moving < free < 0.0  # thrust is along -z


def test_a_tilted_drone_near_the_floor_is_levelled():
    """Rolled 10° to the right, 15 cm up, equal rotor speeds: the low (right)
    rotors push harder, a moment that rolls it back left; the same pitched nose
    down. Free of the floor, equal speeds give no roll or pitch moment."""
    from jineta.core import quaternion

    vehicle = Multirotor(IRIS, ground_effect=GroundEffect())
    vehicle.rotor_speed_command = np.full(4, hover_speed())
    plain = Multirotor(IRIS)
    plain.rotor_speed_command = vehicle.rotor_speed_command
    rolled = level_state(0.15)
    rolled[6:10] = quaternion.from_euler(np.radians(10.0), 0.0, 0.0)
    pitched = level_state(0.15)
    pitched[6:10] = quaternion.from_euler(0.0, np.radians(-10.0), 0.0)
    assert vehicle.force_moment(0.0, rolled)[1][0] < -0.01
    assert vehicle.force_moment(0.0, pitched)[1][1] > 0.01
    for state in (rolled, pitched):
        assert np.max(np.abs(plain.force_moment(0.0, state)[1][:2])) < 1e-12
    # The yaw moment (the reaction torques) is the one free of the floor.
    assert vehicle.force_moment(0.0, rolled)[1][2] == pytest.approx(
        plain.force_moment(0.0, rolled)[1][2], abs=1e-15
    )


@pytest.mark.parametrize("motor_lag", [False, True])
def test_the_batch_derivative_matches_the_reference_for_every_drone(motor_lag):
    """300 drones from 10 cm below the floor to 1 m above it, tilted, moving and
    turning, rotor commands (and speeds) inside and outside their range: every
    derivative within 1e-12 of the reference's (relative to its size)."""
    rng = np.random.default_rng(11)
    n = 300
    size = 17 if motor_lag else 13
    states = np.zeros((n, size))
    states[:, 0:2] = rng.uniform(-2.0, 2.0, (n, 2))
    states[:, 2] = rng.uniform(-1.0, 0.1, n)
    states[:, 3:6] = rng.normal(0.0, 3.0, (n, 3))
    states[:, 6:10] = random_quaternions(rng, n)
    states[:, 10:13] = rng.normal(0.0, 2.0, (n, 3))
    if motor_lag:
        states[:, 13:] = rng.uniform(-50.0, 1200.0, (n, 4))
    commands = rng.uniform(-100.0, 1200.0, (n, 4))
    batch = BatchMultirotor(
        IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR),
        motor_lag=motor_lag, ground_effect=GroundEffect(),
    )  # fmt: skip
    batch.rotor_speed_command = commands
    derivatives = batch.derivative(0.0, states)
    reference = Multirotor(
        IRIS, contact=GroundContact(IRIS_LANDING_GEAR),
        motor_lag=motor_lag, ground_effect=GroundEffect(),
    )  # fmt: skip
    affected = 0
    plain = Multirotor(IRIS, motor_lag=motor_lag)
    for i in range(n):
        reference.rotor_speed_command = commands[i]
        expected = reference.derivative(0.0, states[i])
        scale = np.maximum(1.0, np.abs(expected))
        assert np.max(np.abs(derivatives[i] - expected) / scale) < 1e-12
        plain.rotor_speed_command = commands[i]
        free = plain.force_moment(0.0, states[i])[0][2]
        affected += reference.force_moment(0.0, states[i])[0][2] < free - 1e-3
    assert affected > 50  # many of them close enough to the floor to feel it


def test_without_it_nothing_changes():
    """A ground effect only changes the thrust and its moments: with the rotors
    stopped every force is the same bit for bit; and a vehicle without one is
    the vehicle of before (the other tests of the engine)."""
    rng = np.random.default_rng(5)
    states = np.zeros((20, 13))
    states[:, 2] = rng.uniform(-0.5, 0.0, 20)
    states[:, 3:6] = rng.normal(0.0, 2.0, (20, 3))
    states[:, 6:10] = random_quaternions(rng, 20)
    with_it = BatchMultirotor(IRIS, 20, ground_effect=GroundEffect())
    without = BatchMultirotor(IRIS, 20)
    for a, b in zip(
        with_it.force_moment(0.0, states), without.force_moment(0.0, states),
        strict=True,
    ):  # fmt: skip
        np.testing.assert_array_equal(a, b)


def test_an_episode_with_it_is_the_reference_engine_flight():
    """The environment with the ground effect (and the motor lag) against the
    reference engine with both: every state within 1e-9, the same end and
    rewards; the take-off differs from the flight without it."""
    actions = script(1, 400)
    states, referee, rewards = env_reference_episode(
        actions, motor_lag=True, ground_effect=GroundEffect()
    )
    env = CourseEnv(COURSE, realism=Realism(motor_lag=True, ground_effect=True))
    env.reset()
    for k, action in enumerate(actions):
        _, reward, terminated, truncated, _ = env.step(action[None])
        assert reward[0] == pytest.approx(rewards[k], abs=1e-9)
        if terminated[0] or truncated[0]:
            break
        assert np.max(np.abs(env.flight.states[0] - states[2 * (k + 1)])) < 1e-9
    assert terminated[0] and referee.status == "crashed"
    without, _, _ = env_reference_episode(actions, motor_lag=True)
    assert np.max(np.abs(without[30, 0:3] - states[30, 0:3])) > 1e-3


def test_the_phase_11_policy_with_it():
    """Measured: 3.26 s (3.235 s without), farther from things (21.8 cm)."""
    from jineta.campo.course import load_course
    from jineta.doma.policy import Agent
    from jineta.picadero.training import fly

    agent = Agent.load("policies/first_course.npz")
    course = load_course("courses/first_course.yaml")
    lap = fly(agent, CourseEnv(course, realism=Realism(ground_effect=True)))
    assert lap["lap_time"][0] == pytest.approx(3.26, abs=0.05)


def test_rejects_a_ground_effect_without_a_radius():
    import dataclasses

    plain = dataclasses.replace(IRIS, propeller_radius=0.0)
    with pytest.raises(ValueError):
        Multirotor(plain, ground_effect=GroundEffect())
    with pytest.raises(ValueError):
        BatchMultirotor(plain, 2, ground_effect=GroundEffect())
    with pytest.raises(ValueError):
        dataclasses.replace(IRIS, propeller_radius=-0.1)
