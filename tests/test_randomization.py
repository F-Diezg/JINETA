"""Tests for drones with their own numbers and the domain randomisation (phase
12, step 6): every drone of a batch flies as the reference engine with its own
numbers (one derivative, and a whole episode with the controller keeping the
profile's), the profile's numbers given one by one change nothing, draws inside
their ranges and repeated by the seed, a restart draws only the restarted drone
anew, and the phase 11 policy on drawn drones."""

import numpy as np
import pytest

from jineta.campo.course import load_course
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.multirotor import PER_DRONE, BatchMultirotor
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import RatePilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_HITBOX,
    IRIS_LANDING_GEAR,
    IRIS_RATE_GAINS,
)
from jineta.jinete.proximity import GroundEffect
from jineta.picadero.env import CourseEnv
from jineta.picadero.randomization import NONE, Randomization
from jineta.picadero.realism import RANDOMISED_LAB, Realism
from tests.test_batch_multirotor import random_quaternions
from tests.test_env import COURSE, DT, LIMITS, script

DRAWN = Randomization()
NUMBERS = (
    "mass", "inertia", "thrust_coefficient", "moment_constant", "max_rotor_speed",
    "rotor_drag_coefficient", "body_drag_area", "motor_time_constant_up",
    "motor_time_constant_down",
)  # fmt: skip


def same_numbers(a, b) -> bool:
    """Two profiles with the same per-drone numbers (bit for bit)."""
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in NUMBERS)


def random_states(rng, n, size):
    states = np.zeros((n, size))
    states[:, 0:2] = rng.uniform(-2.0, 2.0, (n, 2))
    states[:, 2] = rng.uniform(-1.0, 0.1, n)
    states[:, 3:6] = rng.normal(0.0, 3.0, (n, 3))
    states[:, 6:10] = random_quaternions(rng, n)
    states[:, 10:13] = rng.normal(0.0, 2.0, (n, 3))
    if size > 13:
        states[:, 13:] = rng.uniform(-50.0, 1200.0, (n, size - 13))
    return states


def test_every_drone_is_the_reference_with_its_own_numbers():
    """300 drones drawn around the Iris (motor lag, ground effect, floor
    contact), random states and commands: every derivative within 1e-12 of the
    reference Multirotor built with that drone's numbers."""
    rng = np.random.default_rng(21)
    n = 300
    batch = BatchMultirotor(
        IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR),
        motor_lag=True, ground_effect=GroundEffect(),
    )  # fmt: skip
    values, _ = DRAWN.draw(IRIS, rng, n)
    batch.set_parameters(np.arange(n), **values)
    states = random_states(rng, n, 17)
    commands = rng.uniform(-100.0, 1200.0, (n, 4))
    batch.rotor_speed_command = commands
    derivatives = batch.derivative(0.0, states)
    for i in range(n):
        reference = Multirotor(
            batch.drone_params(i), contact=GroundContact(IRIS_LANDING_GEAR),
            motor_lag=True, ground_effect=GroundEffect(),
        )  # fmt: skip
        reference.rotor_speed_command = commands[i]
        expected = reference.derivative(0.0, states[i])
        scale = np.maximum(1.0, np.abs(expected))
        assert np.max(np.abs(derivatives[i] - expected) / scale) < 1e-12
    # The drawn drones really differ from the profile.
    assert np.std(batch.body.mass) > 0.02 and np.std(batch.max_rotor_speed) > 10.0


def test_the_profile_numbers_one_by_one_change_nothing():
    """Giving every drone the profile's own numbers switches the batch to one
    value per drone and leaves every force bit for bit as it was."""
    rng = np.random.default_rng(2)
    n = 50
    states = random_states(rng, n, 17)
    commands = rng.uniform(0.0, 1100.0, (n, 4))
    shared = BatchMultirotor(IRIS, n, motor_lag=True, ground_effect=GroundEffect())
    own = BatchMultirotor(IRIS, n, motor_lag=True, ground_effect=GroundEffect())
    values, _ = NONE.draw(IRIS, rng, n)
    own.set_parameters(np.arange(n), **values)
    assert own.per_drone and not shared.per_drone
    shared.rotor_speed_command = own.rotor_speed_command = commands
    np.testing.assert_array_equal(
        shared.derivative(0.0, states), own.derivative(0.0, states)
    )
    assert same_numbers(own.drone_params(3), IRIS)  # the profile itself


def test_a_drone_alone_or_in_a_batch_with_its_own_numbers():
    rng = np.random.default_rng(4)
    n = 40
    batch = BatchMultirotor(IRIS, n, motor_lag=True)
    values, _ = DRAWN.draw(IRIS, rng, n)
    batch.set_parameters(np.arange(n), **values)
    states = random_states(rng, n, 17)
    batch.rotor_speed_command = rng.uniform(0.0, 1100.0, (n, 4))
    together = batch.derivative(0.0, states)
    for i in (0, 17, 39):
        alone = BatchMultirotor(IRIS, 1, motor_lag=True)
        alone.set_parameters([0], **{k: v[i : i + 1] for k, v in values.items()})
        alone.rotor_speed_command = batch.rotor_speed_command[i : i + 1]
        np.testing.assert_array_equal(
            alone.derivative(0.0, states[i : i + 1])[0], together[i]
        )


def test_draws_stay_inside_their_ranges_and_repeat():
    values, latency = DRAWN.draw(IRIS, np.random.default_rng(0), 5000)
    assert set(values) == set(PER_DRONE)
    factor = values["mass"] / IRIS.mass
    assert 0.9 <= factor.min() < 0.91 and 1.09 < factor.max() <= 1.1
    inertia = values["inertia"] / IRIS.inertia[0, 0]
    assert np.all((inertia[:, 0, 0] >= 0.8) & (inertia[:, 0, 0] <= 1.2))
    # One factor for the whole tensor: still a physical inertia.
    np.testing.assert_allclose(
        values["inertia"][:, 2, 2] / values["inertia"][:, 0, 0],
        IRIS.inertia[2, 2] / IRIS.inertia[0, 0],
    )
    speed = values["max_rotor_speed"] / IRIS.max_rotor_speed
    assert 0.85 <= speed.min() and speed.max() <= 1.0
    # Log-uniform: as many halvings as doublings, the median at 1.
    drag = values["rotor_drag_coefficient"] / IRIS.rotor_drag_coefficient
    assert 0.5 <= drag.min() and drag.max() <= 2.0
    assert np.median(drag) == pytest.approx(1.0, abs=0.05)
    assert np.mean(drag < 0.71) == pytest.approx(np.mean(drag > 1.41), abs=0.03)
    up = values["motor_time_constant_up"] / IRIS.motor_time_constant_up
    down = values["motor_time_constant_down"] / IRIS.motor_time_constant_down
    np.testing.assert_allclose(up, down)  # one factor for both
    assert 0.0 <= latency.min() and latency.max() <= 0.04
    again, again_latency = DRAWN.draw(IRIS, np.random.default_rng(0), 5000)
    for key in values:
        np.testing.assert_array_equal(values[key], again[key])
    np.testing.assert_array_equal(latency, again_latency)
    # No randomisation: the profile's numbers, exactly.
    plain, no_latency = NONE.draw(IRIS, np.random.default_rng(1), 3)
    np.testing.assert_array_equal(plain["mass"], IRIS.mass)
    np.testing.assert_array_equal(plain["inertia"], np.tile(IRIS.inertia, (3, 1, 1)))
    np.testing.assert_array_equal(no_latency, 0.0)


def test_no_randomisation_is_the_environment_of_before():
    """Ranges that are single points: the same flight, observations and
    rewards bit for bit as without randomisation, restarts included."""
    n, steps = 3, 200
    actions = np.stack([script(i, steps) for i in range(n)], axis=1)
    plain = CourseEnv(COURSE, n=n, realism=Realism(motor_lag=True))
    drawn = CourseEnv(COURSE, n=n, realism=Realism(motor_lag=True, randomization=NONE))
    np.testing.assert_array_equal(plain.reset(seed=1)[0], drawn.reset(seed=1)[0])
    ends = 0
    for action in actions:
        a, b = plain.step(action), drawn.step(action)
        for x, y in zip(a[:4], b[:4], strict=True):
            np.testing.assert_array_equal(x, y)
        ends += int(np.sum(a[2] | a[3]))
    assert ends >= 2 and drawn.vehicles.per_drone


def reference_with_its_numbers(env: CourseEnv, actions: np.ndarray):
    """Drone 0 of `env` flown by the reference engine: the vehicle with its
    drawn numbers, the rate pilot with the profile's (it computes the rotor
    commands on a vehicle with the profile's numbers, which are then given to
    the real one), its drawn latency."""
    params = env.vehicles.drone_params(0)
    vehicle = Multirotor(
        params, contact=GroundContact(IRIS_LANDING_GEAR), motor_lag=True,
        ground_effect=GroundEffect(),
    )  # fmt: skip
    nominal = Multirotor(IRIS)
    pilot = RatePilot(nominal, IRIS_RATE_GAINS, DT, landed=True)
    referee = COURSE.referee(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    thrust, rates = LIMITS.commands(actions)
    delay = int(env.delays[0])
    sent = {"steps": 0}

    def pre_step(t, y):
        k = sent["steps"] - delay
        sent["steps"] += 1
        if k < 0:
            pilot.thrust_command, pilot.rate_command = 0.0, np.zeros(3)
        else:
            pilot.thrust_command, pilot.rate_command = thrust[k // 2], rates[k // 2]
        pilot(t, y)
        vehicle.rotor_speed_command = nominal.rotor_speed_command

    start = vehicle.initial_state(
        COURSE.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    )
    _, states = run(
        vehicle.derivative, start, DT, 2 * len(actions) * DT,
        stop=referee, pre_step=pre_step, post_step=normalize_attitude,
    )  # fmt: skip
    return states, referee


@pytest.mark.parametrize("seed", [0, 7])
def test_a_drawn_drone_flies_as_the_reference_with_its_numbers(seed):
    """A randomised episode (motor lag, ground effect, drawn numbers and
    latency): every state within 1e-9 of the reference engine flying that
    drone, whose controller keeps the profile's numbers."""
    realism = Realism(motor_lag=True, ground_effect=True, randomization=DRAWN)
    env = CourseEnv(COURSE, realism=realism)
    env.reset(seed=seed)
    actions = script(2, 400)
    states, referee = reference_with_its_numbers(env, actions)
    for k, action in enumerate(actions):
        _, _, terminated, truncated, _ = env.step(action[None])
        if terminated[0] or truncated[0]:
            break
        assert np.max(np.abs(env.flight.states[0] - states[2 * (k + 1)])) < 1e-9
    assert terminated[0] and referee.status == "crashed"


def test_a_restart_draws_only_the_restarted_drone_anew():
    realism = Realism(randomization=DRAWN)
    env = CourseEnv(COURSE, n=2, realism=realism)
    env.reset(seed=3)
    first = [env.vehicles.drone_params(i) for i in range(2)]
    delays = env.delays.copy()
    crash, hold = script(3, 400), script(0, 400)
    for k in range(400):
        terminated = env.step(np.stack([crash[k], hold[k]]))[2]
        if terminated[0]:
            break
    assert terminated[0] and not terminated[1]
    assert not same_numbers(env.vehicles.drone_params(0), first[0])
    assert same_numbers(env.vehicles.drone_params(1), first[1])
    assert env.delays[1] == delays[1]
    again = CourseEnv(COURSE, n=2, realism=realism)
    again.reset(seed=3)
    assert same_numbers(again.vehicles.drone_params(0), first[0])


def test_each_drone_draws_its_own_latency():
    """200 drones: latencies from 0 to 40 ms, in whole physics steps, all five
    present; the ring of commands is long enough for the longest."""
    env = CourseEnv(COURSE, n=200, realism=Realism(randomization=DRAWN))
    env.reset(seed=0)
    assert set(env.delays.tolist()) == {0, 1, 2, 3, 4}
    assert env.ring == 5


def test_rejects_bad_numbers():
    batch = BatchMultirotor(IRIS, 3, motor_lag=True)
    with pytest.raises(ValueError):
        batch.set_parameters([0], wingspan=[1.0])
    for name, bad in (
        ("mass", [0.0]),
        ("thrust_coefficient", [-1e-6]),
        ("max_rotor_speed", [0.0]),
        ("moment_constant", [0.0]),
        ("rotor_drag_coefficient", [-1.0]),
        ("body_drag_area", [[-0.1, 0.0, 0.0]]),
        ("motor_time_constant_up", [0.0]),  # the lag needs it > 0
        ("inertia", [np.diag([1.0, 1.0, 3.0])]),  # triangle inequality
    ):
        with pytest.raises(ValueError):
            batch.set_parameters([0], **{name: bad})
    with pytest.raises(ValueError):
        Randomization(mass=(1.1, 0.9))
    with pytest.raises(ValueError):
        Randomization(rotor_drag=(0.0, 2.0))
    with pytest.raises(ValueError):
        Randomization(latency=(-0.01, 0.02))


def test_the_phase_11_policy_on_drawn_drones():
    """The policy of phase 11 (one exact drone, no latency) flown by 256 drones
    drawn around the Iris (numbers and latency): measured, 9 % finish, and 6 %
    with the lab's other effects too (motor lag, ground effect, sensors)."""
    from jineta.doma.policy import Agent
    from jineta.picadero.training import fly

    agent = Agent.load("policies/first_course.npz")
    course = load_course("courses/first_course.yaml")
    plain = fly(
        agent,
        CourseEnv(course, n=256, realism=Realism(randomization=DRAWN)),
        seed=0,
    )
    lab = fly(agent, CourseEnv(course, n=256, realism=RANDOMISED_LAB), seed=0)
    assert 0.0 < np.mean(plain["status"] == 1) < 0.25
    assert np.mean(lab["status"] == 1) < 0.15
