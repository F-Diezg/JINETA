"""Tests for the batch multirotor and the batch floor contact: every drone against
the reference vehicle (`jinete.multirotor`, `jinete.contact`), a lap of the first
course replayed, and the rows independent of each other."""

import numpy as np
import pytest

import examples.iris_course as iris_course
from jineta.campo.course import load_course
from jineta.campo.wind import ConstantWind
from jineta.core.integrators import rk4_step
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.contact import contact_forces as batch_contact_forces
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.rigid_body import normalize_attitudes
from jineta.jinete.contact import GroundContact, contact_forces, resting_height
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR


def random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def states_near_the_floor(rng: np.random.Generator, n: int) -> np.ndarray:
    """States (n, 13) with the centre of mass from 0.3 m up to 0.1 m below the
    floor, any attitude, a few m/s and rad/s: some feet touch, some do not, some
    move up fast enough for the floor to stop pushing."""
    states = np.zeros((n, 13))
    states[:, 0:2] = rng.normal(size=(n, 2)) * 5.0
    states[:, 2] = rng.uniform(-0.3, 0.1, size=n)
    states[:, 3:6] = rng.normal(size=(n, 3)) * 2.0
    states[:, 6:10] = random_quaternions(rng, n)
    states[:, 10:13] = rng.normal(size=(n, 3)) * 2.0
    return states


def random_commands(rng: np.random.Generator, n: int) -> np.ndarray:
    """Rotor speeds (n, 4) from below zero to above the maximum (both clipped)."""
    return rng.uniform(-100.0, 1.1 * IRIS.max_rotor_speed + 100.0, size=(n, 4))


def exponential_density(altitude):
    """A density that changes with height, written for arrays (8.5 km scale)."""
    return 1.225 * np.exp(-np.asarray(altitude) / 8500.0)


# ---- floor contact ---------------------------------------------------------------


@pytest.mark.parametrize("floor_z", [0.0, -0.05])
def test_contact_matches_reference_for_every_drone(floor_z):
    rng = np.random.default_rng(0)
    states = states_near_the_floor(rng, 400)
    states[:, 2] += floor_z
    forces, moments = batch_contact_forces(IRIS_LANDING_GEAR, states, floor_z)
    touching = 0
    for i, state in enumerate(states):
        force, moment = contact_forces(IRIS_LANDING_GEAR, state, floor_z)
        assert np.allclose(forces[i], force, rtol=1e-12, atol=1e-12)
        assert np.allclose(moments[i], moment, rtol=1e-12, atol=1e-12)
        touching += np.any(force != 0.0)
    assert 40 < touching < len(states) - 40  # both cases really tested


def test_no_contact_force_in_the_air():
    states = np.zeros((5, 13))
    states[:, 2] = -1.0  # 1 m up
    states[:, 6] = 1.0
    forces, moments = BatchGroundContact(IRIS_LANDING_GEAR)(0.0, states)
    assert np.array_equal(forces, np.zeros((5, 3)))
    assert np.array_equal(moments, np.zeros((5, 3)))


# ---- the whole vehicle -----------------------------------------------------------


@pytest.mark.parametrize(
    ("density", "wind"),
    [
        (None, None),  # the defaults: sea-level density, still air
        (exponential_density, ConstantWind([3.0, -2.0, 0.5])),
    ],
)
def test_derivative_matches_reference_for_every_drone(density, wind):
    rng = np.random.default_rng(1)
    n = 400
    states = states_near_the_floor(rng, n)
    states[: n // 2, 2] -= 200.0  # half of them high up (density changes)
    commands = random_commands(rng, n)
    environment = {}
    if density is not None:
        environment = {"density": density, "wind": wind}

    batch = BatchMultirotor(
        IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR), **environment
    )
    batch.rotor_speed_command = commands
    derivatives = batch.derivative(0.0, states)

    for i in range(n):
        vehicle = Multirotor(
            IRIS, contact=GroundContact(IRIS_LANDING_GEAR), **environment
        )
        vehicle.rotor_speed_command = commands[i]
        expected = vehicle.derivative(0.0, states[i])
        assert np.allclose(derivatives[i], expected, rtol=1e-12, atol=1e-12)


def test_derivative_without_contact_matches_reference():
    rng = np.random.default_rng(2)
    states = states_near_the_floor(rng, 200)
    commands = random_commands(rng, 200)
    batch = BatchMultirotor(IRIS, 200)
    batch.rotor_speed_command = commands
    derivatives = batch.derivative(0.0, states)
    for i in range(200):
        vehicle = Multirotor(IRIS)
        vehicle.rotor_speed_command = commands[i]
        assert np.allclose(
            derivatives[i], vehicle.derivative(0.0, states[i]), rtol=1e-12, atol=1e-12
        )


def test_a_drone_gives_the_same_numbers_alone_or_in_a_batch():
    rng = np.random.default_rng(3)
    n = 1000
    states = states_near_the_floor(rng, n)
    commands = random_commands(rng, n)
    contact = BatchGroundContact(IRIS_LANDING_GEAR)
    everyone = BatchMultirotor(IRIS, n, contact=contact)
    everyone.rotor_speed_command = commands
    derivatives = everyone.derivative(0.0, states)
    for i in (0, 1, 500, 999):
        alone = BatchMultirotor(IRIS, 1, contact=contact)
        alone.rotor_speed_command = commands[i : i + 1]
        alone_derivative = alone.derivative(0.0, states[i : i + 1])
        assert np.array_equal(alone_derivative[0], derivatives[i])


def test_iris_standing_on_its_gear_is_at_rest():
    """The reference's resting state, for a whole batch: motors off, level, the
    centre of mass at the resting height; nothing moves (1e-12)."""
    n = 10
    states = np.zeros((n, 13))
    states[:, 0:2] = np.arange(n)[:, None]  # anywhere on the floor
    states[:, 2] = -resting_height(IRIS_LANDING_GEAR, IRIS.mass)
    states[:, 6] = 1.0
    batch = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    assert np.allclose(batch.derivative(0.0, states), 0.0, atol=1e-12)


def test_commands_start_at_zero_one_row_per_drone():
    batch = BatchMultirotor(IRIS, 7)
    assert np.array_equal(batch.rotor_speed_command, np.zeros((7, 4)))


def test_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="n"):
        BatchMultirotor(IRIS, 0)


# ---- a lap of the first course, replayed -----------------------------------------


def record_reference_lap(monkeypatch):
    """Fly the first course with the reference engine (examples.iris_course) and
    keep the rotor speed commands of every step, as the autopilot left them."""
    commands = []
    run = iris_course.run

    def recording_run(f, y0, dt, t_end, stop, pre_step, post_step):
        vehicle = f.__self__  # f is the bound method vehicle.derivative

        def recording_pre_step(t, y):
            pre_step(t, y)
            commands.append(vehicle.rotor_speed_command.copy())

        return run(f, y0, dt, t_end, stop, post_step, recording_pre_step)

    monkeypatch.setattr(iris_course, "run", recording_run)
    course = load_course(iris_course.COURSE_FILE)
    _, states, referee = iris_course.fly_course(course)
    assert referee.status == "finished"
    return states, np.array(commands)


def test_reference_lap_replayed_in_a_batch(monkeypatch):
    """The 882 rotor commands of the reference lap, given to three drones of a
    batch (open loop, with the floor contact): each follows the reference states
    within 1e-9 m (and m/s, rad/s), and the three are identical bit for bit."""
    reference, commands = record_reference_lap(monkeypatch)
    n = 3
    batch = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    states = np.tile(reference[0], (n, 1))
    largest = 0.0
    for k, command in enumerate(commands):
        batch.rotor_speed_command = np.tile(command, (n, 1))
        states = normalize_attitudes(rk4_step(batch.derivative, k * 0.01, states, 0.01))
        largest = max(largest, np.max(np.abs(states - reference[k + 1])))
        assert np.array_equal(states[1:], states[:1].repeat(n - 1, axis=0))
    assert len(commands) == len(reference) - 1
    assert largest < 1e-9
