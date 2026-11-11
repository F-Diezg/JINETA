"""Tests for the batch attitude and velocity loops and the batch autopilot (landed
state, take-off, land detection): every drone against the reference
(`jinete.control`), and the first course flown by drones of a batch."""

import numpy as np
import pytest

import examples.iris_course as iris_course
from jineta.campo.course import load_course
from jineta.core import quaternion
from jineta.core.integrators import rk4_step
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron import quaternion as batch_quaternion
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.control import (
    BatchAttitudeController,
    BatchAutopilot,
    BatchVelocityController,
)
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.rigid_body import normalize_attitudes
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import AttitudeController, Autopilot, VelocityController
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_ATTITUDE_GAINS,
    IRIS_GAINS,
    IRIS_LANDING_GEAR,
    IRIS_VELOCITY_GAINS,
)

DT = 0.01  # 100 Hz
REST = resting_height(IRIS_LANDING_GEAR, IRIS.mass)


def random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def awkward_quaternions(rng: np.random.Generator) -> np.ndarray:
    """Random attitudes, both signs of w, tiny rotations (the small-angle branch of
    the rotation vector) and half turns about each axis (w = 0, where Shepperd's
    method changes formula)."""
    random = random_quaternions(rng, 300)
    tiny = np.column_stack([np.ones(30), rng.normal(size=(30, 3)) * 1e-14])
    tiny[15:] *= -1.0
    half_turns = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    # Half turns about an axis tilted 1e-6 towards another: two diagonal terms of
    # the matrix differ by ~2e-12, and only the formula of the largest one is
    # accurate (the others divide by a root of a rounding error).
    near_half_turns = []
    for big in (1, 2, 3):
        for small in (1, 2, 3):
            if small != big:
                q = np.zeros(4)
                q[big], q[small] = 1.0, 1e-6
                near_half_turns.append(q / np.linalg.norm(q))
    return np.vstack(
        [random, -random[:50], tiny, half_turns, near_half_turns, np.eye(4)[:1]]
    )


# ---- quaternions -----------------------------------------------------------------


def test_quaternion_products_and_conjugates_match_reference():
    rng = np.random.default_rng(0)
    a, b = random_quaternions(rng, 300), random_quaternions(rng, 300)
    products = batch_quaternion.multiply(a, b)
    conjugates = batch_quaternion.conjugate(a)
    for i in range(300):
        assert np.array_equal(products[i], quaternion.multiply(a[i], b[i]))
        assert np.array_equal(conjugates[i], quaternion.conjugate(a[i]))


def test_rotation_vectors_match_reference():
    q = awkward_quaternions(np.random.default_rng(1))
    vectors = batch_quaternion.to_rotation_vector(q)
    for qi, vi in zip(q, vectors, strict=True):
        assert np.allclose(vi, quaternion.to_rotation_vector(qi), rtol=1e-14, atol=0)


def test_quaternions_from_matrices_match_reference_on_every_branch():
    q = awkward_quaternions(np.random.default_rng(2))
    matrices = np.array([quaternion.to_matrix(qi) for qi in q])
    recovered = batch_quaternion.from_matrix(matrices)
    for matrix, ri in zip(matrices, recovered, strict=True):
        assert np.allclose(ri, quaternion.from_matrix(matrix), rtol=0, atol=1e-15)
        assert ri[0] >= 0.0
    # every one of Shepperd's four formulas was used
    traces = np.trace(matrices, axis1=1, axis2=2)
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    assert np.any(traces > 0)
    assert {0, 1, 2} <= set(np.argmax(diagonal[traces <= 0], axis=1))


# ---- the loops, step by step ------------------------------------------------------


def test_attitude_loop_matches_reference():
    rng = np.random.default_rng(3)
    n = 400
    commands = awkward_quaternions(rng)[:n]
    states = np.zeros((len(commands), 13))
    states[:, 6:10] = random_quaternions(rng, len(commands))
    states[:50, 6:10] = commands[:50]  # no error at all
    rates = BatchAttitudeController(IRIS_ATTITUDE_GAINS).update(commands, states)
    reference = AttitudeController(IRIS_ATTITUDE_GAINS)
    clipped = 0
    for i in range(len(commands)):
        expected = reference.update(commands[i], states[i])
        assert np.allclose(rates[i], expected, rtol=1e-14, atol=1e-15)
        clipped += np.any(np.abs(expected) == IRIS_ATTITUDE_GAINS.rate_limit)
    assert clipped > 50


def flying_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """n drones in the air, tilted up to ~40 degrees, moving at a few m/s."""
    states = np.zeros((n, 13))
    states[:, 0:3] = rng.normal(size=(n, 3)) * 3.0
    states[:, 3:6] = rng.normal(size=(n, 3)) * 3.0
    tilt = rng.normal(size=(n, 3)) * np.array([0.35, 0.35, 2.0])
    states[:, 6:10] = [
        quaternion.from_euler(roll, pitch, yaw) for roll, pitch, yaw in tilt
    ]
    states[:, 10:13] = rng.normal(size=(n, 3))
    return states


def test_velocity_loop_matches_reference_step_by_step():
    """200 updates of 40 drones, commands from gentle to far beyond the tilt and
    thrust limits: attitude commands, thrusts and integrals equal the reference's."""
    rng = np.random.default_rng(4)
    n, steps = 40, 200
    batch = BatchVelocityController(IRIS, IRIS_VELOCITY_GAINS, DT, n)
    references = [VelocityController(IRIS, IRIS_VELOCITY_GAINS, DT) for _ in range(n)]
    for _ in range(steps):
        states = flying_states(rng, n)
        velocities = rng.normal(size=(n, 3)) * rng.choice([1.0, 20.0], size=(n, 1))
        yaws = rng.uniform(-np.pi, np.pi, size=n)
        attitudes, thrusts = batch.update(velocities, yaws, states)
        for i, reference in enumerate(references):
            attitude, thrust = reference.update(velocities[i], yaws[i], states[i])
            assert np.allclose(attitudes[i], attitude, rtol=0, atol=1e-14)
            assert thrusts[i] == pytest.approx(thrust, rel=1e-13, abs=1e-12)
            assert np.allclose(batch.integral[i], reference.integral, atol=1e-13)
    assert np.any(np.isclose(np.abs(batch.integral), 5.0))  # anti-windup reached


# ---- the autopilot in flight -------------------------------------------------------


def scenario_command(k: int, wait_steps: int) -> tuple[np.ndarray, float]:
    """The flight of one drone in the mixed batch, at step k: wait on the floor
    (asking to climb too slowly for the second half of the wait), climb at 1 m/s
    for 1 s, hold for 0.5 s turning to 60 degrees, then descend at 0.5 m/s until
    landed and beyond."""
    k = k - wait_steps
    if k < 0:
        # first standing still, then asking to climb, but slower than the take-off
        # speed (0.2 m/s): it must stay on the ground
        slow = np.array([0.0, 0.0, -0.15]) if k >= -wait_steps // 2 else np.zeros(3)
        return slow, 0.0
    if k < 100:
        return np.array([0.0, 0.0, -1.0]), 0.0
    if k < 150:
        return np.zeros(3), np.radians(60.0)
    return np.array([0.0, 0.0, 0.5]), np.radians(60.0)


WAITS = np.array([0, 50, 130, 200])  # steps on the floor before the take-off
STEPS = 750


def standing_starts(n: int) -> np.ndarray:
    states = np.zeros((n, 13))
    states[:, 0] = 2.0 * np.arange(n)
    states[:, 2] = -REST
    states[:, 6] = 1.0
    return states


def fly_mixed_batch(waits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The batch flies every drone's scenario at once; returns the states
    (steps + 1, n, 13) and `landed` before each step (steps, n)."""
    n = len(waits)
    vehicles = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    autopilot = BatchAutopilot(vehicles, IRIS_GAINS, DT, landed=True)
    states = standing_starts(n)
    history, landed = [states], []
    for k in range(STEPS):
        for i, wait in enumerate(waits):
            command, yaw = scenario_command(k, wait)
            autopilot.velocity_command[i] = command
            autopilot.yaw_command[i] = yaw
        landed.append(autopilot.landed.copy())
        autopilot(k * DT, states)
        states = normalize_attitudes(rk4_step(vehicles.derivative, k * DT, states, DT))
        history.append(states)
    return np.array(history), np.array(landed)


def fly_reference(start: np.ndarray, wait: int) -> tuple[np.ndarray, np.ndarray]:
    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    autopilot = Autopilot(vehicle, IRIS_GAINS, DT, landed=True)
    landed = []
    k = 0

    def pre_step(t, state):
        nonlocal k
        autopilot.velocity_command, autopilot.yaw_command = scenario_command(k, wait)
        landed.append(autopilot.landed)
        autopilot(t, state)
        k += 1

    _, states = run(
        vehicle.derivative,
        start,
        DT,
        STEPS * DT,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    return states, np.array(landed)


def test_mixed_batch_of_take_offs_and_landings_matches_reference():
    """Four drones standing on the floor take off at different moments, climb,
    turn, descend and land, all in one batch: each follows its reference flight
    within 1e-9 and is landed at the same steps."""
    history, landed = fly_mixed_batch(WAITS)
    starts = standing_starts(len(WAITS))
    for i, wait in enumerate(WAITS):
        states, reference_landed = fly_reference(starts[i], wait)
        assert np.array_equal(landed[:, i], reference_landed)
        assert np.allclose(history[:, i], states, rtol=0.0, atol=1e-9)
    # really a mix: at some step some drones were landed and others flying
    assert np.any(landed.any(axis=1) & ~landed.all(axis=1))
    # every drone took off and landed again
    took_off = np.argmax(~landed, axis=0)
    assert np.all(took_off == WAITS + 1)
    assert np.all(landed[-1])


def test_a_drone_gives_the_same_numbers_alone_or_in_a_batch():
    everyone, landed = fly_mixed_batch(WAITS)
    alone, alone_landed = fly_mixed_batch(WAITS[2:3])
    # the same flight, only started 4 m further North: all but x bit for bit
    assert np.array_equal(alone[:, 0, 1:], everyone[:, 2, 1:])
    assert np.array_equal(alone_landed[:, 0], landed[:, 2])


def test_reset_of_some_drones_keeps_whether_they_are_landed():
    vehicles = BatchMultirotor(IRIS, 3)
    autopilot = BatchAutopilot(vehicles, IRIS_GAINS, DT, landed=[True, False, False])
    autopilot.velocity_loop.integral[:] = 1.0
    autopilot.rate_loop.integral[:] = 1.0
    autopilot.still_time[:] = 0.3
    autopilot.reset([1])
    assert np.array_equal(autopilot.landed, [True, False, False])
    for memory in (autopilot.velocity_loop.integral, autopilot.rate_loop.integral):
        assert np.all(memory[1] == 0.0)
        assert np.all(memory[[0, 2]] == 1.0)
    assert np.array_equal(autopilot.still_time, [0.3, 0.0, 0.3])


def test_drones_on_the_ground_keep_their_loops_at_zero_as_the_reference():
    """After a call, a drone that stays landed has its integrators and landing
    timer at zero and its motors stopped, as the reference's early return leaves
    them, also when the drone next to it is flying."""
    vehicles = BatchMultirotor(IRIS, 2, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    autopilot = BatchAutopilot(vehicles, IRIS_GAINS, DT, landed=[True, False])
    states = standing_starts(2)
    states[1, 2] = -2.0  # the second one is in the air
    autopilot.velocity_command[:] = [1.0, 0.0, 0.0]  # not a climb: stays landed
    autopilot(0.0, states)
    assert np.all(autopilot.velocity_loop.integral[0] == 0.0)
    assert np.all(autopilot.rate_loop.integral[0] == 0.0)
    assert autopilot.still_time[0] == 0.0
    assert np.all(vehicles.rotor_speed_command[0] == 0.0)
    assert np.any(autopilot.velocity_loop.integral[1] != 0.0)
    assert np.all(vehicles.rotor_speed_command[1] > 0.0)


def test_a_drone_declared_landed_in_the_air_takes_off_from_clean_loops():
    """Flying with charged integrators, a drone is set to landed and asked to climb
    in the same step: like the reference, it forgets its integrators first."""
    rng = np.random.default_rng(5)
    start = flying_states(rng, 1)[0]
    reference_vehicle = Multirotor(IRIS)
    reference = Autopilot(reference_vehicle, IRIS_GAINS, DT)
    vehicles = BatchMultirotor(IRIS, 1)
    batch = BatchAutopilot(vehicles, IRIS_GAINS, DT)
    reference.velocity_command = np.array([3.0, -2.0, 0.0])
    batch.velocity_command[0] = [3.0, -2.0, 0.0]
    reference(0.0, start)
    batch(0.0, start[None])  # integrators charged
    reference.landed = True
    batch.landed[0] = True
    reference.velocity_command = np.array([0.0, 0.0, -1.0])
    batch.velocity_command[0] = [0.0, 0.0, -1.0]
    reference(0.01, start)
    batch(0.01, start[None])
    assert not batch.landed[0]
    assert np.allclose(
        vehicles.rotor_speed_command[0],
        reference_vehicle.rotor_speed_command,
        rtol=1e-13,
    )
    assert np.allclose(
        batch.velocity_loop.integral[0], reference.velocity_loop.integral, atol=1e-15
    )


def test_landed_accepts_one_value_for_all():
    autopilot = BatchAutopilot(BatchMultirotor(IRIS, 5), IRIS_GAINS, DT, landed=True)
    assert np.array_equal(autopilot.landed, np.ones(5, dtype=bool))
    assert np.array_equal(autopilot.velocity_command, np.zeros((5, 3)))
    assert np.array_equal(autopilot.yaw_command, np.zeros(5))


# ---- the first course --------------------------------------------------------------


def record_reference_lap(monkeypatch):
    """Fly the first course with the reference engine (examples.iris_course),
    keeping the referee's next gate seen by the guidance at every step."""
    next_gates = []
    reference_run = iris_course.run

    def recording_run(f, y0, dt, t_end, stop, pre_step, post_step):
        def recording_pre_step(t, y):
            next_gates.append(stop.next_gate)  # stop is the referee
            pre_step(t, y)

        return reference_run(f, y0, dt, t_end, stop, post_step, recording_pre_step)

    monkeypatch.setattr(iris_course, "run", recording_run)
    course = load_course(iris_course.COURSE_FILE)
    _, states, referee = iris_course.fly_course(course)
    assert referee.status == "finished"
    return course, states, next_gates


def test_first_course_flown_in_a_batch(monkeypatch):
    """Three drones of a batch fly the lap of examples/iris_course.py (take-off
    from the floor, 3 m/s straight at each gate centre), with the gates in the
    order the reference referee gave them: each follows the reference lap within
    1e-9 to its last step, and the three are identical bit for bit."""
    course, reference, next_gates = record_reference_lap(monkeypatch)
    n, speed = 3, 3.0
    vehicles = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    autopilot = BatchAutopilot(vehicles, IRIS_GAINS, DT, landed=True)
    states = np.tile(reference[0], (n, 1))
    climbed = np.zeros(n, dtype=bool)
    largest = 0.0
    for k, gate in enumerate(next_gates):
        climbed = climbed | (-states[:, 2] > iris_course.TAKEOFF_ALTITUDE)
        if gate is None:
            command = np.zeros((n, 3))
        else:
            to_gate = course.gates[gate].center - states[:, 0:3]
            distance = np.linalg.norm(to_gate, axis=1, keepdims=True)
            command = np.where(
                climbed[:, None], speed * to_gate / distance, [0.0, 0.0, -1.0]
            )
            autopilot.yaw_command[:] = course.gates[gate].yaw
        autopilot.velocity_command = command
        autopilot(k * DT, states)
        states = normalize_attitudes(rk4_step(vehicles.derivative, k * DT, states, DT))
        largest = max(largest, np.max(np.abs(states - reference[k + 1])))
        assert np.array_equal(states[1:], np.repeat(states[:1], n - 1, axis=0))
    assert len(next_gates) == len(reference) - 1
    assert largest < 1e-9
