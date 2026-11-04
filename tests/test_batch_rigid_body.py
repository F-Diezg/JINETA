"""Tests for the batch rigid body: every body of a batch against the reference
engine (`core.rigid_body`), which stays the truth, and the rows independent of
each other."""

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.integrators import rk4_step
from jineta.core.rigid_body import RigidBody, normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron import quaternion as batch_quaternion
from jineta.escuadron.rigid_body import BatchRigidBody, normalize_attitudes

MASS = 1.5  # [kg]
# A full inertia tensor, products of inertia included, so that no term of Euler's
# equations is zero [kg·m^2].
INERTIA = np.array(
    [
        [0.030, 0.001, -0.002],
        [0.001, 0.050, 0.0005],
        [-0.002, 0.0005, 0.070],
    ]
)


def random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """States (n, 13): positions and velocities of a few metres (per second), any
    attitude, angular rates up to a few rad/s."""
    states = np.zeros((n, 13))
    states[:, 0:3] = rng.normal(size=(n, 3)) * 5.0
    states[:, 3:6] = rng.normal(size=(n, 3)) * 3.0
    states[:, 6:10] = random_quaternions(rng, n)
    states[:, 10:13] = rng.normal(size=(n, 3)) * 3.0
    return states


def test_rotation_matrices_match_reference():
    rng = np.random.default_rng(0)
    q = random_quaternions(rng, 500)
    matrices = batch_quaternion.to_matrix(q)
    assert matrices.shape == (500, 3, 3)
    for qi, matrix in zip(q, matrices, strict=True):
        assert np.allclose(matrix, quaternion.to_matrix(qi), rtol=0.0, atol=1e-15)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-14)


def test_quaternion_derivatives_match_reference_bit_for_bit():
    rng = np.random.default_rng(1)
    q = random_quaternions(rng, 500)
    omega = rng.normal(size=(500, 3)) * 5.0
    q_dot = batch_quaternion.derivative(q, omega)
    for qi, wi, di in zip(q, omega, q_dot, strict=True):
        assert np.array_equal(di, quaternion.derivative(qi, wi))


def test_normalize_matches_reference():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(500, 4)) * 10.0 ** rng.uniform(-3, 3, size=(500, 1))
    unit = batch_quaternion.normalize(q)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-15)
    for qi, ui in zip(q, unit, strict=True):
        assert np.allclose(ui, quaternion.normalize(qi), rtol=0.0, atol=1e-15)


def test_normalize_attitudes_touches_only_the_quaternions():
    rng = np.random.default_rng(3)
    states = random_states(rng, 20)
    states[:, 6:10] *= 1.01  # drifted norm
    fixed = normalize_attitudes(states)
    assert np.array_equal(fixed[:, 0:6], states[:, 0:6])
    assert np.array_equal(fixed[:, 10:13], states[:, 10:13])
    assert np.allclose(np.linalg.norm(fixed[:, 6:10], axis=1), 1.0, atol=1e-15)
    assert not np.shares_memory(fixed, states)  # a copy, like the reference


def test_derivative_matches_reference_for_every_body():
    """Each body with its own force and moment, against a reference RigidBody
    with the same ones; relative to the size of each component."""
    rng = np.random.default_rng(4)
    n = 500
    states = random_states(rng, n)
    forces = rng.normal(size=(n, 3)) * 20.0
    moments = rng.normal(size=(n, 3))

    batch = BatchRigidBody(MASS, INERTIA, lambda t, s: (forces, moments))
    derivatives = batch.derivative(0.0, states)
    assert derivatives.shape == (n, 13)

    for i in range(n):
        body = RigidBody(
            MASS, INERTIA, lambda t, s, i=i: (forces[i].copy(), moments[i].copy())
        )
        expected = body.derivative(0.0, states[i])
        assert np.allclose(derivatives[i], expected, rtol=1e-12, atol=1e-12)


def test_a_body_gives_the_same_numbers_alone_or_in_a_batch():
    """No cross-talk: a row depends only on its own state, bit for bit."""
    rng = np.random.default_rng(5)
    states = random_states(rng, 1000)
    forces = rng.normal(size=(1000, 3)) * 20.0
    moments = rng.normal(size=(1000, 3))

    def model(rows: slice):
        return lambda t, s: (forces[rows], moments[rows])

    everyone = BatchRigidBody(MASS, INERTIA, model(slice(None))).derivative(0.0, states)
    for i in (0, 1, 499, 998, 999):
        alone = BatchRigidBody(MASS, INERTIA, model(slice(i, i + 1))).derivative(
            0.0, states[i : i + 1]
        )
        assert np.array_equal(alone[0], everyone[i])


def batch_damping(t, states):
    """A force and a moment that depend on the state: drag on the body-axis
    velocity, a damped twist and a time-varying roll moment."""
    rotation = batch_quaternion.to_matrix(states[:, 6:10])
    velocity_body = np.einsum("nji,nj->ni", rotation, states[:, 3:6])
    force = -0.4 * velocity_body
    moment = -0.02 * states[:, 10:13] + 0.005 * states[:, 7:10]
    moment[:, 0] += 0.003 * np.cos(t)
    return force, moment


def reference_damping(t, state):
    rotation = quaternion.to_matrix(state[6:10])
    force = -0.4 * (rotation.T @ state[3:6])
    moment = -0.02 * state[10:13] + 0.005 * state[7:10]
    moment[0] += 0.003 * np.cos(t)
    return force, moment


def test_flight_with_rk4_matches_reference():
    """Two seconds of RK4 at 100 Hz with renormalisation, forces depending on the
    state: every body follows its reference flight to rounding."""
    rng = np.random.default_rng(6)
    n, dt, steps = 40, 0.01, 200
    start = random_states(rng, n)

    batch = BatchRigidBody(MASS, INERTIA, batch_damping)
    states = start.copy()
    t = 0.0
    for _ in range(steps):
        states = normalize_attitudes(rk4_step(batch.derivative, t, states, dt))
        t += dt

    body = RigidBody(MASS, INERTIA, reference_damping)
    for i in range(n):
        _, history = run(
            body.derivative,
            start[i],
            dt,
            steps * dt,
            post_step=normalize_attitude,
        )
        assert np.allclose(states[i], history[-1], rtol=0.0, atol=1e-12)


def test_free_fall_is_exact_for_tumbling_bodies():
    """No forces: every body falls g t^2 / 2 whatever its spin (the reference's
    decoupling test, for a whole batch)."""
    rng = np.random.default_rng(7)
    n, dt, steps = 30, 0.01, 200
    states = random_states(rng, n)
    start = states.copy()
    batch = BatchRigidBody(MASS, INERTIA)
    for _ in range(steps):
        states = normalize_attitudes(rk4_step(batch.derivative, 0.0, states, dt))

    t = steps * dt
    expected = start[:, 0:3] + start[:, 3:6] * t
    expected[:, 2] += 0.5 * GRAVITY * t**2
    assert np.allclose(states[:, 0:3], expected, atol=1e-9)
    assert np.allclose(states[:, 5], start[:, 5] + GRAVITY * t, atol=1e-9)


def test_time_reaches_the_force_model_as_given():
    """One time per drone (each drone on its own clock) reaches the model as is."""
    seen = []

    def model(t, states):
        seen.append(t)
        return np.zeros((len(states), 3)), np.zeros((len(states), 3))

    times = np.array([0.0, 1.5, 3.0])
    BatchRigidBody(MASS, INERTIA, model).derivative(times, np.zeros((3, 13)))
    assert seen[0] is times


@pytest.mark.parametrize("mass", [0.0, -1.0])
def test_rejects_non_positive_mass(mass):
    with pytest.raises(ValueError, match="mass"):
        BatchRigidBody(mass, INERTIA)


def test_rejects_non_physical_inertia_and_keeps_it_read_only():
    with pytest.raises(ValueError):
        BatchRigidBody(MASS, np.diag([1.0, 1.0, 3.0]))  # triangle inequality
    batch = BatchRigidBody(MASS, INERTIA)
    with pytest.raises(ValueError):
        batch.inertia[0, 0] = 1.0
    with pytest.raises(ValueError):
        batch.inertia_inv[0, 0] = 1.0
