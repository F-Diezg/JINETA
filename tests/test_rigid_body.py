"""Tests for the full 6-DOF rigid body."""

import numpy as np

from jineta.core.point_mass import GRAVITY
from jineta.core.rigid_body import RigidBody , normalize_attitude
from jineta.core.simulation import run


def test_no_forces_is_free_fall():
    body = RigidBody(mass=2.0, inertia=np.diag([1.0, 2.0, 3.0]))
    y0 = np.array([0, 0, -100.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    times, states = run(body.derivative, y0, dt=0.01, t_end=2.0)

    t = times[-1]
    assert np.isclose(states[-1, 2], -100.0 + 0.5 * GRAVITY * t**2, atol=1e-9)
    assert np.isclose(states[-1, 5], GRAVITY * t, atol=1e-9)
    assert np.allclose(states[-1, 0:2], 0.0)

def spinning_body_state():
    return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0.3, 5.0, 0.4])


def test_quaternion_norm_drifts_without_post_step():
    body = RigidBody(mass=1.0, inertia=np.diag([1.0, 2.0, 3.0]))
    _, states = run(body.derivative, spinning_body_state(), dt=0.1, t_end=100.0)
    assert abs(np.linalg.norm(states[-1, 6:10]) - 1.0) > 1e-4


def test_post_step_keeps_quaternion_unit_norm():
    body = RigidBody(mass=1.0, inertia=np.diag([1.0, 2.0, 3.0]))
    _, states = run(body.derivative, spinning_body_state(), dt=0.1, t_end=100.0,
                    post_step=normalize_attitude)
    norms = np.linalg.norm(states[:, 6:10], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-12)


def test_post_step_none_changes_nothing():
    body = RigidBody(mass=1.0, inertia=np.diag([1.0, 2.0, 3.0]))
    _, a = run(body.derivative, spinning_body_state(), dt=0.01, t_end=5.0)
    _, b = run(body.derivative, spinning_body_state(), dt=0.01, t_end=5.0, post_step=None)
    assert np.array_equal(a, b)   