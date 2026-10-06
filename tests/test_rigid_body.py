"""Tests for the full 6-DOF rigid body."""

import numpy as np

from jineta.core.point_mass import GRAVITY
from jineta.core.rigid_body import RigidBody , normalize_attitude
from jineta.core.simulation import run
from jineta.core import quaternion
import pytest


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

@pytest.mark.parametrize(
    "yaw_deg, expected_horizontal",
    [
        (0.0, [5.0, 0.0]),      # nose North -> thrust North (+x)
        (90.0, [0.0, 5.0]),     # nose East  -> thrust East  (+y)
        (180.0, [-5.0, 0.0]),   # nose South -> thrust South (-x)
    ],
)
def test_thrust_follows_attitude(yaw_deg, expected_horizontal):
    """Body-axis thrust along +x_body must be rotated to NED by the attitude."""
    mass = 2.0
    thrust = 10.0  # [N] along +x_body

    def force_moment(t, state):
        return np.array([thrust, 0.0, 0.0]), np.zeros(3)

    body = RigidBody(mass=mass, inertia=np.eye(3), force_moment=force_moment)
    q = quaternion.from_axis_angle([0, 0, 1], np.radians(yaw_deg))
    state = np.concatenate([np.zeros(3), np.zeros(3), q, np.zeros(3)])

    accel = body.derivative(0.0, state)[3:6]

    assert np.allclose(accel, [*expected_horizontal, GRAVITY])

def test_free_spin_does_not_affect_center_of_mass_fall():
    """With no external force, spinning must not change how the body falls."""
    body = RigidBody(mass=2.0, inertia=np.diag([1.0, 2.0, 3.0]))
    still = np.array([0, 0, -100.0, 3.0, -2.0, 0.0, 1, 0, 0, 0, 0, 0, 0])
    tumbling = np.array([0, 0, -100.0, 3.0, -2.0, 0.0, 1, 0, 0, 0, 0.3, 5.0, 0.4])

    _, s_still = run(body.derivative, still, dt=0.01, t_end=5.0)
    _, s_tumbling = run(body.derivative, tumbling, dt=0.01, t_end=5.0)

    assert np.allclose(s_still[:, 0:6], s_tumbling[:, 0:6], atol=1e-9)