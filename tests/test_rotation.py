"""Tests for quaternion utilities and rotational dynamics."""

import numpy as np

from jineta.core import quaternion
from jineta.core.rotating_body import RotatingBody
from jineta.core.simulation import run


def test_90deg_about_z_maps_north_to_east():
    q = quaternion.from_axis_angle([0, 0, 1], np.pi / 2)
    rotated = quaternion.to_matrix(q) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [0.0, 1.0, 0.0])


def test_rotation_matrix_is_proper_orthogonal():
    q = quaternion.from_axis_angle([1, 2, 3], 0.7)
    R = quaternion.to_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_steady_spin_about_principal_axis():
    body = RotatingBody(inertia=np.diag([1.0, 2.0, 3.0]))
    rate = 0.5  # [rad/s] about body z
    y0 = np.array([1.0, 0, 0, 0, 0, 0, rate])
    _, states = run(body.derivative, y0, dt=0.01, t_end=4.0)

    expected = quaternion.from_axis_angle([0, 0, 1], rate * 4.0)
    assert np.allclose(states[-1, 0:4], expected, atol=1e-8)


def test_constant_torque_spins_up_linearly():
    def torque(t, state):
        return np.array([0.0, 0.0, 1.0])

    body = RotatingBody(inertia=np.diag([2.0, 2.0, 2.0]), torque=torque)
    y0 = np.array([1.0, 0, 0, 0, 0, 0, 0])
    _, states = run(body.derivative, y0, dt=0.01, t_end=3.0)

    assert np.isclose(states[-1, 6], 1.0 / 2.0 * 3.0)  # r = M / I * t


def test_torque_free_motion_conserves_energy_and_momentum():
    inertia = np.diag([1.0, 2.0, 3.0])
    body = RotatingBody(inertia=inertia)
    y0 = np.array([1.0, 0, 0, 0, 0.1, 2.0, 0.1])  # near the intermediate axis
    _, states = run(body.derivative, y0, dt=0.01, t_end=20.0)

    def energy(s):
        omega = s[4:7]
        return 0.5 * omega @ inertia @ omega

    def momentum_ned(s):
        R = quaternion.to_matrix(quaternion.normalize(s[0:4]))
        return R @ (inertia @ s[4:7])

    assert np.isclose(energy(states[-1]), energy(states[0]), rtol=1e-6)
    assert np.allclose(momentum_ned(states[-1]), momentum_ned(states[0]), atol=1e-5)
    assert abs(np.linalg.norm(states[-1, 0:4]) - 1.0) < 1e-6
