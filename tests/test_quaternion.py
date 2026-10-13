"""Tests for the quaternion conjugate and the rotation vector (attitude errors)."""

import numpy as np
import pytest

from jineta.core import quaternion

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_conjugate_is_the_inverse_rotation():
    q = quaternion.from_axis_angle([1, 2, 3], 0.7)
    assert np.allclose(quaternion.multiply(q, quaternion.conjugate(q)), IDENTITY)
    R = quaternion.to_matrix(q)
    assert np.allclose(quaternion.to_matrix(quaternion.conjugate(q)), R.T)


@pytest.mark.parametrize("angle", [0.3, 2.0, np.pi - 0.01])
def test_rotation_vector_round_trip(angle):
    axis = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
    q = quaternion.from_axis_angle(axis, angle)
    assert np.allclose(quaternion.to_rotation_vector(q), angle * axis)


def test_rotation_vector_takes_the_short_way():
    """350 degrees about x is the same attitude as -10 degrees about x."""
    q = quaternion.from_axis_angle([1, 0, 0], np.radians(350.0))
    assert q[0] < 0  # the "long way round" quaternion
    assert np.allclose(quaternion.to_rotation_vector(q), [np.radians(-10.0), 0, 0])


def test_q_and_minus_q_give_the_same_rotation_vector():
    q = quaternion.from_axis_angle([0, 1, 1], 1.2)
    assert np.allclose(
        quaternion.to_rotation_vector(q), quaternion.to_rotation_vector(-q)
    )


def test_rotation_vector_of_a_tiny_rotation():
    q = quaternion.from_axis_angle([0, 0, 1], 1e-14)
    assert np.allclose(quaternion.to_rotation_vector(q), [0, 0, 1e-14], atol=1e-20)
    assert np.array_equal(quaternion.to_rotation_vector(IDENTITY), np.zeros(3))



def test_from_matrix_round_trip():
    rng = np.random.default_rng(seed=0)
    for _ in range(100):
        q = quaternion.from_axis_angle(rng.normal(size=3), rng.uniform(0, 2 * np.pi))
        q = q if q[0] >= 0 else -q  # from_matrix returns the w >= 0 representative
        assert np.allclose(quaternion.from_matrix(quaternion.to_matrix(q)), q)


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
def test_from_matrix_half_turns(axis):
    """180 degrees: w = 0, so the method must divide by x, y or z instead."""
    R = quaternion.to_matrix(quaternion.from_axis_angle(axis, np.pi))
    q = quaternion.from_matrix(R)
    assert np.allclose(quaternion.to_matrix(q), R)
    assert np.isclose(np.linalg.norm(q), 1.0)