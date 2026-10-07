"""Tests for inertia tensor validation and its use in the bodies."""

import numpy as np
import pytest

from jineta.core.inertia import inertia_tensor
from jineta.core.rigid_body import RigidBody
from jineta.core.rotating_body import RotatingBody


def test_valid_tensor_is_returned_as_a_read_only_copy():
    original = np.diag([1.0, 2.0, 2.5])
    tensor = inertia_tensor(original)

    assert np.array_equal(tensor, original)
    assert tensor is not original
    with pytest.raises(ValueError):
        tensor[0, 0] = 5.0  # read-only: in-place changes are refused


def test_accepts_lists_and_products_of_inertia():
    tensor = inertia_tensor([[2.0, -0.1, 0.0], [-0.1, 3.0, 0.2], [0.0, 0.2, 4.0]])
    assert tensor.shape == (3, 3)


def test_flat_plate_limit_is_accepted():
    # Thin flat plate: I_z = I_x + I_y exactly
    inertia_tensor(np.diag([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "bad_inertia",
    [
        np.eye(2),  # not 3x3
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # not symmetric
        np.diag([1.0, 0.0, 1.0]),  # zero principal moment
        np.diag([1.0, -2.0, 3.0]),  # negative principal moment
        np.diag([1.0, 1.0, 3.0]),  # 1 + 1 < 3: breaks the triangle inequality
    ],
)
def test_rejects_non_physical_tensors(bad_inertia):
    with pytest.raises(ValueError):
        inertia_tensor(bad_inertia)


def test_bodies_validate_their_inertia():
    with pytest.raises(ValueError):
        RotatingBody(inertia=np.diag([1.0, 1.0, 3.0]))
    with pytest.raises(ValueError):
        RigidBody(mass=1.0, inertia=np.diag([1.0, 1.0, 3.0]))


def test_body_inertia_cannot_be_changed_in_place():
    body = RigidBody(mass=1.0, inertia=np.diag([1.0, 2.0, 2.5]))
    with pytest.raises(ValueError):
        body.inertia[2, 2] = 10.0  # would leave inertia_inv stale
