"""Validation of inertia tensors, shared by RotatingBody and RigidBody."""

import numpy as np
import numpy.typing as npt


def inertia_tensor(inertia: npt.ArrayLike) -> np.ndarray:
    """Return a validated, read-only copy of a 3x3 inertia tensor [kg·m^2].

    A physical inertia tensor must be:
    - symmetric (I_xy = I_yx, ...),
    - positive definite (every principal moment > 0),
    - and its principal moments must satisfy the triangle inequality
      I_a + I_b >= I_c: no body can have one principal moment larger than the sum
      of the other two (equality is the limit of a flat plate).

    The copy is read-only because the bodies compute its inverse only once: changing
    the tensor in place would silently leave a stale inverse behind.
    """
    tensor = np.array(inertia, dtype=float)  # np.array copies, np.asarray may not
    if tensor.shape != (3, 3):
        raise ValueError("inertia must be a 3x3 matrix")
    if not np.allclose(tensor, tensor.T):
        raise ValueError("inertia must be symmetric")

    principal = np.linalg.eigvalsh(tensor)  # eigenvalues of a symmetric matrix, sorted
    if principal[0] <= 0.0:
        raise ValueError("inertia must be positive definite (principal moments > 0)")
    if principal[0] + principal[1] < principal[2] * (1.0 - 1e-9):
        raise ValueError("principal moments must satisfy I_a + I_b >= I_c")

    tensor.setflags(write=False)
    return tensor
