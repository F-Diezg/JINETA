"""Quaternion operations for many drones at once.

The same conventions as `core.quaternion` ([w, x, y, z], Hamilton product, BODY ->
NED), but every function takes a stack of quaternions, shape (..., 4), usually
(N, 4) with one row per drone, and works on all of them in one array operation.

Each formula is written as in `core.quaternion`, term by term and in the same
order, so a drone gets the same numbers as with the reference functions up to the
last bit: `derivative` is equal bit for bit; `to_matrix` and `normalize` can differ
by one unit in the last place (~1e-16) in a few elements. In the reference, w, x,
y, z are NumPy scalars, and a scalar's `x**2` goes through C's `pow`, which is not
always exactly `x * x`; on an array `x**2` is computed as `x * x`. The norm of
`core.quaternion.normalize` adds its four squares in its own order.
"""

import numpy as np


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (body -> NED), shape (..., 3, 3), of unit quaternions
    q (..., 4)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    matrix = np.empty((*q.shape[:-1], 3, 3))
    matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    matrix[..., 0, 1] = 2 * (x * y - w * z)
    matrix[..., 0, 2] = 2 * (x * z + w * y)
    matrix[..., 1, 0] = 2 * (x * y + w * z)
    matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    matrix[..., 1, 2] = 2 * (y * z - w * x)
    matrix[..., 2, 0] = 2 * (x * z - w * y)
    matrix[..., 2, 1] = 2 * (y * z + w * x)
    matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)
    return matrix


def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Time derivatives (..., 4) of quaternions q (..., 4) given the angular
    velocities in body axes omega_body (..., 3) [rad/s]: q_dot = 1/2 q ⊗ [0, omega].

    The Hamilton product with a pure quaternion [0, p, q, r], with the terms
    multiplied by its zero left out (they add exactly nothing).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    p, r_q, r = omega_body[..., 0], omega_body[..., 1], omega_body[..., 2]
    q_dot = np.empty(q.shape)
    q_dot[..., 0] = -x * p - y * r_q - z * r
    q_dot[..., 1] = w * p + y * r - z * r_q
    q_dot[..., 2] = w * r_q - x * r + z * p
    q_dot[..., 3] = w * r + x * r_q - y * p
    return 0.5 * q_dot


def normalize(q: np.ndarray) -> np.ndarray:
    """Quaternions q (..., 4) scaled to unit norm (removes numerical drift)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    return q / norm[..., None]
