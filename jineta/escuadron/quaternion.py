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


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products a ⊗ b of stacks of quaternions (..., 4)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    product = np.empty(np.broadcast_shapes(a.shape, b.shape))
    product[..., 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    product[..., 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    product[..., 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    product[..., 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return product


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugates [w, -x, -y, -z] (..., 4): for unit quaternions, the inverse."""
    result = -q
    result[..., 0] = q[..., 0]
    return result


def to_rotation_vector(q: np.ndarray) -> np.ndarray:
    """Rotation vectors angle * axis [rad], shape (..., 3), of unit quaternions
    (..., 4), always the short way (angle in [0, pi]), as
    `core.quaternion.to_rotation_vector`: q with w < 0 is flipped to -q first, and
    below sin(angle / 2) = 1e-12 the vector is 2 * (x, y, z)."""
    q = np.where(q[..., 0:1] < 0.0, -q, q)
    vector = q[..., 1:4]
    v0, v1, v2 = vector[..., 0], vector[..., 1], vector[..., 2]
    sin_half = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    small = sin_half < 1e-12
    angle = 2.0 * np.arctan2(sin_half, q[..., 0])
    divisor = np.where(small, 1.0, sin_half)  # no division by ~0 where unused
    return np.where(
        small[..., None],
        2.0 * vector,
        angle[..., None] * vector / divisor[..., None],
    )


def from_rotation_vector(vector: np.ndarray) -> np.ndarray:
    """Unit quaternions (..., 4) of rotation vectors angle * axis (..., 3) [rad]:
    [cos(angle / 2), sin(angle / 2) axis] (the inverse of `to_rotation_vector`;
    `core.quaternion.from_axis_angle` on stacks). Below an angle of 1e-12 the
    vector part is vector / 2 (sin(a / 2) / a -> 1 / 2)."""
    v0, v1, v2 = vector[..., 0], vector[..., 1], vector[..., 2]
    angle = np.sqrt(v0 * v0 + v1 * v1 + v2 * v2)
    small = angle < 1e-12
    scale = np.where(small, 0.5, np.sin(0.5 * angle) / np.where(small, 1.0, angle))
    return np.concatenate(
        [np.cos(0.5 * angle)[..., None], scale[..., None] * vector], axis=-1
    )


def from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Unit quaternions (w >= 0), shape (..., 4), of rotation matrices (..., 3, 3)
    (body -> NED), as `core.quaternion.from_matrix` (Shepperd's method).

    The reference picks one of four formulas by an `if` on the matrix; here all
    four are computed for every matrix and each one keeps the formula its
    condition picks (the others may divide by zero or take the root of a negative
    number: those results are thrown away).
    """
    r = matrix
    r00, r01, r02 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    r10, r11, r12 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    r20, r21, r22 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]
    trace = r00 + r11 + r22
    use_w = trace > 0.0
    use_x = ~use_w & (r00 > r11) & (r00 > r22)
    use_y = ~use_w & ~use_x & (r11 > r22)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 2.0 * np.sqrt(1.0 + trace)  # s = 4w
        from_w = np.stack(
            [s / 4, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s], -1
        )
        s = 2.0 * np.sqrt(1.0 + r00 - r11 - r22)  # s = 4x
        from_x = np.stack(
            [(r21 - r12) / s, s / 4, (r01 + r10) / s, (r02 + r20) / s], -1
        )
        s = 2.0 * np.sqrt(1.0 + r11 - r00 - r22)  # s = 4y
        from_y = np.stack(
            [(r02 - r20) / s, (r01 + r10) / s, s / 4, (r12 + r21) / s], -1
        )
        s = 2.0 * np.sqrt(1.0 + r22 - r00 - r11)  # s = 4z
        from_z = np.stack(
            [(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, s / 4], -1
        )
    q = np.where(
        use_w[..., None],
        from_w,
        np.where(use_x[..., None], from_x, np.where(use_y[..., None], from_y, from_z)),
    )
    q = normalize(q)
    return np.where(q[..., 0:1] < 0.0, -q, q)
