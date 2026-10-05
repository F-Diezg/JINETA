"""Quaternion utilities for attitude representation.

Convention: q = [w, x, y, z] (scalar first), Hamilton product, unit norm.
A quaternion describes the attitude of a body: it rotates vectors from the
BODY frame to the NED frame, i.e. v_ned = to_matrix(q) @ v_body.
"""

import numpy as np


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b (composition of rotations)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion for a rotation of `angle` [rad] about `axis` (any length)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = angle / 2
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm (removes numerical drift)."""
    return q / np.linalg.norm(q)


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body -> NED) equivalent to the unit quaternion q."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
    ])


def derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Time derivative of q given the angular velocity in body axes [rad/s]."""
    return 0.5 * multiply(q, np.concatenate([[0.0], omega_body]))