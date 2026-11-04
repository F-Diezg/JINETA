"""Small vector operations, written out for speed.

NumPy's general functions pay a fixed cost on every call (checking shapes and
axes, moving axes around) before doing any arithmetic. For a 3-vector that set-up
costs far more than the six multiplications themselves: `np.cross` takes ~10 times
longer than writing the formula out. The simulator calls a cross product several
times per derivative, so this matters.

The functions here give exactly the same numbers as their NumPy counterparts (the
same operations in the same order, so the same rounding), only faster.
"""

import numpy as np
import numpy.typing as npt


def cross(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    """Cross product a x b of 3-vectors, bit for bit equal to `np.cross(a, b)`.

    a and b are 3-vectors (shape (3,)) or stacks of them (shape (..., 3)), which
    broadcast against each other like any NumPy operation: a (n, 3) stack crossed
    with one (3,) vector gives n products.

        a x b = [a1 b2 - a2 b1,  a2 b0 - a0 b2,  a0 b1 - a1 b0]
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[-1:] != (3,) or b.shape[-1:] != (3,):
        raise ValueError("cross needs 3-vectors (last axis of size 3)")
    if a.ndim == 1 and b.ndim == 1:
        # One pair: indexing single numbers is the cheapest path.
        return np.array(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )
    out = np.empty(np.broadcast_shapes(a.shape, b.shape))
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out[..., 0] = a1 * b2 - a2 * b1
    out[..., 1] = a2 * b0 - a0 * b2
    out[..., 2] = a0 * b1 - a1 * b0
    return out
