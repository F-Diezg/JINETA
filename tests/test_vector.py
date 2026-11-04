"""Tests for the written-out vector operations against NumPy: `cross` bit for bit,
`matvec` to rounding, for single vectors, stacks and broadcasting."""

import numpy as np
import pytest

from jineta.core.vector import cross, matvec


def random_vectors(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Vectors with components spread over 16 orders of magnitude and both signs,
    so that rounding is exercised everywhere."""
    return rng.normal(size=shape) * 10.0 ** rng.uniform(-8, 8, size=shape)


def test_basis_vectors():
    x, y, z = np.eye(3)
    assert np.array_equal(cross(x, y), z)
    assert np.array_equal(cross(y, z), x)
    assert np.array_equal(cross(z, x), y)
    assert np.array_equal(cross(y, x), -z)


def test_single_pairs_equal_numpy_bit_for_bit():
    rng = np.random.default_rng(0)
    for a, b in zip(
        random_vectors(rng, (1000, 3)), random_vectors(rng, (1000, 3)), strict=True
    ):
        assert np.array_equal(cross(a, b), np.cross(a, b))


@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [
        ((500, 3), (500, 3)),  # one pair per row
        ((500, 3), (3,)),  # every row with the same vector
        ((3,), (500, 3)),
        ((200, 4, 3), (4, 3)),  # e.g. n drones x 4 rotor positions
        ((200, 1, 3), (1, 4, 3)),  # broadcasting on both sides
    ],
)
def test_stacks_equal_numpy_bit_for_bit(shape_a, shape_b):
    rng = np.random.default_rng(1)
    a = random_vectors(rng, shape_a)
    b = random_vectors(rng, shape_b)
    result = cross(a, b)
    expected = np.cross(a, b)
    assert result.shape == expected.shape
    assert np.array_equal(result, expected)


def test_perpendicular_to_both():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 3))
    b = rng.normal(size=(100, 3))
    c = cross(a, b)
    assert np.allclose(np.sum(c * a, axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.sum(c * b, axis=1), 0.0, atol=1e-12)


def test_accepts_lists_and_integers():
    result = cross([1, 0, 0], [0, 1, 0])
    assert result.dtype == float
    assert np.array_equal(result, [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (np.zeros(2), np.zeros(3)),
        (np.zeros(3), np.zeros(4)),
        (np.zeros((5, 2)), np.zeros(3)),
        (np.float64(1.0), np.zeros(3)),
    ],
)
def test_rejects_vectors_that_are_not_3d(a, b):
    with pytest.raises(ValueError, match="3-vectors"):
        cross(a, b)


@pytest.mark.parametrize(
    ("shape_m", "shape_v"),
    [
        ((3, 3), (3,)),  # one matrix, one vector
        ((3, 3), (400, 3)),  # one matrix (an inertia) times many vectors
        ((400, 3, 3), (400, 3)),  # one matrix per vector (a rotation per drone)
        ((400, 3, 3), (3,)),
        ((100, 4, 3, 3), (4, 3)),
    ],
)
def test_matvec_equals_matmul_to_rounding(shape_m, shape_v):
    rng = np.random.default_rng(3)
    m = rng.normal(size=shape_m)
    v = rng.normal(size=shape_v)
    result = matvec(m, v)
    expected = (m @ v[..., None])[..., 0]
    assert result.shape == expected.shape
    assert np.allclose(result, expected, rtol=1e-15, atol=1e-15)


def test_matvec_adds_the_terms_in_order():
    """Each component is exactly ((M_i0 v_0 + M_i1 v_1) + M_i2 v_2)."""
    rng = np.random.default_rng(4)
    for m, v in zip(
        random_vectors(rng, (200, 3, 3)), random_vectors(rng, (200, 3)), strict=True
    ):
        expected = [m[i, 0] * v[0] + m[i, 1] * v[1] + m[i, 2] * v[2] for i in range(3)]
        assert np.array_equal(matvec(m, v), expected)


def test_matvec_rows_do_not_depend_on_the_batch():
    rng = np.random.default_rng(5)
    m = rng.normal(size=(1000, 3, 3))
    v = rng.normal(size=(1000, 3))
    everyone = matvec(m, v)
    for i in (0, 500, 999):
        assert np.array_equal(matvec(m[i : i + 1], v[i : i + 1])[0], everyone[i])
        assert np.array_equal(matvec(m[i], v[i]), everyone[i])


@pytest.mark.parametrize(
    ("m", "v"),
    [(np.zeros((2, 2)), np.zeros(2)), (np.zeros((3, 3)), np.zeros(4))],
)
def test_matvec_rejects_wrong_shapes(m, v):
    with pytest.raises(ValueError, match="3x3"):
        matvec(m, v)
