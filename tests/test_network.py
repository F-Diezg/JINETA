"""Tests for the neural network: the forward pass by hand, the gradients against
finite differences, the initial weights, Adam against its formulas, fitting a
known function, gradient clipping, and saving and loading."""

import numpy as np
import pytest

from jineta.doma.network import (
    MLP,
    Adam,
    clip_by_global_norm,
    load_arrays,
    orthogonal,
    save_arrays,
)


def numerical_gradients(net: MLP, loss, h: float = 1e-6) -> list[np.ndarray]:
    """Central differences of loss() with respect to every parameter."""
    grads = []
    for p in net.parameters:
        g = np.zeros_like(p)
        for index in np.ndindex(p.shape):
            keep = p[index]
            p[index] = keep + h
            up = loss()
            p[index] = keep - h
            down = loss()
            p[index] = keep
            g[index] = (up - down) / (2.0 * h)
        grads.append(g)
    return grads


def test_forward_by_hand():
    net = MLP(
        [np.array([[1.0, -1.0], [0.5, 2.0]]), np.array([[2.0], [-3.0]])],
        [np.array([0.1, 0.0]), np.array([0.5])],
    )
    x = np.array([[1.0, 2.0], [0.0, -1.0]])
    hidden = np.tanh(x @ np.array([[1.0, -1.0], [0.5, 2.0]]) + [0.1, 0.0])
    expected = hidden @ np.array([[2.0], [-3.0]]) + 0.5
    np.testing.assert_allclose(net(x), expected, rtol=0, atol=1e-15)
    assert net.sizes == (2, 2, 1)


def test_one_layer_is_linear():
    rng = np.random.default_rng(0)
    net = MLP.create([3, 2], rng)
    x = rng.standard_normal((5, 3))
    np.testing.assert_allclose(net(x), x @ net.weights[0], atol=1e-15)


@pytest.mark.parametrize("sizes", [(3, 5, 2), (4, 6, 5, 3), (2, 1)])
def test_gradients_match_finite_differences(sizes):
    rng = np.random.default_rng(1)
    net = MLP.create(sizes, rng, output_gain=1.0)
    for b in net.biases:
        b[:] = rng.standard_normal(b.shape)  # biases away from zero too
    x = rng.standard_normal((7, sizes[0]))
    target = rng.standard_normal((7, sizes[-1]))

    def loss() -> float:
        return float(np.sum((net(x) - target) ** 2) / 2.0)

    y, cache = net.forward(x)
    analytic = net.backward(cache, y - target)
    numeric = numerical_gradients(net, loss)
    for a, n in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a, n, rtol=1e-6, atol=1e-8)


def test_orthogonal_initial_weights():
    rng = np.random.default_rng(2)
    tall = orthogonal(10, 4, 2.0, rng)  # orthonormal columns, times 2
    np.testing.assert_allclose(tall.T @ tall, 4.0 * np.eye(4), atol=1e-12)
    wide = orthogonal(4, 10, 1.0, rng)  # orthonormal rows
    np.testing.assert_allclose(wide @ wide.T, np.eye(4), atol=1e-12)
    net = MLP.create([5, 8, 3], rng, output_gain=0.01)
    np.testing.assert_allclose(
        net.weights[0] @ net.weights[0].T, 2.0 * np.eye(5), atol=1e-12
    )
    np.testing.assert_allclose(
        net.weights[1].T @ net.weights[1], 1e-4 * np.eye(3), atol=1e-15
    )
    assert all(np.all(b == 0.0) for b in net.biases)
    again = MLP.create([5, 8, 3], np.random.default_rng(3), output_gain=0.01)
    same = MLP.create([5, 8, 3], np.random.default_rng(3), output_gain=0.01)
    for p, q in zip(again.parameters, same.parameters, strict=True):
        np.testing.assert_array_equal(p, q)


def test_a_row_is_the_same_alone_or_in_a_batch_to_rounding():
    rng = np.random.default_rng(4)
    net = MLP.create([43, 64, 64, 4], rng)
    x = rng.standard_normal((2048, 43))
    batch = net(x)
    for i in range(0, 2048, 97):
        np.testing.assert_allclose(net(x[i : i + 1])[0], batch[i], rtol=0, atol=1e-13)


def test_adam_by_its_formulas():
    """Three steps on one parameter, computed by hand."""
    p = np.array([1.0])
    adam = Adam([p], learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    m = v = 0.0
    value = 1.0
    for t, g in enumerate([2.0, -1.0, 0.5], start=1):
        adam.step([np.array([g])])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        m_hat, v_hat = m / (1 - 0.9**t), v / (1 - 0.999**t)
        value -= 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
        assert p[0] == pytest.approx(value, abs=1e-15)
    # The first step is the learning rate whatever the gradient's size.
    q = np.array([0.0])
    Adam([q], learning_rate=0.01, epsilon=1e-12).step([np.array([1e6])])
    assert q[0] == pytest.approx(-0.01, rel=1e-9)


def test_fits_a_known_function():
    """sin(x) on [-pi, pi] learned by a small network from samples."""
    rng = np.random.default_rng(5)
    net = MLP.create([1, 32, 32, 1], rng)
    adam = Adam(net.parameters, learning_rate=3e-3)
    for _ in range(3000):
        x = rng.uniform(-np.pi, np.pi, (64, 1))
        y, cache = net.forward(x)
        adam.step(net.backward(cache, (y - np.sin(x)) / len(x)))
    grid = np.linspace(-np.pi, np.pi, 201)[:, None]
    error = net(grid) - np.sin(grid)
    assert np.sqrt(np.mean(error**2)) < 0.02


def test_clipping_by_global_norm():
    grads = [np.array([3.0]), np.array([[4.0, 0.0]])]
    clipped, norm = clip_by_global_norm(grads, 1.0)
    assert norm == pytest.approx(5.0)
    np.testing.assert_allclose(clipped[0], [0.6])
    np.testing.assert_allclose(clipped[1], [[0.8, 0.0]])
    kept, _ = clip_by_global_norm(grads, 10.0)
    for a, b in zip(kept, grads, strict=True):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        clip_by_global_norm(grads, 0.0)


def test_save_and_load_bit_for_bit(tmp_path):
    rng = np.random.default_rng(6)
    net = MLP.create([4, 7, 2], rng)
    adam = Adam(net.parameters)
    x = rng.standard_normal((3, 4))
    y, cache = net.forward(x)
    adam.step(net.backward(cache, y))
    save_arrays(tmp_path / "net.npz", {**net.state("pi_"), **adam.state("adam_")})
    loaded = load_arrays(tmp_path / "net.npz")
    copy = MLP.from_state(loaded, "pi_")
    np.testing.assert_array_equal(copy(x), net(x))
    other = Adam(copy.parameters)
    other.load_state(loaded, "adam_")
    assert other.steps == 1
    y, cache = net.forward(x)
    adam.step(net.backward(cache, y))
    y, cache = copy.forward(x)
    other.step(copy.backward(cache, y))
    for p, q in zip(net.parameters, copy.parameters, strict=True):
        np.testing.assert_array_equal(p, q)


def test_rejects_bad_input():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        MLP.create([3], rng)
    with pytest.raises(ValueError):
        MLP.create([3, 0, 2], rng)
    with pytest.raises(ValueError):
        MLP([np.ones((2, 3))], [np.ones(2)])
    with pytest.raises(ValueError):
        MLP([np.ones((2, 3)), np.ones((4, 1))], [np.ones(3), np.ones(1)])
    with pytest.raises(ValueError):
        MLP.create([3, 2], rng)(np.ones((1, 4)))
    with pytest.raises(ValueError):
        Adam([np.ones(2)], learning_rate=0.0)
    with pytest.raises(ValueError):
        Adam([np.ones(2)], beta1=1.0)
    with pytest.raises(ValueError):
        Adam([np.ones(2)]).step([])
