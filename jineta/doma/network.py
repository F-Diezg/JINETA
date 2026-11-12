"""A neural network in NumPy: the multilayer perceptron, its gradients by hand
(backpropagation) and the Adam optimiser.

The network. A multilayer perceptron (MLP) is a chain of layers; each multiplies
its input by a matrix of weights, adds a vector of biases and, except the last,
bends the result with tanh:

    h_0 = x,   h_(k+1) = tanh(h_k W_k + b_k),   y = h_(L-1) W_(L-1) + b_(L-1)

Inputs are rows: x is (batch, inputs), each row one sample, and the batch is
processed at once (one matrix product per layer). The weights are what learning
changes. With tanh between layers the network can bend into almost any smooth
function of its inputs; without it, a chain of matrices would still be one
matrix.

The gradient. Learning moves each weight against the gradient of a loss (how
much the loss changes per unit of that weight). Backpropagation is the chain
rule applied from the output back: given dL/dy, each layer gives the gradient of
its weights (h_k^T dL/dz_k), of its biases (the column sums of dL/dz_k) and of
its input (dL/dz_k W_k^T), and tanh multiplies by its own slope, 1 - tanh^2.
The forward pass keeps every layer's output (the cache) for the backward pass.

The optimiser. Adam (Kingma & Ba, 2015) steps each parameter by a learning rate
times its gradient's running mean over the square root of its running mean
square: every parameter moves about the same distance per step whatever the
scale of its gradient, and noise averages out.

Rounding: the products use NumPy's matrix product (BLAS), which may add terms in
a different order for different batch sizes, so a row gives the same numbers
alone or in a batch to rounding (~1e-14), not bit for bit (unlike the physics).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt

HIDDEN_GAIN = np.sqrt(2.0)  # scale of the hidden layers' initial weights


def orthogonal(rows: int, columns: int, gain: float, rng: np.random.Generator):
    """A (rows, columns) matrix with orthonormal rows or columns (whichever are
    fewer) times `gain`: the usual initial weights of a policy network (Saxe et
    al., 2014), which keep the size of a signal through the layers. Drawn as Q of
    the QR decomposition of a Gaussian matrix, its signs fixed by R's diagonal so
    that it is uniform over orthogonal matrices."""
    big, small = max(rows, columns), min(rows, columns)
    q, r = np.linalg.qr(rng.standard_normal((big, small)))
    q = q * np.sign(np.diag(r))
    return gain * (q if rows >= columns else q.T)


@dataclass
class MLP:
    """A multilayer perceptron with tanh between its layers and none after the
    last. `weights[k]` is (inputs_k, outputs_k), `biases[k]` is (outputs_k,)."""

    weights: list[np.ndarray]
    biases: list[np.ndarray]

    def __post_init__(self) -> None:
        if len(self.weights) == 0 or len(self.weights) != len(self.biases):
            raise ValueError("an MLP needs one bias vector per weight matrix (>= 1)")
        self.weights = [np.array(w, dtype=float) for w in self.weights]
        self.biases = [np.array(b, dtype=float) for b in self.biases]
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ValueError(f"layer {k}: weights (m, n) need biases (n,)")
            if k > 0 and w.shape[0] != self.weights[k - 1].shape[1]:
                raise ValueError(f"layer {k}: its inputs must be the last outputs")

    @classmethod
    def create(
        cls,
        sizes: Sequence[int],
        rng: np.random.Generator,
        output_gain: float = 1.0,
    ) -> "MLP":
        """A new network with these layer sizes (inputs, hidden..., outputs):
        orthogonal weights (gain sqrt(2) for hidden layers, `output_gain` for the
        last), zero biases."""
        if len(sizes) < 2 or min(sizes) < 1:
            raise ValueError("sizes must be >= 2 positive numbers")
        last = len(sizes) - 2
        weights = [
            orthogonal(m, n, output_gain if k == last else HIDDEN_GAIN, rng)
            for k, (m, n) in enumerate(pairwise(sizes))
        ]
        return cls(weights, [np.zeros(n) for n in sizes[1:]])

    @property
    def sizes(self) -> tuple[int, ...]:
        return (self.weights[0].shape[0], *(w.shape[1] for w in self.weights))

    @property
    def parameters(self) -> list[np.ndarray]:
        """Every weight and bias (W_0, b_0, W_1, b_1, ...): the arrays an
        optimiser changes in place, in the order of `backward`'s gradients."""
        return [p for pair in zip(self.weights, self.biases, strict=True) for p in pair]

    def forward(self, x: npt.ArrayLike) -> tuple[np.ndarray, list[np.ndarray]]:
        """Outputs (batch, outputs) for inputs (batch, inputs), and the cache that
        `backward` needs (the input and every hidden layer's output)."""
        h = np.asarray(x, dtype=float)
        if h.ndim != 2 or h.shape[1] != self.sizes[0]:
            raise ValueError(f"inputs must be (batch, {self.sizes[0]})")
        cache = [h]
        last = len(self.weights) - 1
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if k < last:
                h = np.tanh(h)
                cache.append(h)
        return h, cache

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        """Outputs (batch, outputs) for inputs (batch, inputs)."""
        return self.forward(x)[0]

    def backward(
        self, cache: list[np.ndarray], grad_output: npt.ArrayLike
    ) -> list[np.ndarray]:
        """Gradients of a loss with respect to every parameter (in the order of
        `parameters`), given the cache of the forward pass and the gradient of
        the loss with respect to the outputs (batch, outputs)."""
        grad = np.asarray(grad_output, dtype=float)
        grads: list[np.ndarray] = []
        for k in reversed(range(len(self.weights))):
            if k < len(self.weights) - 1:
                grad = grad * (1.0 - cache[k + 1] * cache[k + 1])  # tanh' = 1 - tanh^2
            grads.append(grad.sum(axis=0))  # biases
            grads.append(cache[k].T @ grad)  # weights
            grad = grad @ self.weights[k].T  # to the layer below
        return grads[::-1]

    def state(self, prefix: str = "") -> dict[str, np.ndarray]:
        """The parameters by name (`<prefix>W0`, `<prefix>b0`, ...), for saving."""
        named = {}
        for k, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            named[f"{prefix}W{k}"], named[f"{prefix}b{k}"] = w.copy(), b.copy()
        return named

    @classmethod
    def from_state(cls, state: dict[str, np.ndarray], prefix: str = "") -> "MLP":
        """The network saved by `state` under this prefix."""
        layers = sum(1 for key in state if key.startswith(f"{prefix}W"))
        return cls(
            [state[f"{prefix}W{k}"] for k in range(layers)],
            [state[f"{prefix}b{k}"] for k in range(layers)],
        )


@dataclass
class Adam:
    """The Adam optimiser over a list of arrays changed in place.

    For each parameter p with gradient g, at step t:
        m = beta1 m + (1 - beta1) g          running mean of the gradient
        v = beta2 v + (1 - beta2) g^2        running mean of its square
        p -= learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    with m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t) (both start at zero,
    which would make the first steps too small without that correction).
    """

    parameters: list[np.ndarray]
    learning_rate: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-5
    steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.learning_rate > 0.0:
            raise ValueError("the learning rate must be > 0")
        if not (0.0 <= self.beta1 < 1.0 and 0.0 <= self.beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in [0, 1)")
        if not self.epsilon > 0.0:
            raise ValueError("epsilon must be > 0")
        self.mean = [np.zeros_like(p) for p in self.parameters]
        self.square = [np.zeros_like(p) for p in self.parameters]

    def step(self, grads: Sequence[np.ndarray]) -> None:
        """Move every parameter one step against its gradient (same order)."""
        if len(grads) != len(self.parameters):
            raise ValueError("one gradient per parameter")
        self.steps += 1
        correction1 = 1.0 - self.beta1**self.steps
        correction2 = 1.0 - self.beta2**self.steps
        for p, g, m, v in zip(
            self.parameters, grads, self.mean, self.square, strict=True
        ):
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= (
                self.learning_rate
                * (m / correction1)
                / (np.sqrt(v / correction2) + self.epsilon)
            )

    def state(self, prefix: str = "") -> dict[str, np.ndarray]:
        """The optimiser's memory by name, for saving a training to resume it."""
        named = {f"{prefix}steps": np.array(self.steps)}
        for k, (m, v) in enumerate(zip(self.mean, self.square, strict=True)):
            named[f"{prefix}m{k}"], named[f"{prefix}v{k}"] = m.copy(), v.copy()
        return named

    def load_state(self, state: dict[str, np.ndarray], prefix: str = "") -> None:
        """Restore the memory saved by `state`."""
        self.steps = int(state[f"{prefix}steps"])
        for k in range(len(self.parameters)):
            self.mean[k][...] = state[f"{prefix}m{k}"]
            self.square[k][...] = state[f"{prefix}v{k}"]


def clip_by_global_norm(
    grads: Sequence[np.ndarray], max_norm: float
) -> tuple[list[np.ndarray], float]:
    """The gradients scaled down together so that their joint length (the square
    root of the sum of every squared component) is at most `max_norm`, and that
    length before clipping. A rare huge gradient then cannot throw the network far
    in one step; the direction is kept."""
    if not max_norm > 0.0:
        raise ValueError("max_norm must be > 0")
    norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
    scale = min(1.0, max_norm / (norm + 1e-12))
    return [g * scale for g in grads], norm


def save_arrays(path: str | Path, arrays: dict[str, np.ndarray]) -> None:
    """Save named arrays to one .npz file (NumPy's zipped format)."""
    np.savez(Path(path), **arrays)


def load_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """The named arrays saved by `save_arrays`."""
    with np.load(Path(path)) as data:
        return {key: data[key] for key in data.files}
