"""A stochastic policy, its critic and the normaliser of its observations.

The policy. To learn, the drone has to try things: its policy is not one action
per observation but a probability distribution over actions, a Gaussian for each
of the four numbers. A network gives the mean (what it would do), and a learned
spread (one standard deviation per action number, the same for every
observation) says how much it explores around it:

    a ~ N(mu(o), sigma^2),   mu = network(o),   sigma = exp(log_std)

The logarithm of the spread is learned rather than the spread itself so that it
can never become negative. Learning makes the actions that turned out better
than expected more likely; the log-probability of an action,

    log pi(a | o) = - sum_i [ (a_i - mu_i)^2 / (2 sigma_i^2) + log sigma_i ]
                    - (k / 2) log(2 pi),

is what it changes, and its gradients with respect to mu and log sigma are
simple: (a - mu) / sigma^2 and ((a - mu) / sigma)^2 - 1. The entropy, a measure
of how spread the distribution is, sum_i (log sigma_i + log(2 pi e) / 2), depends
only on sigma. Flying for real (or measuring a lap) uses the mean: the
deterministic policy. Actions outside [-1, 1] are clipped by the environment, not
by the policy.

The critic. A second network estimates the value of an observation: the return
(discounted sum of rewards) expected from there with the current policy. It is
what tells better than expected from worse (PPO, `doma/ppo.py`).

The normaliser. The observation mixes metres, metres per second, radians per
second and plain numbers, from ~0.1 to ~15. Networks learn best when every input
is around 0 with a spread of about 1, so each number is shifted by its running
mean and divided by its running standard deviation over everything seen so far
(clipped to +-10). The statistics are updated during training and frozen
afterwards; they are part of the policy (saved with it).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from jineta.doma.network import MLP, load_arrays, save_arrays

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass
class RunningNormalizer:
    """Running mean and variance of vectors of `size` numbers, and the vectors
    scaled by them: (x - mean) / sqrt(var + epsilon), clipped to +-clip."""

    size: int
    clip: float = 10.0
    epsilon: float = 1e-8
    count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be >= 1")
        if not (self.clip > 0.0 and self.epsilon > 0.0):
            raise ValueError("clip and epsilon must be > 0")
        self.mean = np.zeros(self.size)
        self.var = np.ones(self.size)

    def update(self, batch: npt.ArrayLike) -> None:
        """Add a batch of vectors (m, size) to the statistics. Two groups' means
        and variances combine exactly (Chan, Golub & LeVeque, 1979):
        mean = mean_a + delta * m_b / m, and the sums of squared deviations add
        up plus delta^2 * m_a * m_b / m, with delta = mean_b - mean_a, so the
        result is the statistics of all the data seen, whatever the pieces."""
        x = np.asarray(batch, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.size or len(x) == 0:
            raise ValueError(f"a batch must be (m >= 1, {self.size})")
        m_b = len(x)
        mean_b = x.mean(axis=0)
        var_b = x.var(axis=0)
        if self.count == 0:
            self.mean, self.var, self.count = mean_b, var_b, m_b
            return
        m_a, m = self.count, self.count + m_b
        delta = mean_b - self.mean
        squares = self.var * m_a + var_b * m_b + delta * delta * (m_a * m_b / m)
        self.mean = self.mean + delta * (m_b / m)
        self.var = squares / m
        self.count = m

    def __call__(self, x: npt.ArrayLike) -> np.ndarray:
        """The vectors (m, size) scaled by the statistics."""
        scaled = (np.asarray(x, dtype=float) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(scaled, -self.clip, self.clip)

    def state(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}mean": self.mean.copy(),
            f"{prefix}var": self.var.copy(),
            f"{prefix}count": np.array(self.count),
            f"{prefix}clip": np.array(self.clip),
            f"{prefix}epsilon": np.array(self.epsilon),
        }

    @classmethod
    def from_state(
        cls, state: dict[str, np.ndarray], prefix: str = ""
    ) -> "RunningNormalizer":
        normalizer = cls(
            len(state[f"{prefix}mean"]),
            float(state[f"{prefix}clip"]),
            float(state[f"{prefix}epsilon"]),
        )
        normalizer.mean = state[f"{prefix}mean"].copy()
        normalizer.var = state[f"{prefix}var"].copy()
        normalizer.count = int(state[f"{prefix}count"])
        return normalizer


def gaussian_log_prob(
    actions: np.ndarray, mean: np.ndarray, log_std: np.ndarray
) -> np.ndarray:
    """log N(a; mean, exp(log_std)^2) summed over the action numbers: (m,) for
    actions and means (m, k) and log_std (k,)."""
    z = (actions - mean) * np.exp(-log_std)
    k = mean.shape[-1]
    return -0.5 * np.sum(z * z, axis=-1) - np.sum(log_std) - 0.5 * k * LOG_2PI


def gaussian_log_prob_gradients(
    actions: np.ndarray, mean: np.ndarray, log_std: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients (m, k) of each sample's log-probability with respect to the mean
    and to log_std: (a - mu) / sigma^2 and ((a - mu) / sigma)^2 - 1."""
    inverse = np.exp(-log_std)
    z = (actions - mean) * inverse
    return z * inverse, z * z - 1.0


def gaussian_entropy(log_std: np.ndarray) -> float:
    """Entropy of the Gaussian (summed over the action numbers)."""
    return float(np.sum(log_std) + 0.5 * len(log_std) * (LOG_2PI + 1.0))


@dataclass
class GaussianPolicy:
    """A Gaussian policy: `network` gives the mean action of (normalised)
    observations, `log_std` (k,) the logarithm of its spread."""

    network: MLP
    log_std: np.ndarray

    def __post_init__(self) -> None:
        self.log_std = np.array(self.log_std, dtype=float)
        if self.log_std.shape != (self.network.sizes[-1],):
            raise ValueError("one log_std per action number")

    @classmethod
    def create(
        cls,
        observation_size: int,
        action_size: int,
        hidden: tuple[int, ...],
        rng: np.random.Generator,
        log_std: float = -0.5,
    ) -> "GaussianPolicy":
        """A new policy whose mean starts near zero (last layer's gain 0.01) with
        spread exp(log_std) on every action number."""
        network = MLP.create(
            (observation_size, *hidden, action_size), rng, output_gain=0.01
        )
        return cls(network, np.full(action_size, log_std))

    @property
    def parameters(self) -> list[np.ndarray]:
        """The network's parameters and then log_std."""
        return [*self.network.parameters, self.log_std]

    def mean(self, observations: npt.ArrayLike) -> np.ndarray:
        """The mean actions (m, k): the deterministic policy."""
        return self.network(observations)

    def sample(
        self, observations: npt.ArrayLike, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Actions drawn from the policy (m, k) and their log-probabilities (m,)."""
        mean = self.network(observations)
        actions = mean + np.exp(self.log_std) * rng.standard_normal(mean.shape)
        return actions, gaussian_log_prob(actions, mean, self.log_std)

    def entropy(self) -> float:
        return gaussian_entropy(self.log_std)


@dataclass
class Agent:
    """What training produces: the observation normaliser, the policy and the
    critic (value network). `act` is all that flying needs."""

    normalizer: RunningNormalizer
    policy: GaussianPolicy
    value: MLP
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        observation_size: int,
        action_size: int,
        hidden: tuple[int, ...],
        rng: np.random.Generator,
        log_std: float = -0.5,
    ) -> "Agent":
        policy = GaussianPolicy.create(
            observation_size, action_size, hidden, rng, log_std
        )
        value = MLP.create((observation_size, *hidden, 1), rng, output_gain=1.0)
        return cls(RunningNormalizer(observation_size), policy, value)

    def act(
        self,
        observations: npt.ArrayLike,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Actions (m, k) for raw observations (m, size): the mean (deterministic
        policy) without `rng`, drawn from the policy with it."""
        normalised = self.normalizer(observations)
        if rng is None:
            return self.policy.mean(normalised)
        return self.policy.sample(normalised, rng)[0]

    def values(self, observations: npt.ArrayLike) -> np.ndarray:
        """The critic's values (m,) of raw observations (m, size)."""
        return self.value(self.normalizer(observations))[:, 0]

    def save(self, path: str | Path) -> None:
        """Everything to one .npz file (`info`: numbers and strings only)."""
        arrays = {
            **self.normalizer.state("norm_"),
            **self.policy.network.state("pi_"),
            "pi_log_std": self.policy.log_std.copy(),
            **self.value.state("v_"),
        }
        for key, value in self.info.items():
            arrays[f"info_{key}"] = np.array(value)
        save_arrays(path, arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Agent":
        state = load_arrays(path)
        info = {
            key[len("info_") :]: value.item()
            for key, value in state.items()
            if key.startswith("info_")
        }
        return cls(
            RunningNormalizer.from_state(state, "norm_"),
            GaussianPolicy(MLP.from_state(state, "pi_"), state["pi_log_std"]),
            MLP.from_state(state, "v_"),
            info,
        )
