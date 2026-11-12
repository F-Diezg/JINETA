"""Proximal Policy Optimization (PPO; Schulman et al., 2017) in NumPy.

The loop. PPO alternates two phases:

  1. Collect experience: every drone (environment copy) flies `rollout_steps`
     actions drawn from the current policy, and the observation, action, its
     log-probability, the reward and how each step ended are stored.
  2. Learn from it: a few passes (`epochs`) over the experience, in shuffled
     minibatches, each a gradient step for the policy and one for the critic.
     Then the experience is thrown away (PPO learns only from what the current
     policy did: it is "on-policy").

The advantage. Whether an action was good is judged against what the critic
expected: the advantage A_t is how much better the rewards that followed were
than the value of the observation. One step's surprise is the temporal
difference

    delta_t = r_t + gamma * V(next observation) - V(o_t),

and GAE (generalised advantage estimation; Schulman et al., 2016) adds up the
surprises that follow, each weighted by (gamma * lambda)^k:

    A_t = delta_t + gamma * lambda * A_(t+1)      (cut where an episode ends)

lambda = 0 trusts the critic after one step (little noise, biased while the
critic is wrong); lambda = 1 adds up the real rewards to the end (unbiased,
noisy). The critic learns the targets A_t + V(o_t).

Where an episode ends, the next observation is not the next row (the drone was
restarted): a crash or a finish (terminated) has nothing after it, so
V(next) = 0; a cut by the time limit (truncated) could have gone on, so its value
is the critic's value of the episode's last observation, `final_observation`.
Treating a truncation as an end would teach the drone that the clock running out
is a disaster that only depends on time, which it cannot see.

The objective. The policy's step is limited so that one batch of experience
cannot change it too much: with the ratio rho = pi_new(a|o) / pi_old(a|o) of an
action's probability after and before, PPO maximises

    L = mean( min( rho A, clip(rho, 1 - eps, 1 + eps) A ) )

which stops paying once an action has become (1 +- eps) times as likely as it
was. The critic minimises the squared error to its targets. Advantages are
normalised per minibatch (mean 0, spread 1), gradients are clipped by their
global norm, and the learning rate falls linearly to zero over the training
(the usual choices of Stable-Baselines3 and CleanRL).

The environment: anything with the shape of `picadero.env.CourseEnv` (n copies
at once, each restarted in the step its episode ends, its last observation in
`info["final_observation"]`): `reset(seed)` -> (observations (n, size), info);
`step(actions (n, k))` -> (observations, rewards (n,), terminated (n,),
truncated (n,), info).

A training is repeatable: the same seed gives the same numbers on the same
computer (all randomness comes from one seeded generator).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from jineta.doma.network import Adam, clip_by_global_norm
from jineta.doma.policy import (
    Agent,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_log_prob_gradients,
)


class VectorEnv(Protocol):
    """What PPO needs from an environment (the shape of CourseEnv): reset, step
    and an `action_space` with a `shape`."""

    action_space: Any

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]: ...

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: ...


@dataclass(frozen=True)
class PPOSettings:
    """The settings of a training.

    rollout_steps: environment steps per drone between two updates.
    epochs: passes over each batch of experience.
    minibatch_size: samples per gradient step.
    gamma: discount per environment step (0.99 at 50 Hz: a reward 1.4 s later
        counts half).
    gae_lambda: the GAE weight (see the module).
    clip_range: eps of the clipped objective.
    learning_rate: Adam's at the start; it falls linearly to 0 if `anneal`.
    max_grad_norm: gradients clipped to this global norm.
    entropy_coef: weight of an entropy bonus (0: none).
    hidden: sizes of the hidden layers of the policy and the critic.
    log_std: the initial log standard deviation of the actions.
    """

    rollout_steps: int = 64
    epochs: int = 5
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    anneal: bool = True
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0
    hidden: tuple[int, ...] = (64, 64)
    log_std: float = -0.5

    def __post_init__(self) -> None:
        if min(self.rollout_steps, self.epochs, self.minibatch_size) < 1:
            raise ValueError("rollout_steps, epochs and minibatch_size must be >= 1")
        if not (0.0 <= self.gamma <= 1.0 and 0.0 <= self.gae_lambda <= 1.0):
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if not (self.clip_range > 0.0 and self.learning_rate > 0.0):
            raise ValueError("clip_range and learning_rate must be > 0")
        if not (self.max_grad_norm > 0.0 and self.entropy_coef >= 0.0):
            raise ValueError("max_grad_norm must be > 0 and entropy_coef >= 0")
        if len(self.hidden) == 0 or min(self.hidden) < 1:
            raise ValueError("hidden must be one or more layer sizes >= 1")


def gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    ends: np.ndarray,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Advantages (T, n) by GAE for T steps of n environments.

    rewards, values (T, n): r_t and V(o_t).
    next_values (T, n): the value of what followed each step: V(o_(t+1)) while
        the episode goes on, the value of its last observation where it was
        truncated, 0 where it terminated.
    ends (T, n): where an episode ended (terminated or truncated): the sum of
        surprises is cut there.
    """
    deltas = rewards + gamma * next_values - values
    advantages = np.zeros_like(deltas)
    following = np.zeros(deltas.shape[1:])
    for t in reversed(range(len(deltas))):
        following = deltas[t] + gamma * lam * np.where(ends[t], 0.0, following)
        advantages[t] = following
    return advantages


@dataclass
class Rollout:
    """One batch of experience: T steps of n environments (arrays (T, n, ...)),
    the observations already normalised as the policy saw them."""

    observations: np.ndarray  # (T, n, size), normalised
    actions: np.ndarray  # (T, n, k)
    log_probs: np.ndarray  # (T, n)
    rewards: np.ndarray  # (T, n)
    values: np.ndarray  # (T, n)
    next_values: np.ndarray  # (T, n)
    ends: np.ndarray  # (T, n)
    raw_observations: np.ndarray  # (T, n, size), for the normaliser


@dataclass
class Episodes:
    """The episodes that ended during a rollout: one entry each."""

    returns: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)


class PPO:
    """Trains an agent on a vectorised environment.

    env: the environment (n copies; see the module).
    settings: the training's settings.
    seed: seeds the environment, the initial networks and every random draw.
    agent: an agent to keep training; a new one if None.
    episode_info: keys of the step's info copied for each episode that ends
        (e.g. "status", "lap_time").
    """

    def __init__(
        self,
        env: VectorEnv,
        settings: PPOSettings | None = None,
        seed: int = 0,
        agent: Agent | None = None,
        episode_info: tuple[str, ...] = (),
    ) -> None:
        self.env = env
        self.settings = settings or PPOSettings()
        self.rng = np.random.default_rng(seed)
        self.observations, _ = env.reset(seed=seed)
        self.n, size = self.observations.shape
        if agent is None:
            k = int(np.prod(env.action_space.shape))
            agent = Agent.create(
                size, k, self.settings.hidden, self.rng, self.settings.log_std
            )
        self.agent = agent
        self.policy_optimizer = Adam(
            agent.policy.parameters, self.settings.learning_rate
        )
        self.value_optimizer = Adam(agent.value.parameters, self.settings.learning_rate)
        self.episode_info = episode_info
        self.steps = 0  # environment steps so far, all copies together

    # ---- experience ----------------------------------------------------------------

    def collect(self) -> tuple[Rollout, Episodes]:
        """Fly `rollout_steps` steps of every environment copy with the current
        policy (normaliser frozen meanwhile, updated after)."""
        s, agent = self.settings, self.agent
        T, n = s.rollout_steps, self.n
        size = self.observations.shape[1]
        k = agent.policy.log_std.shape[0]
        rollout = Rollout(
            observations=np.zeros((T, n, size)),
            actions=np.zeros((T, n, k)),
            log_probs=np.zeros((T, n)),
            rewards=np.zeros((T, n)),
            values=np.zeros((T, n)),
            next_values=np.zeros((T, n)),
            ends=np.zeros((T, n), dtype=bool),
            raw_observations=np.zeros((T, n, size)),
        )
        episodes = Episodes()
        normalised = agent.normalizer(self.observations)
        values = agent.value(normalised)[:, 0]
        for t in range(T):
            actions, log_probs = agent.policy.sample(normalised, self.rng)
            observations, rewards, terminated, truncated, info = self.env.step(actions)
            rollout.raw_observations[t] = self.observations
            rollout.observations[t] = normalised
            rollout.actions[t] = actions
            rollout.log_probs[t] = log_probs
            rollout.rewards[t] = rewards
            rollout.values[t] = values
            ends = terminated | truncated
            rollout.ends[t] = ends
            normalised = agent.normalizer(observations)
            values = agent.value(normalised)[:, 0]
            next_values = np.where(ends, 0.0, values)
            if np.any(truncated):
                final = agent.normalizer(info["final_observation"][truncated])
                next_values[truncated] = agent.value(final)[:, 0]
            rollout.next_values[t] = next_values
            for i in np.flatnonzero(ends):
                episodes.returns.append(float(info["episode_return"][i]))
                episodes.lengths.append(int(info["episode_steps"][i]))
                episodes.infos.append(
                    {key: info[key][i].item() for key in self.episode_info}
                )
            self.observations = observations
        self.steps += T * n
        agent.normalizer.update(rollout.raw_observations.reshape(-1, size))
        return rollout, episodes

    # ---- learning ------------------------------------------------------------------

    def update(self, rollout: Rollout, learning_rate: float) -> dict[str, float]:
        """`epochs` passes over the rollout in shuffled minibatches; returns the
        diagnostics of the update (averages over its minibatches)."""
        s, agent = self.settings, self.agent
        policy = agent.policy
        advantages = gae(
            rollout.rewards,
            rollout.values,
            rollout.next_values,
            rollout.ends,
            s.gamma,
            s.gae_lambda,
        )
        targets = advantages + rollout.values
        size = rollout.observations.shape[-1]
        k = rollout.actions.shape[-1]
        obs = rollout.observations.reshape(-1, size)
        actions = rollout.actions.reshape(-1, k)
        old_log_probs = rollout.log_probs.reshape(-1)
        advantages = advantages.reshape(-1)
        targets = targets.reshape(-1)
        total = len(obs)
        batch = min(s.minibatch_size, total)
        for optimizer in (self.policy_optimizer, self.value_optimizer):
            optimizer.learning_rate = learning_rate
        sums: dict[str, float] = {}
        count = 0
        for _ in range(s.epochs):
            order = self.rng.permutation(total)
            for start in range(0, total - batch + 1, batch):
                rows = order[start : start + batch]
                diagnostics = self._minibatch(
                    obs[rows],
                    actions[rows],
                    old_log_probs[rows],
                    advantages[rows],
                    targets[rows],
                )
                for key, value in diagnostics.items():
                    sums[key] = sums.get(key, 0.0) + value
                count += 1
        result = {key: value / count for key, value in sums.items()}
        values = rollout.values.reshape(-1)
        variance = np.var(targets)
        result["explained_variance"] = (
            float(1.0 - np.var(targets - values) / variance) if variance > 0 else 0.0
        )
        result["std"] = float(np.mean(np.exp(policy.log_std)))
        result["learning_rate"] = learning_rate
        return result

    def _minibatch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        targets: np.ndarray,
    ) -> dict[str, float]:
        """One gradient step of the policy and one of the critic."""
        s, agent = self.settings, self.agent
        policy = agent.policy
        m = len(obs)
        std = advantages.std()
        advantages = (advantages - advantages.mean()) / (std + 1e-8)

        # Policy: loss = -mean(min(rho A, clip(rho) A)) - c * entropy.
        mean, cache = policy.network.forward(obs)
        log_probs = gaussian_log_prob(actions, mean, policy.log_std)
        log_ratio = log_probs - old_log_probs
        ratio = np.exp(log_ratio)
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1.0 - s.clip_range, 1.0 + s.clip_range) * advantages
        policy_loss = -float(np.mean(np.minimum(unclipped, clipped)))
        # The gradient flows through rho only where the unclipped term is the
        # smaller one: d(rho)/d(log pi) = rho.
        active = unclipped <= clipped
        d_log_prob = -np.where(active, ratio * advantages, 0.0) / m
        d_mean, d_log_std = gaussian_log_prob_gradients(actions, mean, policy.log_std)
        grad_mean = d_log_prob[:, None] * d_mean
        grad_log_std = (d_log_prob[:, None] * d_log_std).sum(axis=0)
        grad_log_std -= s.entropy_coef  # d(entropy)/d(log_std) = 1
        grads = [*policy.network.backward(cache, grad_mean), grad_log_std]
        grads, policy_norm = clip_by_global_norm(grads, s.max_grad_norm)
        self.policy_optimizer.step(grads)

        # Critic: loss = mean((V - target)^2) / 2.
        values, value_cache = agent.value.forward(obs)
        error = values[:, 0] - targets
        value_grads = agent.value.backward(value_cache, error[:, None] / m)
        value_grads, value_norm = clip_by_global_norm(value_grads, s.max_grad_norm)
        self.value_optimizer.step(value_grads)

        return {
            "policy_loss": policy_loss,
            "value_loss": float(0.5 * np.mean(error * error)),
            "entropy": gaussian_entropy(policy.log_std),
            # Schulman's estimator of KL(old || new): mean((rho - 1) - log rho).
            "approx_kl": float(np.mean((ratio - 1.0) - log_ratio)),
            "clip_fraction": float(np.mean(np.abs(ratio - 1.0) > s.clip_range)),
            "policy_grad_norm": policy_norm,
            "value_grad_norm": value_norm,
        }

    def learn(
        self,
        total_steps: int,
        callback: Callable[[dict[str, Any]], bool | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Train for `total_steps` environment steps (all copies together; rounded
        up to whole rollouts). After every update `callback(stats)` is called with
        that iteration's diagnostics and episodes; returning True stops the
        training. Returns the list of every iteration's stats."""
        s = self.settings
        per_rollout = s.rollout_steps * self.n
        iterations = max(1, -(-total_steps // per_rollout))
        history = []
        start = time.perf_counter()
        for iteration in range(iterations):
            fraction = iteration / iterations
            rate = s.learning_rate * (1.0 - fraction if s.anneal else 1.0)
            rollout, episodes = self.collect()
            stats: dict[str, Any] = self.update(rollout, rate)
            stats.update(
                iteration=iteration + 1,
                steps=self.steps,
                seconds=time.perf_counter() - start,
                episodes=episodes,
            )
            history.append(stats)
            if callback is not None and callback(stats):
                break
        return history
