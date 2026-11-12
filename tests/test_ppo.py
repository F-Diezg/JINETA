"""Tests for PPO: GAE by hand and in its limits, truncated against terminated
episodes, two problems with a known optimum (a continuous bandit and the linear-
quadratic regulator of a double integrator, whose optimal gain the Riccati
equation gives), the same seed giving the same training, and a short training
on the course."""

import copy

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.ppo import PPO, PPOSettings, gae
from jineta.picadero.env import CourseEnv


class Space:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class LinearSystemEnv:
    """n double integrators (x'' = u) sampled every dt, with the shape of
    CourseEnv: the state (x, v) is the observation, the action a in [-1, 1] is
    u = u_max a, the reward -(q_x x^2 + q_v v^2 + r u^2); every episode starts
    at a random state and is truncated after `time_limit` steps (it never
    terminates), so the optimum is the infinite-horizon one only if the
    truncations are completed with the critic's value."""

    def __init__(self, n, time_limit=8, dt=0.1, u_max=4.0, q=(1.0, 0.1), r=0.1):
        self.n, self.time_limit, self.dt, self.u_max = n, time_limit, dt, u_max
        self.q, self.r = np.array(q), r
        self.action_space = Space((1,))

    def _starts(self, k: int) -> np.ndarray:
        return self.rng.uniform(-1.0, 1.0, (k, 2))

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.state = self._starts(self.n)
        self.steps = np.zeros(self.n, dtype=int)
        self.returns = np.zeros(self.n)
        return self.state.copy(), {}

    def step(self, action):
        u = self.u_max * np.clip(np.asarray(action)[:, 0], -1.0, 1.0)
        x, v = self.state[:, 0], self.state[:, 1]
        reward = -(self.q[0] * x * x + self.q[1] * v * v + self.r * u * u)
        self.state = np.column_stack([x + self.dt * v, v + self.dt * u])
        self.steps += 1
        self.returns += reward
        truncated = self.steps >= self.time_limit
        terminated = np.zeros(self.n, dtype=bool)
        info = {
            "final_observation": self.state.copy(),
            "episode_return": self.returns.copy(),
            "episode_steps": self.steps.copy(),
        }
        if np.any(truncated):
            self.state[truncated] = self._starts(int(truncated.sum()))
            self.steps[truncated] = 0
            self.returns[truncated] = 0.0
        return self.state.copy(), reward, terminated, truncated, info

    def lqr_gain(self, gamma: float) -> np.ndarray:
        """The optimal u = -K s of the discounted infinite-horizon problem
        (discounting folded into A and B by sqrt(gamma))."""
        a = np.sqrt(gamma) * np.array([[1.0, self.dt], [0.0, 1.0]])
        b = np.sqrt(gamma) * np.array([[0.0], [self.dt]])
        q, r = np.diag(self.q), np.array([[self.r]])
        p = solve_discrete_are(a, b, q, r)
        return np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)[0]


class BanditEnv:
    """One-step episodes (terminated at once): a context c in [-1, 1]^2, the
    reward -(a - W c)^2 summed over two action numbers; the best action is W c."""

    W = np.array([[0.5, -0.3], [0.2, 0.4]])

    def __init__(self, n):
        self.n = n
        self.action_space = Space((2,))

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.context = self.rng.uniform(-1.0, 1.0, (self.n, 2))
        return self.context.copy(), {}

    def step(self, action):
        error = np.asarray(action) - self.context @ self.W.T
        reward = -np.sum(error * error, axis=1)
        done = np.ones(self.n, dtype=bool)
        info = {
            "final_observation": self.context.copy(),
            "episode_return": reward.copy(),
            "episode_steps": np.ones(self.n, dtype=int),
        }
        self.context = self.rng.uniform(-1.0, 1.0, (self.n, 2))
        return self.context.copy(), reward, done, ~done, info


class ScriptedEnv:
    """Two copies with a fixed script: observations are (step, copy, 1); copy 0
    terminates at step 2, copy 1 is truncated at step 3 (its last observation
    (99, 99, 99)); each is restarted at (-1, copy, 1)."""

    def __init__(self):
        self.n = 2
        self.action_space = Space((1,))

    def reset(self, seed=None):
        self.t = 0
        return self._observe(), {}

    def _observe(self):
        return np.array([[self.t, 0.0, 1.0], [self.t, 1.0, 1.0]])

    def step(self, action):
        self.t += 1
        terminated = np.array([self.t == 2, False])
        truncated = np.array([False, self.t == 3])
        final = self._observe()
        final[1] = 99.0
        observations = self._observe()
        observations[terminated | truncated, 0] = -1.0
        info = {
            "final_observation": final,
            "episode_return": np.zeros(2),
            "episode_steps": np.full(2, self.t),
        }
        return observations, np.full(2, 0.5), terminated, truncated, info


def test_the_experience_completes_each_step_as_it_ended():
    ppo = PPO(ScriptedEnv(), PPOSettings(rollout_steps=4), seed=0)
    during = copy.deepcopy(ppo.agent.normalizer)  # fresh, frozen in the rollout
    rollout, episodes = ppo.collect()
    agent = ppo.agent

    def value(observation):
        return agent.value(during(np.array([observation], dtype=float)))[0, 0]

    # Copy 0: steps 1 and 3 go on, step 2 terminated (nothing after it).
    assert rollout.next_values[0, 0] == pytest.approx(value([1, 0, 1]), abs=1e-13)
    assert rollout.next_values[1, 0] == 0.0
    assert rollout.next_values[2, 0] == pytest.approx(value([3, 0, 1]), abs=1e-13)
    # Copy 1: step 3 truncated: the value of its last observation, not of the
    # restarted one.
    assert rollout.next_values[2, 1] == pytest.approx(value([99, 99, 99]), abs=1e-13)
    assert abs(rollout.next_values[2, 1] - value([-1, 1, 1])) > 1e-3
    assert rollout.next_values[1, 1] == pytest.approx(value([2, 1, 1]), abs=1e-13)
    np.testing.assert_array_equal(
        rollout.ends, [[False, False], [True, False], [False, True], [False, False]]
    )
    # values[t] is the value at the start of step t (after step t - 1).
    np.testing.assert_allclose(rollout.values[1:], np.vstack(
        [[value([1, 0, 1]), value([1, 1, 1])],
         [value([-1, 0, 1]), value([2, 1, 1])],
         [value([3, 0, 1]), value([-1, 1, 1])]]
    ), rtol=0, atol=1e-13)  # fmt: skip
    assert episodes.lengths == [2, 3]
    assert agent.normalizer.count == 8


# ---- GAE ---------------------------------------------------------------------------


def test_gae_by_hand():
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.5], [1.0], [1.5]])
    next_values = np.array([[1.0], [1.5], [2.0]])
    ends = np.zeros((3, 1), dtype=bool)
    g, lam = 0.9, 0.5
    d = rewards[:, 0] + g * next_values[:, 0] - values[:, 0]  # 1.4, 2.35, 4.3
    a2 = d[2]
    a1 = d[1] + g * lam * a2
    a0 = d[0] + g * lam * a1
    np.testing.assert_allclose(
        gae(rewards, values, next_values, ends, g, lam)[:, 0], [a0, a1, a2]
    )


def test_gae_with_lambda_one_is_the_monte_carlo_return_minus_the_value():
    rng = np.random.default_rng(0)
    T, n, g = 20, 3, 0.95
    rewards = rng.standard_normal((T, n))
    values = rng.standard_normal((T, n))
    last = rng.standard_normal(n)  # the value after the last step
    next_values = np.vstack([values[1:], last])
    ends = np.zeros((T, n), dtype=bool)
    advantages = gae(rewards, values, next_values, ends, g, 1.0)
    for t in range(T):
        discounts = g ** np.arange(T - t)
        returns = (discounts[:, None] * rewards[t:]).sum(axis=0) + g ** (T - t) * last
        np.testing.assert_allclose(advantages[t], returns - values[t], atol=1e-12)


def test_gae_with_lambda_zero_is_one_step():
    rng = np.random.default_rng(1)
    r, v, nv = (rng.standard_normal((5, 2)) for _ in range(3))
    ends = np.zeros((5, 2), dtype=bool)
    np.testing.assert_allclose(gae(r, v, nv, ends, 0.9, 0.0), r + 0.9 * nv - v)


def test_an_episode_end_cuts_the_sum():
    """Two episodes in a row of one environment: the first's advantages do not
    see the second's rewards; a truncated end adds the value of its last
    observation (given in next_values), a terminated one adds nothing."""
    rewards = np.array([[1.0], [1.0], [10.0], [10.0]])
    values = np.zeros((4, 1))
    ends = np.array([[False], [True], [False], [False]])
    g = 0.5
    terminated = gae(rewards, values, np.zeros((4, 1)), ends, g, 1.0)
    np.testing.assert_allclose(terminated[:2, 0], [1.0 + g * 1.0, 1.0])
    next_values = np.array([[0.0], [4.0], [0.0], [0.0]])  # truncated: V(last) = 4
    truncated = gae(rewards, values, next_values, ends, g, 1.0)
    np.testing.assert_allclose(truncated[:2, 0], [1.0 + g * (1.0 + g * 4.0), 3.0])
    np.testing.assert_allclose(truncated[2:, 0], terminated[2:, 0])


# ---- problems with a known optimum --------------------------------------------------


def test_learns_the_best_action_of_a_continuous_bandit():
    env = BanditEnv(256)
    settings = PPOSettings(
        rollout_steps=8, epochs=10, minibatch_size=512, gamma=0.0,
        learning_rate=3e-3, hidden=(16,), log_std=-1.0,
    )  # fmt: skip
    ppo = PPO(env, settings, seed=0)
    ppo.learn(150_000)
    contexts = np.random.default_rng(1).uniform(-0.8, 0.8, (200, 2))
    error = ppo.agent.act(contexts) - contexts @ BanditEnv.W.T
    assert np.sqrt(np.mean(error**2)) < 0.03


def lqr_training(seed: int = 0, steps: int = 1_000_000) -> tuple[np.ndarray, PPO]:
    """PPO on the double integrator; returns the gain of the learned
    deterministic policy (a least-squares plane through its actions over
    |x|, |v| <= 0.3, in u units) and the training."""
    env = LinearSystemEnv(64)
    gamma = 0.9
    settings = PPOSettings(
        rollout_steps=32, epochs=10, minibatch_size=512, gamma=gamma,
        learning_rate=1e-3, hidden=(16, 16), log_std=-1.0,
    )  # fmt: skip
    ppo = PPO(env, settings, seed=seed)
    ppo.learn(steps)
    grid = np.linspace(-0.3, 0.3, 7)
    states = np.array([[x, v] for x in grid for v in grid])
    actions = ppo.agent.act(states)[:, 0] * env.u_max
    plane = np.column_stack([states, np.ones(len(states))])
    coefficients, *_ = np.linalg.lstsq(plane, actions, rcond=None)
    return -coefficients[:2], ppo


def test_learns_the_optimal_regulator_of_a_double_integrator():
    """The learned feedback matches the LQR gain of the discounted infinite-
    horizon problem (Riccati), although every episode is cut after 8 steps:
    only if the truncations are completed with the critic's value (treating
    them as ends gives a gain about a third of it, measured)."""
    gain, ppo = lqr_training()
    optimum = LinearSystemEnv(1).lqr_gain(0.9)  # [1.723, 1.742]
    np.testing.assert_allclose(gain, optimum, rtol=0.05)
    assert ppo.steps == 1_001_472  # 489 rollouts of 32 x 64 steps


def test_the_same_seed_gives_the_same_training():
    """Bit for bit on the same computer: one seeded generator for everything."""
    first = PPO(LinearSystemEnv(16), PPOSettings(rollout_steps=8, minibatch_size=32))
    second = PPO(LinearSystemEnv(16), PPOSettings(rollout_steps=8, minibatch_size=32))
    a = first.learn(2_000)
    b = second.learn(2_000)
    for p, q in zip(
        first.agent.policy.parameters + first.agent.value.parameters,
        second.agent.policy.parameters + second.agent.value.parameters,
        strict=True,
    ):
        np.testing.assert_array_equal(p, q)
    assert [s["policy_loss"] for s in a] == [s["policy_loss"] for s in b]
    other = PPO(
        LinearSystemEnv(16), PPOSettings(rollout_steps=8, minibatch_size=32), seed=1
    )
    other.learn(2_000)
    assert not np.array_equal(
        other.agent.policy.log_std + other.agent.policy.network.weights[0][0, 0],
        first.agent.policy.log_std + first.agent.policy.network.weights[0][0, 0],
    )


def test_learning_rate_falls_to_zero_and_stats_are_reported():
    ppo = PPO(
        LinearSystemEnv(8),
        PPOSettings(rollout_steps=4, minibatch_size=16, learning_rate=1e-3),
    )
    history = ppo.learn(8 * 4 * 5)
    rates = [s["learning_rate"] for s in history]
    np.testing.assert_allclose(rates, [1e-3, 8e-4, 6e-4, 4e-4, 2e-4])
    for key in (
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
        "explained_variance", "std", "steps", "seconds",
    ):  # fmt: skip
        assert key in history[-1]
    assert history[-1]["steps"] == 160 and ppo.steps == 160
    stopped = PPO(LinearSystemEnv(8), PPOSettings(rollout_steps=4)).learn(
        10_000, callback=lambda stats: stats["iteration"] == 3
    )
    assert len(stopped) == 3


def test_a_short_training_on_the_course():
    """Two rollouts of 8 drones on the first course: the shapes fit, crashed
    episodes are reported with the info asked for, and a seed repeats it."""
    course = load_course(COURSE_FILE)
    settings = PPOSettings(rollout_steps=64, minibatch_size=256, epochs=2)

    def train():
        env = CourseEnv(course, n=8)
        ppo = PPO(env, settings, seed=4, episode_info=("status", "lap_time"))
        return ppo, ppo.learn(2 * 64 * 8)

    ppo, history = train()
    assert len(history) == 2 and ppo.steps == 1024
    episodes = [e for s in history for e in s["episodes"].infos]
    assert len(episodes) > 0  # untrained drones crash within ~1 s (64 steps)
    assert all(set(e) == {"status", "lap_time"} for e in episodes)
    assert ppo.agent.normalizer.count == 1024
    again, _ = train()
    np.testing.assert_array_equal(
        again.agent.policy.network.weights[0], ppo.agent.policy.network.weights[0]
    )


def test_rejects_bad_settings():
    for bad in (
        {"rollout_steps": 0}, {"gamma": 1.5}, {"gae_lambda": -0.1},
        {"clip_range": 0.0}, {"learning_rate": 0.0}, {"max_grad_norm": 0.0},
        {"entropy_coef": -1.0}, {"hidden": ()},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            PPOSettings(**bad)
