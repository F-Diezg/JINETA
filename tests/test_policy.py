"""Tests for the policy: the Gaussian's log-probability, entropy and samples
against scipy.stats, its gradients against finite differences, the observation
normaliser against the statistics of all the data at once, and the agent saved
and loaded."""

from itertools import pairwise

import numpy as np
import pytest
from scipy import stats

from jineta.doma.policy import (
    Agent,
    GaussianPolicy,
    RunningNormalizer,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_log_prob_gradients,
)


def test_log_prob_and_entropy_against_scipy():
    rng = np.random.default_rng(0)
    mean = rng.standard_normal((50, 4))
    log_std = np.array([-1.0, -0.2, 0.0, 0.7])
    actions = mean + rng.standard_normal((50, 4)) * 2.0
    expected = stats.norm.logpdf(actions, mean, np.exp(log_std)).sum(axis=1)
    np.testing.assert_allclose(
        gaussian_log_prob(actions, mean, log_std), expected, rtol=0, atol=1e-12
    )
    assert gaussian_entropy(log_std) == pytest.approx(
        stats.norm.entropy(0.0, np.exp(log_std)).sum(), abs=1e-12
    )


def test_log_prob_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    mean = rng.standard_normal((6, 3))
    log_std = rng.uniform(-1.0, 0.5, 3)
    actions = mean + rng.standard_normal((6, 3))
    d_mean, d_log_std = gaussian_log_prob_gradients(actions, mean, log_std)
    h = 1e-6
    for i in range(3):
        step = np.zeros(3)
        step[i] = h
        numeric_mean = (
            gaussian_log_prob(actions, mean + step, log_std)
            - gaussian_log_prob(actions, mean - step, log_std)
        ) / (2 * h)
        numeric_std = (
            gaussian_log_prob(actions, mean, log_std + step)
            - gaussian_log_prob(actions, mean, log_std - step)
        ) / (2 * h)
        np.testing.assert_allclose(d_mean[:, i], numeric_mean, rtol=1e-7, atol=1e-8)
        np.testing.assert_allclose(d_log_std[:, i], numeric_std, rtol=1e-7, atol=1e-8)


def test_samples_follow_the_distribution():
    rng = np.random.default_rng(2)
    policy = GaussianPolicy.create(5, 4, (16,), rng, log_std=-0.5)
    policy.log_std[:] = [-1.0, -0.5, 0.0, 0.3]
    observation = rng.standard_normal((1, 5))
    observations = np.repeat(observation, 100_000, axis=0)
    actions, log_probs = policy.sample(observations, np.random.default_rng(3))
    mean = policy.mean(observation)[0]
    np.testing.assert_allclose(actions.mean(axis=0), mean, atol=0.02)
    np.testing.assert_allclose(actions.std(axis=0), np.exp(policy.log_std), rtol=0.01)
    np.testing.assert_allclose(
        log_probs,
        gaussian_log_prob(actions, policy.mean(observations), policy.log_std),
        atol=1e-12,
    )
    # The average of -log pi over samples is the entropy.
    assert -log_probs.mean() == pytest.approx(policy.entropy(), abs=0.02)
    again, _ = policy.sample(observations[:10], np.random.default_rng(3))
    np.testing.assert_allclose(again, actions[:10], rtol=0, atol=1e-14)


def test_a_new_policy_starts_near_zero_with_its_spread():
    rng = np.random.default_rng(4)
    policy = GaussianPolicy.create(43, 4, (64, 64), rng, log_std=-0.5)
    observations = rng.standard_normal((1000, 43))
    assert np.max(np.abs(policy.mean(observations))) < 0.1
    np.testing.assert_array_equal(policy.log_std, np.full(4, -0.5))
    assert len(policy.parameters) == 7  # 3 layers x (W, b) + log_std


def test_normaliser_in_pieces_equals_all_the_data():
    rng = np.random.default_rng(5)
    data = rng.normal([3.0, -200.0, 0.01], [10.0, 0.5, 1e-3], (1000, 3))
    normaliser = RunningNormalizer(3)
    cuts = [0, 1, 8, 9, 150, 151, 600, 1000]
    for a, b in pairwise(cuts):
        normaliser.update(data[a:b])
    assert normaliser.count == 1000
    np.testing.assert_allclose(normaliser.mean, data.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(normaliser.var, data.var(axis=0), rtol=1e-12)
    scaled = normaliser(data)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    # epsilon (1e-8) is felt by the third column, whose variance is ~1e-6.
    var = data.var(axis=0)
    np.testing.assert_allclose(
        scaled.std(axis=0), np.sqrt(var / (var + 1e-8)), rtol=1e-9
    )
    assert scaled.std(axis=0)[2] < 0.999


def test_normaliser_clips_and_survives_a_constant_input():
    normaliser = RunningNormalizer(2, clip=5.0)
    normaliser.update(np.array([[1.0, 7.0], [3.0, 7.0]]))
    np.testing.assert_allclose(normaliser([[2.0, 7.0]]), [[0.0, 0.0]])
    np.testing.assert_allclose(normaliser([[100.0, 8.0]]), [[5.0, 5.0]])
    fresh = RunningNormalizer(2)
    np.testing.assert_allclose(fresh([[1.5, -2.0]]), [[1.5, -2.0]], rtol=1e-8)


def test_agent_acts_with_the_mean_and_saves_bit_for_bit(tmp_path):
    rng = np.random.default_rng(6)
    agent = Agent.create(7, 4, (16, 16), rng)
    agent.normalizer.update(rng.normal(2.0, 3.0, (100, 7)))
    agent.info = {"course": "First course", "steps": 1234}
    observations = rng.normal(2.0, 3.0, (5, 7))
    np.testing.assert_array_equal(
        agent.act(observations),
        agent.policy.mean(agent.normalizer(observations)),
    )
    agent.save(tmp_path / "agent.npz")
    copy = Agent.load(tmp_path / "agent.npz")
    np.testing.assert_array_equal(copy.act(observations), agent.act(observations))
    np.testing.assert_array_equal(copy.values(observations), agent.values(observations))
    np.testing.assert_array_equal(
        copy.act(observations, np.random.default_rng(7)),
        agent.act(observations, np.random.default_rng(7)),
    )
    assert copy.info == {"course": "First course", "steps": 1234}
    assert copy.normalizer.count == 100


def test_rejects_bad_input():
    rng = np.random.default_rng(8)
    with pytest.raises(ValueError):
        RunningNormalizer(0)
    with pytest.raises(ValueError):
        RunningNormalizer(2).update(np.ones((3, 3)))
    with pytest.raises(ValueError):
        RunningNormalizer(2).update(np.ones((0, 2)))
    policy = GaussianPolicy.create(3, 2, (4,), rng)
    with pytest.raises(ValueError):
        GaussianPolicy(policy.network, np.zeros(3))
