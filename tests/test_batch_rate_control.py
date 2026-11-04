"""Tests for the batch body-rate loop: every drone against the reference
`RateController`, step by step and in closed loop, the exact discrete response,
and each drone's memory its own."""

import dataclasses

import numpy as np
import pytest

from jineta.core.environment import GRAVITY
from jineta.core.integrators import rk4_step
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.control import BatchRateController, rotor_speeds_for_wrench
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.rigid_body import normalize_attitudes
from jineta.jinete.control import RateController, RateGains
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_RATE_GAINS

DT = 0.01  # 100 Hz
WEIGHT = IRIS.mass * GRAVITY
NO_ROTOR_DRAG = dataclasses.replace(IRIS, rotor_drag_coefficient=0.0)


def random_quaternions(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def hovering_states(rng: np.random.Generator, n: int) -> np.ndarray:
    """n drones 10 m up, nearly level, turning at up to ~1 rad/s."""
    states = np.zeros((n, 13))
    states[:, 0:2] = rng.normal(size=(n, 2))
    states[:, 2] = -10.0
    tilt = rng.normal(size=(n, 3)) * 0.1
    states[:, 6] = 1.0
    states[:, 7:10] = tilt / 2  # small rotation vector -> quaternion
    states[:, 6:10] /= np.linalg.norm(states[:, 6:10], axis=1, keepdims=True)
    states[:, 10:13] = rng.normal(size=(n, 3)) * 0.5
    return states


def test_wrench_to_rotor_speeds_matches_the_mixer():
    """Thrust and moments from far too little to far too much (both clips)."""
    rng = np.random.default_rng(0)
    n = 500
    wrench = np.column_stack(
        [rng.uniform(-5.0, 40.0, size=n), rng.normal(size=(n, 3)) * 0.3]
    )
    speeds = rotor_speeds_for_wrench(IRIS, wrench)
    clipped = 0
    for i in range(n):
        thrusts = IRIS.rotor_thrusts(wrench[i])
        clipped += np.any((thrusts == 0.0) | (thrusts == IRIS.max_rotor_thrust))
        expected = IRIS.rotor_speeds_for(thrusts)
        assert np.allclose(speeds[i], expected, rtol=1e-12, atol=1e-9)
    assert 100 < clipped < n - 100  # clipped and free cases both tested


def test_updates_match_reference_step_by_step():
    """300 updates with random commands and states, the integrals saturating now
    and then: rotor speeds and integrals of every drone equal the reference's."""
    rng = np.random.default_rng(1)
    n, steps = 50, 300
    batch = BatchRateController(IRIS, IRIS_RATE_GAINS, DT, n)
    references = [RateController(IRIS, IRIS_RATE_GAINS, DT) for _ in range(n)]
    for _ in range(steps):
        states = hovering_states(rng, n)
        rates = rng.normal(size=(n, 3)) * 3.0
        thrusts = rng.uniform(0.0, 30.0, size=n)
        speeds = batch.update(rates, thrusts, states)
        for i, reference in enumerate(references):
            expected = reference.update(rates[i], thrusts[i], states[i])
            assert np.allclose(speeds[i], expected, rtol=1e-12, atol=1e-9)
            assert np.allclose(batch.integral[i], reference.integral, atol=1e-12)
    limit = IRIS_RATE_GAINS.integral_limit
    assert np.any(np.isclose(np.abs(batch.integral), limit))  # anti-windup reached


def test_a_drone_gives_the_same_numbers_alone_or_in_a_batch():
    rng = np.random.default_rng(2)
    n = 1000
    everyone = BatchRateController(IRIS, IRIS_RATE_GAINS, DT, n)
    alone = BatchRateController(IRIS, IRIS_RATE_GAINS, DT, 1)
    i = 637
    for _ in range(20):
        states = hovering_states(rng, n)
        rates = rng.normal(size=(n, 3))
        thrusts = rng.uniform(5.0, 25.0, size=n)
        speeds = everyone.update(rates, thrusts, states)
        mine = alone.update(rates[i : i + 1], thrusts[i : i + 1], states[i : i + 1])
        assert np.array_equal(mine[0], speeds[i])
        assert np.array_equal(alone.integral[0], everyone.integral[i])


def fly_batch(n, gains, params, start, commands, steps):
    """Fly n drones with the batch rate loop; commands(k) -> (rates, thrusts)."""
    vehicles = BatchMultirotor(params, n)
    loop = BatchRateController(params, gains, DT, n)
    states = start.copy()
    history = [states]
    for k in range(steps):
        rates, thrusts = commands(k)
        vehicles.rotor_speed_command = loop.update(rates, thrusts, states)
        states = normalize_attitudes(rk4_step(vehicles.derivative, k * DT, states, DT))
        history.append(states)
    return np.array(history)  # (steps + 1, n, 13)


def test_closed_loop_flights_match_reference():
    """3 s of flight with rate and thrust commands that change every 0.5 s, every
    drone its own: each follows its reference flight (Multirotor + RateController in
    pre_step) within 1e-9."""
    rng = np.random.default_rng(3)
    n, steps = 12, 300
    start = hovering_states(rng, n)
    rates = rng.normal(size=(6, n, 3)) * 1.5
    thrusts = WEIGHT * rng.uniform(0.8, 1.2, size=(6, n))

    def commands(k):
        return rates[k // 50], thrusts[k // 50]

    history = fly_batch(n, IRIS_RATE_GAINS, IRIS, start, commands, steps)

    for i in range(n):
        states = reference_flight(start[i], rates[:, i], thrusts[:, i], steps)
        assert np.allclose(history[:, i], states, rtol=0.0, atol=1e-9)


def reference_flight(start, rates, thrusts, steps):
    """One drone with the reference engine: Multirotor + RateController in
    pre_step, the commands changing every 50 steps."""
    vehicle = Multirotor(IRIS)
    loop = RateController(IRIS, IRIS_RATE_GAINS, DT)
    k = 0

    def pre_step(t, state):
        nonlocal k
        vehicle.rotor_speed_command = loop.update(
            rates[k // 50], thrusts[k // 50], state
        )
        k += 1

    _, states = run(
        vehicle.derivative,
        start,
        DT,
        steps * DT,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    return states


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_p_only_step_is_the_exact_discrete_first_order(axis):
    """The reference's closed-form check, for a batch: with P only and no rotor
    drag, omega_dot = kp (omega_cmd - omega_k) held for a step gives
    e_(k+1) = (1 - kp dt) e_k, for drones with different commands."""
    kp = IRIS_RATE_GAINS.kp
    gains = RateGains(kp=kp, ki=[0.0, 0.0, 0.0], integral_limit=[0.0, 0.0, 0.0])
    n, steps = 4, 50
    start = np.tile([0.0, 0.0, -100.0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0], (n, 1))
    targets = np.array([0.5, -0.3, 0.1, 0.0])
    command = np.zeros((n, 3))
    command[:, axis] = targets

    history = fly_batch(
        n,
        gains,
        NO_ROTOR_DRAG,
        start,
        lambda k: (command, np.full(n, WEIGHT)),
        steps,
    )
    factor = (1.0 - kp[axis] * DT) ** np.arange(steps + 1)
    expected = targets[None, :] * (1.0 - factor[:, None])
    assert np.allclose(history[:, :, 10 + axis], expected, atol=1e-9)
    others = [j for j in range(3) if j != axis]
    assert np.allclose(history[:, :, 10 + np.array(others)], 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "drones",
    [np.array([1, 3]), np.array([False, True, False, True]), None],
    ids=["indices", "mask", "all"],
)
def test_reset_forgets_only_the_chosen_drones(drones):
    loop = BatchRateController(IRIS, IRIS_RATE_GAINS, DT, 4)
    loop.integral[:] = 1.0
    loop.reset(drones)
    chosen = np.zeros(4, dtype=bool)
    chosen[slice(None) if drones is None else drones] = True
    assert np.all(loop.integral[chosen] == 0.0)
    assert np.all(loop.integral[~chosen] == 1.0)


def test_rejects_invalid_period_and_size():
    with pytest.raises(ValueError, match="dt"):
        BatchRateController(IRIS, IRIS_RATE_GAINS, 0.0, 3)
    with pytest.raises(ValueError, match="n"):
        BatchRateController(IRIS, IRIS_RATE_GAINS, DT, 0)
