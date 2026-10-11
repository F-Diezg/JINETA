"""Tests for the simulation loop hooks (pre_step: discrete control, zero-order hold)."""

import dataclasses

import numpy as np

from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS


def test_pre_step_is_called_once_per_step_with_the_start_of_the_step():
    calls = []

    def recorder(t, y):
        calls.append((t, y.copy()))

    def f(t, y):
        return np.array([1.0])

    times, states = run(f, np.array([0.0]), dt=0.1, t_end=1.0, pre_step=recorder)

    assert len(calls) == len(times) - 1  # no call after the last step
    for (t, y), t_start, y_start in zip(calls, times[:-1], states[:-1], strict=True):
        assert t == t_start
        assert np.array_equal(y, y_start)


def test_pre_step_cannot_alter_the_history():
    def vandal(t, y):
        y[:] = 999.0

    def f(t, y):
        return np.array([1.0])

    _, states = run(f, np.array([0.0]), dt=0.1, t_end=1.0, pre_step=vandal)
    assert np.allclose(states[:, 0], np.linspace(0.0, 1.0, 11))


def test_pre_step_none_changes_nothing():
    def f(t, y):
        return -y

    _, a = run(f, np.array([1.0]), dt=0.1, t_end=2.0)
    _, b = run(f, np.array([1.0]), dt=0.1, t_end=2.0, pre_step=None)
    assert np.array_equal(a, b)


def test_command_is_held_during_the_step():
    """dy/dt = u, with u sampled at the start of each step and held (zero-order
    hold). Sampling u = t gives a staircase, so y is the left Riemann sum of t:
    y(T) = dt^2 * n(n-1)/2, not the exact integral T^2/2."""
    dt, n_steps = 0.1, 10
    held = {"u": 0.0}
    seen = []

    def controller(t, y):
        held["u"] = t

    def f(t, y):
        seen.append((t, held["u"]))
        return np.array([held["u"]])

    _, states = run(f, np.array([0.0]), dt=dt, t_end=n_steps * dt, pre_step=controller)

    for t_eval, u in seen:  # every RK4 stage sees the value sampled at its step start
        assert u <= t_eval < u + dt + 1e-12
    assert np.isclose(states[-1, 0], dt**2 * n_steps * (n_steps - 1) / 2)
    assert not np.isclose(states[-1, 0], (n_steps * dt) ** 2 / 2)


def test_motors_cut_at_one_second_then_free_fall():
    """Hover for 1 s, then a controller switches the motors off: from then on the
    quad falls from rest, z = z0 + g (t - 1)^2 / 2 (drag removed: exact parabola)."""
    params = dataclasses.replace(
        IRIS, rotor_drag_coefficient=0.0, body_drag_area=np.zeros(3)
    )
    quad = Multirotor(params)
    hover = params.rotor_speeds_for(np.full(4, params.mass * GRAVITY / 4))

    def controller(t, state):
        # Sample times accumulate rounding error (0.1 + 0.1 + ...), hence the margin.
        quad.rotor_speed_command = hover if t < 1.0 - 1e-9 else np.zeros(4)

    y0 = np.array([0, 0, -100.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    times, states = run(
        quad.derivative,
        y0,
        dt=0.01,
        t_end=2.0,
        pre_step=controller,
        post_step=normalize_attitude,
    )

    hovering = times <= 1.0 + 1e-9
    assert np.allclose(states[hovering, 2], -100.0, atol=1e-9)
    falling = ~hovering
    expected_z = -100.0 + 0.5 * GRAVITY * (times[falling] - 1.0) ** 2
    assert np.allclose(states[falling, 2], expected_z, atol=1e-9)
