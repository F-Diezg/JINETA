"""Tests for the numerical integrators using free fall (known exact solution)."""

import numpy as np

from jineta.core.integrators import euler_step, rk4_step

G = 9.81  # gravity [m/s^2]


def free_fall(t: float, y: np.ndarray) -> np.ndarray:
    """State y = [z, vz]. Returns its derivative [vz, -g]."""
    z, vz = y
    return np.array([vz, -G])


def simulate(step_fn, y0: np.ndarray, dt: float, t_end: float) -> np.ndarray:
    """Run a fixed-step simulation from t=0 to t_end and return the final state."""
    y = y0.copy()
    t = 0.0
    n_steps = round(t_end / dt)
    for _ in range(n_steps):
        y = step_fn(free_fall, t, y, dt)
        t += dt
    return y


def exact_free_fall(z0: float, t: float) -> float:
    return z0 - 0.5 * G * t**2


def test_rk4_free_fall_is_exact():
    y0 = np.array([100.0, 0.0])
    y_final = simulate(rk4_step, y0, dt=0.1, t_end=2.0)
    assert np.isclose(y_final[0], exact_free_fall(100.0, 2.0), atol=1e-9)


def test_euler_error_shrinks_with_smaller_dt():
    y0 = np.array([100.0, 0.0])
    exact = exact_free_fall(100.0, 2.0)
    error_coarse = abs(simulate(euler_step, y0, dt=0.1, t_end=2.0)[0] - exact)
    error_fine = abs(simulate(euler_step, y0, dt=0.01, t_end=2.0)[0] - exact)
    # First-order method: 10x smaller dt -> ~10x smaller error
    assert error_fine < error_coarse / 5