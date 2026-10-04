"""Numerical integrators for ordinary differential equations (ODEs).

All integrators solve problems of the form dy/dt = f(t, y), where y is
the state vector and f returns its time derivative.
"""

from collections.abc import Callable

import numpy as np

# Type alias: a function that takes (time, state) and returns d(state)/dt
Derivative = Callable[[float, np.ndarray], np.ndarray]


def euler_step(f: Derivative, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance the state one step with the explicit Euler method (1st order)."""
    return y + dt * f(t, y)


def rk4_step(f: Derivative, t: float, y: np.ndarray, dt: float) -> np.ndarray:
    """Advance the state one step with the classic Runge-Kutta method (4th order)."""
    k1 = f(t, y)
    k2 = f(t + dt / 2, y + dt / 2 * k1)
    k3 = f(t + dt / 2, y + dt / 2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)