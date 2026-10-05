"""Fixed-step simulation loop."""

from collections.abc import Callable

import numpy as np

from jineta.core.integrators import Derivative, rk4_step

StopCondition = Callable[[float, np.ndarray], bool]


def run(
    f: Derivative,
    y0: np.ndarray,
    dt: float,
    t_end: float,
    stop: StopCondition | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate dy/dt = f(t, y) from t=0 with RK4.

    Returns (times, states), where states[i] is the state at times[i].
    Stops early if stop(t, y) returns True.
    """
    n_steps = round(t_end / dt)
    times = np.zeros(n_steps + 1)
    states = np.zeros((n_steps + 1, y0.size))
    states[0] = y0

    for i in range(n_steps):
        states[i + 1] = rk4_step(f, times[i], states[i], dt)
        times[i + 1] = times[i] + dt
        if stop is not None and stop(times[i + 1], states[i + 1]):
            return times[: i + 2], states[: i + 2]

    return times, states