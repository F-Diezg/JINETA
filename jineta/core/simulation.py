"""Fixed-step simulation loop."""

from collections.abc import Callable

import numpy as np

from jineta.core.integrators import Derivative, rk4_step

StopCondition = Callable[[float, np.ndarray], bool]
PostStep = Callable[[np.ndarray], np.ndarray]

def run(
    f: Derivative,
    y0: np.ndarray,
    dt: float,
    t_end: float,
    stop: StopCondition | None = None,
    post_step: PostStep | None = None,
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
        y_next = rk4_step(f, times[i], states[i], dt)
        states[i + 1] = post_step(y_next) if post_step is not None else y_next
        times[i + 1] = times[i] + dt
        if stop is not None and stop(times[i + 1], states[i + 1]):
            return times[: i + 2], states[: i + 2]

    return times, states