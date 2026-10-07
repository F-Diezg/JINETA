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
    """Integrate dy/dt = f(t, y) from t=0 to t_end with fixed-step RK4.

    Args:
        f: derivative function, f(t, y) -> dy/dt.
        y0: initial state at t = 0.
        dt: time step [s].
        t_end: final time [s]; the number of steps is round(t_end / dt).
        stop: optional condition stop(t, y) -> bool, checked after every step. When
            it returns True the run ends and that step is the last one returned (it
            is the first sample past the event, not the exact event).
        post_step: optional correction post_step(y) -> y, applied to the new state
            after every step and before `stop` is checked. Use it for constraints
            the integrator does not keep by itself, for example
            rigid_body.normalize_attitude to keep the quaternion at unit norm.

    Returns:
        (times, states), where states[i] is the state at times[i].
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
