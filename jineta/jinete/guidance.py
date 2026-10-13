"""Guidance laws: from the positions of a pursuer and a target to a velocity command.

A guidance law only decides WHERE to go; the autopilot (control.py) decides how to
get there. The two meet at one interface: the NED velocity command that the
autopilot follows. Swapping the guidance law (pure pursuit now, proportional
navigation in phase 10) therefore never touches the control loops.
"""

import numpy as np
import numpy.typing as npt


def pure_pursuit_velocity(
    pursuer_position: npt.ArrayLike,
    target_position: npt.ArrayLike,
    speed: float,
    min_distance: float = 1e-6,
) -> np.ndarray:
    """Pure pursuit: fly at `speed` straight at where the target is right now.

    v_cmd = speed * (r_target - r_pursuer) / |r_target - r_pursuer|

    The unit vector is the line of sight (LOS). It is not defined when both
    positions coincide, so closer than `min_distance` the command is zero.

    Args:
        pursuer_position: NED position of the pursuer [m].
        target_position: NED position of the target [m].
        speed: commanded speed [m/s], >= 0.
        min_distance: below this separation the LOS is undefined [m].

    Returns:
        NED velocity command [m/s].
    """
    if speed < 0.0:
        raise ValueError("speed must be >= 0")
    pursuer = np.asarray(pursuer_position, dtype=float)
    target = np.asarray(target_position, dtype=float)
    if pursuer.shape != (3,) or target.shape != (3,):
        raise ValueError("positions must have 3 components (NED)")

    line_of_sight = target - pursuer
    distance = np.linalg.norm(line_of_sight)
    if distance < min_distance:
        return np.zeros(3)
    return speed * line_of_sight / distance
