"""Aerodynamic helpers shared by every vehicle model.

Aerodynamic forces depend on how the air hits the vehicle, so they are computed
from the airspeed (velocity relative to the air, v - wind) expressed in BODY axes.
The rigid-body state keeps the ground velocity in NED axes; this module converts.
"""

import numpy as np

from jineta.core import quaternion
from jineta.core.environment import WindModel, no_wind


def body_airspeed(t: float, state: np.ndarray, wind: WindModel = no_wind) -> np.ndarray:
    """Airspeed vector in body axes [m/s] for a 13-element rigid-body state.

    v_air_body = R(q)^T @ (v_ned - wind(t, position)). R rotates BODY -> NED and is
    orthogonal, so its transpose is its inverse and rotates NED -> BODY.

    Components: u (forward, along the nose), v (right wing), w (down). For example,
    w > 0 means the air comes from below the nose: positive angle of attack.
    """
    position = state[0:3]
    velocity = state[3:6]
    q = state[6:10]
    air_velocity_ned = velocity - wind(t, position)
    return quaternion.to_matrix(q).T @ air_velocity_ned
