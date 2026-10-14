"""Aerodynamic helpers shared by every vehicle model.

Aerodynamic forces depend on how the air hits the vehicle, so they are computed
from the airspeed (velocity relative to the air, v - wind) expressed in BODY axes.
The rigid-body state keeps the ground velocity in NED axes; this module converts.

A fixed-wing aircraft describes "how the air hits it" with three numbers instead of
the three body components (u, v, w): the airspeed V and two angles,
  - alpha (angle of attack): how far the air comes from below the nose, measured in
    the plane of symmetry (body x-z plane);
  - beta (sideslip angle): how far the air comes from the side, out of that plane.
Lift and drag tables are written as functions of alpha and beta, not of u, v, w.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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


@dataclass(frozen=True)
class AirData:
    """Airspeed magnitude and the two angles that say where the air comes from."""

    speed: float  # V = |v_air| [m/s]
    alpha: float  # angle of attack [rad], in (-pi, pi]
    beta: float  # sideslip angle [rad], in [-pi/2, pi/2]


def air_data(airspeed: npt.ArrayLike, min_speed: float = 1e-6) -> AirData:
    """Airspeed V, angle of attack alpha and sideslip beta from [u, v, w] in body axes.

        V     = sqrt(u^2 + v^2 + w^2)
        alpha = atan2(w, u)     the air in the plane of symmetry (x-z)
        beta  = asin(v / V)     the air out of that plane

    alpha > 0: the air comes from below the nose. beta > 0: the air comes from the
    right, i.e. the vehicle slides towards its right wing. atan2 (not atan) keeps
    the quadrant, so flying tail first gives alpha = pi instead of 0.

    Below `min_speed` the direction of the air is undefined: the angles are 0.

    Args:
        airspeed: airspeed in body axes [u, v, w] [m/s] (see `body_airspeed`).
        min_speed: airspeed below which the angles are not defined [m/s].
    """
    u, v, w = _three_components(airspeed)
    speed = float(np.sqrt(u * u + v * v + w * w))
    if speed < min_speed:
        return AirData(speed=speed, alpha=0.0, beta=0.0)
    # clip: rounding can push v / V a hair outside [-1, 1], where asin is undefined
    beta = float(np.arcsin(np.clip(v / speed, -1.0, 1.0)))
    return AirData(speed=speed, alpha=float(np.arctan2(w, u)), beta=beta)


def airspeed_from_air_data(speed: float, alpha: float, beta: float) -> np.ndarray:
    """Inverse of `air_data`: the body airspeed [u, v, w] for given V, alpha, beta.

    u = V cos(alpha) cos(beta),  v = V sin(beta),  w = V sin(alpha) cos(beta)
    """
    if speed < 0.0:
        raise ValueError("speed must be >= 0")
    return speed * np.array(
        [
            np.cos(alpha) * np.cos(beta),
            np.sin(beta),
            np.sin(alpha) * np.cos(beta),
        ]
    )


def _three_components(vector: npt.ArrayLike) -> np.ndarray:
    """The vector as a float array, checking it has exactly 3 components."""
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,):
        raise ValueError("the airspeed must have 3 components [u, v, w]")
    return vector
