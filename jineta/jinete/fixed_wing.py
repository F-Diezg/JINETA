"""Fixed-wing aerodynamics: forces and moments from stability and control derivatives.

The air acts on an aircraft through the dynamic pressure qbar = 1/2 rho V^2 times a
reference area S, scaled by dimensionless coefficients:

    lift  L = qbar S C_L      drag   D = qbar S C_D      side force Y = qbar S C_Y
    roll  l = qbar S b C_ell  pitch  m = qbar S c C_m    yaw        n = qbar S b C_n

(b span, c mean chord). Each coefficient is a short sum of "derivatives": how much it
changes per radian of angle of attack alpha or sideslip beta, per unit of
nondimensional angular rate (p b / 2V, q c / 2V, r b / 2V) and per radian of control
surface deflection. The rates are made nondimensional so that the same coefficient
works at any speed: what matters is the angle the rotation adds to the local
airflow at the wing tip (p b/2 against V), not the rate alone.

Lift and drag are defined against the airflow: drag along it, lift perpendicular to
it. In the plane of symmetry the airflow comes at alpha below the nose, so they are
rotated into body axes by alpha (the stability axes):

    F_x = -D cos(alpha) + L sin(alpha),   F_z = -D sin(alpha) - L cos(alpha)

Model and equations as in Beard & McLain, Small Unmanned Aircraft (2012), ch. 4:
linear derivatives, a parabolic drag polar, and a smooth blend from the linear lift
curve to a flat plate past the stall. Side force and the moments use the sideslip
directly in body axes (good for small beta, as the source).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.core.inertia import inertia_tensor
from jineta.jinete.aero import air_data


@dataclass(frozen=True)
class ControlSurfaces:
    """Control surface deflections [rad], with the sign conventions of the source."""

    elevator: float = 0.0  # > 0: trailing edge down, pitches the nose down
    aileron: float = 0.0  # > 0: rolls right (right wing down)
    rudder: float = 0.0  # > 0: yaws the nose left


@dataclass(frozen=True)
class FixedWingParams:
    """Mass, geometry and aerodynamic derivatives of a fixed-wing aircraft.

    Names follow the source: c_lift_alpha is dC_L/dalpha [1/rad], c_m_q is dC_m/dq_hat
    with q_hat = q c / (2V), c_n_rudder is dC_n/d(rudder) [1/rad], and so on.
    c_ell is the rolling-moment coefficient (C_l in most books, renamed to avoid
    confusion with lift).
    """

    mass: float  # [kg]
    inertia: np.ndarray  # 3x3 about the centre of mass, body axes [kg·m^2]
    wing_area: float  # S [m^2]
    span: float  # b [m]
    chord: float  # c, mean aerodynamic chord [m]
    oswald: float  # e, span efficiency of the drag polar, (0, 1]
    # Lift: linear part, its blend into a flat plate past the stall, rate, elevator
    c_lift_0: float
    c_lift_alpha: float
    c_lift_q: float
    c_lift_elevator: float
    stall_sharpness: float  # M: how abruptly the blend switches [1/rad]
    stall_angle: float  # alpha_0: angle of attack where the blend is centred [rad]
    # Drag: parabolic polar C_D_p + C_L_lin^2 / (pi e AR), rate, elevator
    c_drag_p: float
    c_drag_q: float
    c_drag_elevator: float
    # Pitching moment
    c_m_0: float
    c_m_alpha: float
    c_m_q: float
    c_m_elevator: float
    # Side force
    c_y_0: float
    c_y_beta: float
    c_y_p: float
    c_y_r: float
    c_y_aileron: float
    c_y_rudder: float
    # Rolling moment
    c_ell_0: float
    c_ell_beta: float
    c_ell_p: float
    c_ell_r: float
    c_ell_aileron: float
    c_ell_rudder: float
    # Yawing moment
    c_n_0: float
    c_n_beta: float
    c_n_p: float
    c_n_r: float
    c_n_aileron: float
    c_n_rudder: float

    def __post_init__(self) -> None:
        for name in (
            "mass",
            "wing_area",
            "span",
            "chord",
            "stall_sharpness",
            "stall_angle",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 < self.oswald <= 1.0:
            raise ValueError("oswald must be in (0, 1]")
        if self.c_drag_p < 0.0:
            raise ValueError("c_drag_p must be >= 0")
        object.__setattr__(self, "inertia", inertia_tensor(self.inertia))

    @property
    def aspect_ratio(self) -> float:
        """AR = b^2 / S: long thin wings (high AR) make less induced drag."""
        return self.span**2 / self.wing_area


def stall_blend(alpha: float, sharpness: float, stall_angle: float) -> float:
    """Weight sigma in [0, 1] of the flat-plate lift: ~0 for |alpha| < alpha_0
    (attached flow, linear lift curve), ~1 beyond it (stalled wing).

        sigma = (1 + e^(-M(alpha - a0)) + e^(M(alpha + a0)))
                / ((1 + e^(-M(alpha - a0))) (1 + e^(M(alpha + a0))))
    """
    # clip the exponents: beyond ~700, exp overflows a float and sigma is 0 or 1 anyway
    below = np.exp(np.clip(-sharpness * (alpha - stall_angle), -700.0, 700.0))
    above = np.exp(np.clip(sharpness * (alpha + stall_angle), -700.0, 700.0))
    return float((1.0 + below + above) / ((1.0 + below) * (1.0 + above)))


def lift_coefficient(params: FixedWingParams, alpha: float) -> float:
    """Static lift coefficient C_L(alpha): the linear lift curve blended into a flat
    plate (C_L = 2 sin^2(alpha) cos(alpha) sign(alpha)) past the stall."""
    sigma = stall_blend(alpha, params.stall_sharpness, params.stall_angle)
    linear = params.c_lift_0 + params.c_lift_alpha * alpha
    flat_plate = 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    return float((1.0 - sigma) * linear + sigma * flat_plate)


def drag_coefficient(params: FixedWingParams, alpha: float) -> float:
    """Static drag coefficient: parasitic drag plus the induced drag of the linear
    lift, C_D = C_D_p + (C_L_0 + C_L_alpha alpha)^2 / (pi e AR)."""
    linear_lift = params.c_lift_0 + params.c_lift_alpha * alpha
    induced = linear_lift**2 / (np.pi * params.oswald * params.aspect_ratio)
    return float(params.c_drag_p + induced)


def aero_force_moment(
    params: FixedWingParams,
    airspeed: npt.ArrayLike,
    rates: npt.ArrayLike,
    surfaces: ControlSurfaces,
    density: float,
    min_speed: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Aerodynamic force and moment about the centre of mass, in body axes.

    Args:
        params: the aircraft.
        airspeed: body airspeed [u, v, w] [m/s] (see `aero.body_airspeed`).
        rates: body angular rates [p, q, r] [rad/s].
        surfaces: control surface deflections.
        density: air density [kg/m^3].
        min_speed: below this airspeed there is no aerodynamic load [m/s].

    Returns:
        (force [N], moment [N·m]), both 3-vectors in body axes.
    """
    air = air_data(airspeed, min_speed)
    if air.speed < min_speed:
        return np.zeros(3), np.zeros(3)
    alpha, beta, speed = air.alpha, air.beta, air.speed
    p, q, r = np.asarray(rates, dtype=float)
    b, c = params.span, params.chord
    p_hat, q_hat, r_hat = p * b / (2 * speed), q * c / (2 * speed), r * b / (2 * speed)
    qbar_s = 0.5 * density * speed**2 * params.wing_area
    de, da, dr = surfaces.elevator, surfaces.aileron, surfaces.rudder

    lift = qbar_s * (
        lift_coefficient(params, alpha)
        + params.c_lift_q * q_hat
        + params.c_lift_elevator * de
    )
    drag = qbar_s * (
        drag_coefficient(params, alpha)
        + params.c_drag_q * q_hat
        + params.c_drag_elevator * de
    )
    side = qbar_s * (
        params.c_y_0
        + params.c_y_beta * beta
        + params.c_y_p * p_hat
        + params.c_y_r * r_hat
        + params.c_y_aileron * da
        + params.c_y_rudder * dr
    )
    force = np.array(
        [
            -drag * np.cos(alpha) + lift * np.sin(alpha),
            side,
            -drag * np.sin(alpha) - lift * np.cos(alpha),
        ]
    )

    roll = (
        qbar_s
        * b
        * (
            params.c_ell_0
            + params.c_ell_beta * beta
            + params.c_ell_p * p_hat
            + params.c_ell_r * r_hat
            + params.c_ell_aileron * da
            + params.c_ell_rudder * dr
        )
    )
    pitch = (
        qbar_s
        * c
        * (
            params.c_m_0
            + params.c_m_alpha * alpha
            + params.c_m_q * q_hat
            + params.c_m_elevator * de
        )
    )
    yaw = (
        qbar_s
        * b
        * (
            params.c_n_0
            + params.c_n_beta * beta
            + params.c_n_p * p_hat
            + params.c_n_r * r_hat
            + params.c_n_aileron * da
            + params.c_n_rudder * dr
        )
    )
    return force, np.array([roll, pitch, yaw])
