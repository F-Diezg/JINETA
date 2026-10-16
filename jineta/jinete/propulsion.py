"""Electric propulsion: a DC motor turning a propeller, from the throttle to thrust.

The throttle sets the voltage on the motor, V_in = throttle * V_max. The motor and the
propeller then settle at the rotor speed Omega where the torque the motor gives
equals the torque the propeller asks for (their dynamics, a few tens of ms, are not
modelled: the speed is reached instantly).

Motor (brushed-DC model, also used for brushless motors with their ESC):

    current  i   = (V_in - K Omega) / R      K Omega: back-EMF, the voltage the
                                             spinning motor generates against V_in
    torque   Q_m = K (i - i0)                i0: no-load current (friction, losses)

K is the motor constant in SI units: the same number is the back-EMF constant
[V·s/rad] and the torque constant [N·m/A] (energy conservation: K i Omega is both the
electrical power converted and the mechanical power produced). Datasheets give it as
KV [RPM/V]: K = 60 / (2 pi KV).

Propeller (fits of measured data, as functions of the advance ratio J):

    J = V / (n D)      n = Omega / (2 pi) [rev/s], V the axial airspeed
    T = rho n^2 D^4 C_T(J)      C_T = C_T0 + C_T1 J + C_T2 J^2
    Q = rho n^2 D^5 C_Q(J)      C_Q = C_Q0 + C_Q1 J + C_Q2 J^2

J says how far the aircraft moves per propeller turn, in diameters. With J small the
blades meet the air at a large angle and push hard; as J grows the air reaches them
more head-on and they push less; past some J they brake, like a windmill.
Multiplying out, n^2 C_T(J) needs no division by n, so the same formulas hold for a
stopped propeller (n = 0):

    T = rho (D^4 C_T0 n^2 + D^3 C_T1 V n + D^2 C_T2 V^2)     (same for Q with D^5...)

Setting Q_m = Q gives a quadratic a Omega^2 + b Omega + c = 0 for the rotor speed:

    a = rho D^5 C_Q0 / (2 pi)^2
    b = rho D^4 C_Q1 V / (2 pi) + K^2 / R
    c = rho D^3 C_Q2 V^2 - K V_in / R + K i0

and the physical speed is the positive root. Model and parameters from the
propulsion addendum of Beard & McLain, Small Unmanned Aircraft (McLain).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PropulsionParams:
    """A DC motor driving a fixed-pitch propeller: only data, validated once."""

    diameter: float  # D, propeller diameter [m]
    motor_constant: float  # K [V·s/rad] = [N·m/A]; K = 60 / (2 pi KV[RPM/V])
    resistance: float  # R, motor winding resistance [ohm]
    no_load_current: float  # i0 [A]
    max_voltage: float  # V_max, battery voltage at full throttle [V]
    c_thrust: tuple[float, float, float]  # (C_T0, C_T1, C_T2) of C_T(J)
    c_torque: tuple[float, float, float]  # (C_Q0, C_Q1, C_Q2) of C_Q(J)

    def __post_init__(self) -> None:
        for name in ("diameter", "motor_constant", "resistance", "max_voltage"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.no_load_current < 0.0:
            raise ValueError("no_load_current must be >= 0")
        for name in ("c_thrust", "c_torque"):
            values = tuple(float(x) for x in getattr(self, name))
            if len(values) != 3:
                raise ValueError(f"{name} must have 3 coefficients")
            object.__setattr__(self, name, values)
        if self.c_torque[0] <= 0.0:
            # a = 0 would turn the quadratic for the rotor speed into a linear one
            raise ValueError("c_torque[0] (static torque coefficient) must be > 0")


@dataclass(frozen=True)
class PropellerOutput:
    """What the propeller does at one operating point."""

    thrust: float  # T along the propeller axis [N]; < 0 when it brakes
    torque: float  # Q the air exerts against the rotation [N·m]
    rotor_speed: float  # Omega [rad/s], >= 0


def rotor_speed(
    params: PropulsionParams, axial_speed: float, throttle: float, density: float
) -> float:
    """Rotor speed Omega [rad/s] where motor torque = propeller torque.

    Args:
        params: motor and propeller.
        axial_speed: airspeed along the propeller axis [m/s]; values < 0 (reverse
            flow, outside the data of the fits) are taken as 0.
        throttle: in [0, 1]; values outside are clipped.
        density: air density [kg/m^3].
    """
    speed = max(axial_speed, 0.0)
    v_in = params.max_voltage * float(np.clip(throttle, 0.0, 1.0))
    d = params.diameter
    k, r = params.motor_constant, params.resistance
    cq0, cq1, cq2 = params.c_torque
    two_pi = 2.0 * np.pi
    a = density * d**5 * cq0 / two_pi**2
    b = density * d**4 * cq1 * speed / two_pi + k**2 / r
    c = density * d**3 * cq2 * speed**2 - k * v_in / r + k * params.no_load_current
    omega = (-b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
    # A motor below its no-load voltage cannot turn the propeller: it stays stopped
    # (the root would be a small negative speed, the motor spinning backwards).
    return float(max(omega, 0.0))


def propeller_thrust_torque(
    params: PropulsionParams, axial_speed: float, throttle: float, density: float
) -> PropellerOutput:
    """Thrust and torque of the propeller for a throttle setting and an airspeed.

    Args: as `rotor_speed`.
    """
    speed = max(axial_speed, 0.0)
    omega = rotor_speed(params, speed, throttle, density)
    n = omega / (2.0 * np.pi)
    d = params.diameter
    ct0, ct1, ct2 = params.c_thrust
    cq0, cq1, cq2 = params.c_torque
    thrust = density * (
        d**4 * ct0 * n**2 + d**3 * ct1 * speed * n + d**2 * ct2 * speed**2
    )
    torque = density * (
        d**5 * cq0 * n**2 + d**4 * cq1 * speed * n + d**3 * cq2 * speed**2
    )
    return PropellerOutput(float(thrust), float(torque), omega)
