"""Trim: the controls and attitude that keep a fixed-wing aircraft in steady flight.

In steady straight and level flight nothing changes: the velocity is constant and
the aircraft does not rotate. So the three forces (gravity included) and the three
moments on it must all vanish: six equations. The unknowns are what the pilot and
the aircraft can choose:

    alpha, beta                          where the air comes from
    elevator, aileron, rudder, throttle  the controls
    (roll angle phi)

That is seven unknowns for six equations, so one is fixed: wings level, phi = 0.
With the nose at pitch theta = alpha the velocity is horizontal. The lateral
unknowns are not all zero: the propeller torque rolls the aircraft, and the aileron
(with a little rudder and sideslip) has to hold it.

The six equations are nonlinear (lift goes with V^2, the thrust with the propeller
model, the angles through sines and cosines), so they are solved numerically:
`scipy.optimize.root` with Powell's hybrid method (MINPACK), a Newton-like
iteration. From a guess x it linearises the equations, f(x + dx) ~ f(x) + J dx,
steps to where that linear model is zero, and repeats until f(x) is ~0.

The guess matters. The propulsion model clips the throttle to [0, 1], so beyond 1
the thrust stops changing and the iteration has no slope to follow: started from a
fixed guess it can jump past full throttle and get lost. So it starts from a
textbook estimate instead (lift = weight, pitching moment = 0, thrust = drag), which
is already close to the answer.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, root

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.jinete.aero import airspeed_from_air_data
from jineta.jinete.fixed_wing import ControlSurfaces, FixedWing, drag_coefficient
from jineta.jinete.propulsion import propeller_thrust_torque


@dataclass(frozen=True)
class Trim:
    """A trimmed flight condition: the state and the controls that hold it."""

    state: np.ndarray  # 13-element rigid-body state
    surfaces: ControlSurfaces
    throttle: float  # in [0, 1]
    alpha: float  # angle of attack [rad]
    beta: float  # sideslip [rad]

    def __post_init__(self) -> None:
        state = np.array(self.state, dtype=float)
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


def level_flight_state(
    aircraft: FixedWing,
    speed: float,
    alpha: float,
    beta: float,
    altitude: float,
    course: float,
) -> np.ndarray:
    """State of wings-level flight with horizontal airspeed, no rotation.

    The nose is at pitch theta = alpha (so the air velocity is horizontal) and at
    yaw psi = course - beta (so the air velocity points along `course`). The ground
    velocity is the air velocity plus the aircraft's wind at t = 0 at that point.
    """
    position = np.array([0.0, 0.0, -altitude])
    attitude = quaternion.from_euler(0.0, alpha, course - beta)
    air_velocity = quaternion.to_matrix(attitude) @ airspeed_from_air_data(
        speed, alpha, beta
    )
    velocity = air_velocity + np.asarray(aircraft.wind(0.0, position), dtype=float)
    return np.concatenate([position, velocity, attitude, np.zeros(3)])


def longitudinal_estimate(
    aircraft: FixedWing, speed: float, altitude: float
) -> tuple[float, float, float]:
    """Textbook estimate of (alpha, elevator, throttle) for level flight.

    Lift = weight and C_m = 0 are two linear equations in alpha and elevator (the
    linear lift curve, no rotation); then the throttle whose thrust equals the drag
    (clipped to [0, 1]). It ignores the thrust's share of the lift and the lateral
    trim, so it is only the starting point of `trim_level_flight`.
    """
    params, propulsion = aircraft.params, aircraft.propulsion
    rho = aircraft.density(altitude)
    qbar_s = 0.5 * rho * speed**2 * params.wing_area
    lift_needed = params.mass * GRAVITY / qbar_s
    matrix = np.array(
        [
            [params.c_lift_alpha, params.c_lift_elevator],
            [params.c_m_alpha, params.c_m_elevator],
        ]
    )
    alpha, elevator = np.linalg.solve(
        matrix, [lift_needed - params.c_lift_0, -params.c_m_0]
    )
    drag = qbar_s * (
        drag_coefficient(params, alpha) + params.c_drag_elevator * elevator
    )

    def excess_thrust(throttle: float) -> float:
        axial_speed = speed * np.cos(alpha)
        return (
            propeller_thrust_torque(propulsion, axial_speed, throttle, rho).thrust
            - drag
        )

    if excess_thrust(1.0) <= 0.0:
        throttle = 1.0
    elif excess_thrust(0.0) >= 0.0:
        throttle = 0.0
    else:
        throttle = brentq(excess_thrust, 0.0, 1.0)
    return float(alpha), float(elevator), float(throttle)


def trim_level_flight(
    aircraft: FixedWing,
    speed: float,
    altitude: float = 100.0,
    course: float = 0.0,
    tolerance: float = 1e-9,
) -> Trim:
    """Trim the aircraft for straight, wings-level flight at a given airspeed.

    Solves for alpha, beta, elevator, aileron, rudder and throttle so that the
    linear and angular accelerations are zero. Uses the aircraft's own force model,
    density and wind (the trim holds exactly in a uniform, steady wind). Leaves the
    aircraft commanded with the trimmed controls.

    Args:
        aircraft: a FixedWing with propulsion.
        speed: airspeed [m/s].
        altitude: [m].
        course: direction of the air velocity, clockwise from North [rad].
        tolerance: largest residual acceleration accepted [m/s^2, rad/s^2].

    Raises:
        ValueError: no propulsion, or no trim (for example a speed that needs more
            than full throttle).
    """
    if aircraft.propulsion is None:
        raise ValueError("level flight needs propulsion")
    if speed <= 0.0:
        raise ValueError("speed must be > 0")

    def unpack(x: np.ndarray) -> tuple[np.ndarray, ControlSurfaces, float]:
        alpha, beta, elevator, aileron, rudder, throttle = (float(v) for v in x)
        state = level_flight_state(aircraft, speed, alpha, beta, altitude, course)
        return state, ControlSurfaces(elevator, aileron, rudder), throttle

    def residual(x: np.ndarray) -> np.ndarray:
        state, aircraft.surfaces, aircraft.throttle = unpack(x)
        derivative = aircraft.derivative(0.0, state)
        return np.concatenate([derivative[3:6], derivative[10:13]])

    alpha, elevator, throttle = longitudinal_estimate(aircraft, speed, altitude)
    guess = np.array([alpha, 0.0, elevator, 0.0, 0.0, throttle])
    solution = root(residual, guess, method="hybr", options={"xtol": 1e-12})
    error = float(np.max(np.abs(residual(solution.x))))
    throttle = solution.x[5]
    if not solution.success or error > tolerance or not 0.0 <= throttle <= 1.0:
        raise ValueError(
            f"no level-flight trim at {speed} m/s and {altitude} m "
            f"(residual {error:.2e}, throttle {throttle:.2f})"
        )
    state, surfaces, throttle = unpack(solution.x)
    aircraft.surfaces, aircraft.throttle = surfaces, throttle
    return Trim(state, surfaces, throttle, float(solution.x[0]), float(solution.x[1]))
