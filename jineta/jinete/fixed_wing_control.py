"""Fixed-wing autopilot by successive loop closure (Beard & McLain, ch. 6).

The aircraft is flown by three commands, each held by a chain of loops, the inner
loop much faster than the outer one so that, seen from outside, it simply does what
it is told:

    course   -> roll angle  -> aileron        (lateral)
    sideslip -> rudder                        (turn coordination, beta -> 0)
    altitude -> pitch angle -> elevator       (longitudinal)
    airspeed -> throttle

Each inner loop is designed on a second-order model of its axis, built from the
stability derivatives at the trim (the coefficients a_* of Beard & McLain ch. 5):

    roll:     phi''   = -a_phi1 phi' + a_phi2 aileron
    pitch:    theta'' = -a_theta1 theta' - a_theta2 theta + a_theta3 elevator
    airspeed: V'      = -a_V1 V + a_V2 throttle
    sideslip: beta'   = -a_beta1 beta + a_beta2 rudder

The roll and pitch gains follow the rule of the source: the proportional gain is
the surface limit divided by the error that should just saturate it (so the loop
uses the whole surface, and its stiffness does not depend on the speed), and the
derivative gain sets the damping. The outer loops are made `separation` times slower
than their inner loop and get an integral term to remove steady errors (wind,
the weight in a turn, model errors).

Trim feedforward: the elevator, throttle and pitch of the trim are added to the
loop outputs, so the integrators only correct what the trim does not explain.
Without it the slow altitude integrator had to build the trim elevator alone and
the response oscillated for tens of seconds.

Anti-windup: an integrator stops integrating while the output it drives is
saturated (conditional integration).
"""

from dataclasses import dataclass, field

import numpy as np

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.jinete.aero import air_data, body_airspeed
from jineta.jinete.fixed_wing import ControlSurfaces, FixedWing, drag_coefficient
from jineta.jinete.propulsion import propeller_thrust_torque
from jineta.jinete.trim import Trim


@dataclass(frozen=True)
class AutopilotDesign:
    """Design choices of the autopilot: limits, bandwidths and damping ratios.

    Profile data (like the multirotor gains): the gains themselves are computed
    from these and the aircraft at its trim by `design_gains`.
    """

    surface_limit: float  # largest deflection of every surface [rad]
    max_bank: float  # [rad]
    max_pitch: float  # [rad]
    roll_error_at_saturation: float  # roll error that gives full aileron [rad]
    roll_damping: float
    course_separation: float  # roll loop natural frequency / course loop one
    course_damping: float
    pitch_error_at_saturation: float  # [rad]
    pitch_damping: float
    altitude_separation: float  # pitch loop natural frequency / altitude loop one
    altitude_damping: float
    airspeed_frequency: float  # [rad/s]
    airspeed_damping: float
    sideslip_frequency: float  # [rad/s]
    sideslip_damping: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")


@dataclass(frozen=True)
class AutopilotGains:
    """Gains of every loop, and the natural frequencies they give."""

    roll_kp: float  # aileron per roll error
    roll_kd: float  # aileron per roll rate [s]
    course_kp: float  # roll command per course error
    course_ki: float  # [1/s]
    pitch_kp: float  # elevator per pitch error
    pitch_kd: float  # elevator per pitch rate [s]
    altitude_kp: float  # pitch command per altitude error [rad/m]
    altitude_ki: float  # [rad/(m s)]
    airspeed_kp: float  # throttle per airspeed error [s/m]
    airspeed_ki: float  # [1/m]
    sideslip_kp: float  # rudder per sideslip
    sideslip_ki: float  # [1/s]
    roll_frequency: float  # natural frequencies of the closed loops [rad/s]
    course_frequency: float
    pitch_frequency: float
    altitude_frequency: float
    pitch_dc_gain: float  # steady pitch / pitch command of the pitch loop


def design_gains(
    aircraft: FixedWing, trim: Trim, design: AutopilotDesign
) -> AutopilotGains:
    """Gains from the second-order models of each axis at the trim.

    Args:
        aircraft: the FixedWing (with propulsion) the autopilot will fly.
        trim: the operating point (its airspeed and altitude set the models).
        design: limits, bandwidths and damping ratios.
    """
    params, prop = aircraft.params, aircraft.propulsion
    if prop is None:
        raise ValueError("the autopilot needs an aircraft with propulsion")
    speed = float(np.linalg.norm(body_airspeed(0.0, trim.state, aircraft.wind)))
    rho = aircraft.density(-trim.state[2])
    s, b, c, m = params.wing_area, params.span, params.chord, params.mass
    qbar_s = 0.5 * rho * speed**2 * s
    jx, jy, jz = np.diag(params.inertia)
    jxz = -params.inertia[0, 2]  # the tensor stores -J_xz off the diagonal

    # Roll: the roll acceleration per rate and per aileron, with the J_xz coupling
    # of the yawing moment (Gamma_3, Gamma_4 of the source).
    gamma = jx * jz - jxz**2
    c_p_p = (jz * params.c_ell_p + jxz * params.c_n_p) / gamma
    c_p_aileron = (jz * params.c_ell_aileron + jxz * params.c_n_aileron) / gamma
    a_phi1 = -qbar_s * b * c_p_p * b / (2 * speed)
    a_phi2 = qbar_s * b * c_p_aileron
    roll_kp = design.surface_limit / design.roll_error_at_saturation * np.sign(a_phi2)
    roll_frequency = np.sqrt(a_phi2 * roll_kp)
    roll_kd = (2 * design.roll_damping * roll_frequency - a_phi1) / a_phi2

    # Course: the bank turns the velocity, chi' = g / V tan(phi) ~ (g / V) phi.
    course_frequency = roll_frequency / design.course_separation
    course_kp = 2 * design.course_damping * course_frequency * speed / GRAVITY
    course_ki = course_frequency**2 * speed / GRAVITY

    # Pitch.
    a_theta1 = -qbar_s * c * params.c_m_q * c / (2 * speed) / jy
    a_theta2 = -qbar_s * c * params.c_m_alpha / jy
    a_theta3 = qbar_s * c * params.c_m_elevator / jy
    pitch_kp = (
        design.surface_limit / design.pitch_error_at_saturation * np.sign(a_theta3)
    )
    pitch_frequency = np.sqrt(a_theta2 + pitch_kp * a_theta3)
    pitch_kd = (2 * design.pitch_damping * pitch_frequency - a_theta1) / a_theta3
    pitch_dc_gain = pitch_kp * a_theta3 / (a_theta2 + pitch_kp * a_theta3)

    # Altitude: the pitch tilts the velocity, h' = V sin(theta) ~ V theta.
    altitude_frequency = pitch_frequency / design.altitude_separation
    altitude_kp = (
        2 * design.altitude_damping * altitude_frequency / (pitch_dc_gain * speed)
    )
    altitude_ki = altitude_frequency**2 / (pitch_dc_gain * speed)

    # Airspeed: drag and the propeller's loss of thrust with speed slow it down;
    # the throttle speeds it up. Slopes of the thrust by central differences.
    axial = speed * np.cos(trim.alpha)
    h = 1e-4

    def thrust(v: float, throttle: float) -> float:
        return propeller_thrust_torque(prop, v, throttle, rho).thrust

    thrust_per_speed = (
        thrust(axial + h, trim.throttle) - thrust(axial - h, trim.throttle)
    ) / (2 * h)
    thrust_per_throttle = (
        thrust(axial, trim.throttle + h) - thrust(axial, trim.throttle - h)
    ) / (2 * h)
    c_drag = (
        drag_coefficient(params, trim.alpha)
        + params.c_drag_elevator * trim.surfaces.elevator
    )
    a_v1 = rho * speed * s * c_drag / m - thrust_per_speed / m
    a_v2 = thrust_per_throttle / m
    w_v, z_v = design.airspeed_frequency, design.airspeed_damping
    airspeed_kp = (2 * z_v * w_v - a_v1) / a_v2
    airspeed_ki = w_v**2 / a_v2

    # Sideslip: the side force of the rudder against the side force of beta.
    a_beta1 = -rho * speed * s * params.c_y_beta / (2 * m)
    a_beta2 = rho * speed * s * params.c_y_rudder / (2 * m)
    w_b, z_b = design.sideslip_frequency, design.sideslip_damping
    sideslip_kp = (2 * z_b * w_b - a_beta1) / a_beta2
    sideslip_ki = w_b**2 / a_beta2

    return AutopilotGains(
        roll_kp=float(roll_kp),
        roll_kd=float(roll_kd),
        course_kp=float(course_kp),
        course_ki=float(course_ki),
        pitch_kp=float(pitch_kp),
        pitch_kd=float(pitch_kd),
        altitude_kp=float(altitude_kp),
        altitude_ki=float(altitude_ki),
        airspeed_kp=float(airspeed_kp),
        airspeed_ki=float(airspeed_ki),
        sideslip_kp=float(sideslip_kp),
        sideslip_ki=float(sideslip_ki),
        roll_frequency=float(roll_frequency),
        course_frequency=float(course_frequency),
        pitch_frequency=float(pitch_frequency),
        altitude_frequency=float(altitude_frequency),
        pitch_dc_gain=float(pitch_dc_gain),
    )


def wrap_angle(angle: float) -> float:
    """The same angle in [-pi, pi): turning the short way."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class FixedWingAutopilot:
    """Course, altitude and airspeed hold: the whole set of loops in one pre_step.

    Set `course_command` (direction of the ground velocity, clockwise from North
    [rad]), `altitude_command` [m] and `airspeed_command` [m/s]; they start at the
    trim. Pass the autopilot itself as `pre_step` to `run`: each call writes the
    vehicle's control surfaces and throttle.
    """

    vehicle: FixedWing
    trim: Trim
    design: AutopilotDesign
    dt: float  # control period [s]: the simulation step
    course_command: float = field(init=False)
    altitude_command: float = field(init=False)
    airspeed_command: float = field(init=False)
    gains: AutopilotGains = field(init=False)
    _course_integral: float = field(init=False, repr=False)
    _altitude_integral: float = field(init=False, repr=False)
    _airspeed_integral: float = field(init=False, repr=False)
    _sideslip_integral: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.gains = design_gains(self.vehicle, self.trim, self.design)
        velocity = self.trim.state[3:6]
        self.course_command = float(np.arctan2(velocity[1], velocity[0]))
        self.altitude_command = float(-self.trim.state[2])
        self.airspeed_command = float(
            np.linalg.norm(body_airspeed(0.0, self.trim.state, self.vehicle.wind))
        )
        self.reset()

    def reset(self) -> None:
        """Clear the integrators (start of a new flight)."""
        self._course_integral = 0.0
        self._altitude_integral = 0.0
        self._airspeed_integral = 0.0
        self._sideslip_integral = 0.0

    def __call__(self, t: float, state: np.ndarray) -> None:
        """The pre_step hook: update the vehicle's surfaces and throttle."""
        g, d, dt = self.gains, self.design, self.dt
        roll, pitch, _ = quaternion.to_euler(state[6:10])
        p, q, _ = state[10:13]
        air = air_data(body_airspeed(t, state, self.vehicle.wind))
        course = np.arctan2(state[4], state[3])
        altitude = -state[2]

        # Course -> roll command -> aileron.
        error = wrap_angle(self.course_command - course)
        roll_command = g.course_kp * error + g.course_ki * (
            self._course_integral + error * dt
        )
        if abs(roll_command) < d.max_bank:
            self._course_integral += error * dt
        roll_command = float(np.clip(roll_command, -d.max_bank, d.max_bank))
        aileron = g.roll_kp * (roll_command - roll) - g.roll_kd * p

        # Sideslip -> rudder (drive beta to zero).
        error = -air.beta
        rudder = g.sideslip_kp * error + g.sideslip_ki * (
            self._sideslip_integral + error * dt
        )
        if abs(rudder) < d.surface_limit:
            self._sideslip_integral += error * dt

        # Altitude -> pitch command -> elevator.
        error = self.altitude_command - altitude
        pitch_command = (
            self.trim.alpha
            + g.altitude_kp * error
            + g.altitude_ki * (self._altitude_integral + error * dt)
        )
        if abs(pitch_command) < d.max_pitch:
            self._altitude_integral += error * dt
        pitch_command = float(np.clip(pitch_command, -d.max_pitch, d.max_pitch))
        elevator = (
            self.trim.surfaces.elevator
            + g.pitch_kp * (pitch_command - pitch)
            - g.pitch_kd * q
        )

        # Airspeed -> throttle.
        error = self.airspeed_command - air.speed
        throttle = (
            self.trim.throttle
            + g.airspeed_kp * error
            + g.airspeed_ki * (self._airspeed_integral + error * dt)
        )
        if 0.0 < throttle < 1.0:
            self._airspeed_integral += error * dt

        limit = d.surface_limit
        self.vehicle.surfaces = ControlSurfaces(
            elevator=float(np.clip(elevator, -limit, limit)),
            aileron=float(np.clip(aileron, -limit, limit)),
            rudder=float(np.clip(rudder, -limit, limit)),
        )
        self.vehicle.throttle = float(np.clip(throttle, 0.0, 1.0))


def orbit_course(
    position: np.ndarray,
    center: np.ndarray,
    radius: float,
    clockwise: bool = True,
    gain: float = 2.0,
) -> float:
    """Course command that flies onto a circle and around it (seen from above).

    Vector-field guidance (Beard & McLain, ch. 10): at angle phi around the centre
    and distance d from it, fly

        chi = phi + lambda (pi/2 + atan(gain (d - radius) / radius))

    with lambda = +1 clockwise, -1 anticlockwise. On the circle that is the tangent;
    far outside it tends to phi + lambda pi, straight at the centre; inside, away from
    it. `gain` sets how sharply the field bends onto the circle.

    Args:
        position: NED [m] (only North and East are used).
        center: NED [m] of the circle (only North and East are used).
        radius: [m].
        clockwise: direction of the orbit seen from above.
        gain: > 0.

    Returns:
        Course command [rad], clockwise from North, in [-pi, pi).
    """
    if radius <= 0.0 or gain <= 0.0:
        raise ValueError("radius and gain must be > 0")
    north = position[0] - center[0]
    east = position[1] - center[1]
    phi = np.arctan2(east, north)
    distance = np.hypot(north, east)
    direction = 1.0 if clockwise else -1.0
    course = phi + direction * (
        np.pi / 2 + np.arctan(gain * (distance - radius) / radius)
    )
    return wrap_angle(course)
