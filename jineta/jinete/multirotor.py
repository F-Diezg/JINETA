"""Multirotor: rotor layout and constants, the motor mixer and the force model.

Each rotor pushes along -z_body (up, for a level vehicle) with thrust T_i and twists
the body with a reaction torque Q_i = moment_constant * T_i about z_body. The mixer
is the linear map between the rotor thrusts and their total effect on the vehicle,
the "wrench" [T_total, Mx, My, Mz] (total thrust along -z_body, moments in body
axes):

    wrench = A @ thrusts        A (4 x n rotors) is the allocation matrix

    row 0:  1                       every rotor adds to the total thrust
    row 1: -y_i                     roll: a rotor on the left (y < 0) rolls right
    row 2:  x_i                     pitch: a rotor ahead of the CoM pitches nose up
    row 3:  spin_i * moment_constant   yaw: reaction torque of the rotor

Rows 1 and 2 are the moment r_i x F_i of a force F_i = [0, 0, -T_i]. The height z_i
of the rotor does not appear: a force along z_body has no lever arm along z_body.

Rotor speed and air density do not appear here either: the mixer works with thrusts,
and the conversion thrust <-> rotor speed belongs to the force model.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.environment import DensityModel, WindModel, no_wind, sea_level_density
from jineta.core.inertia import inertia_tensor
from jineta.core.rigid_body import ForceMomentModel, RigidBody
from jineta.core.vector import cross
from jineta.jinete.aero import body_airspeed


@dataclass(frozen=True)
class Rotor:
    """One rotor: where it is and which way it turns.

    position: [x, y, z] in body axes (FRD) relative to the centre of mass [m].
    spin: +1 if it turns counter-clockwise seen from above (PX4/Gazebo "ccw"),
        -1 if clockwise. The body reacts the opposite way, so a +1 rotor yaws
        the body to the right (positive r).
    """

    position: np.ndarray
    spin: int

    def __post_init__(self) -> None:
        position = np.array(self.position, dtype=float)
        if position.shape != (3,):
            raise ValueError("rotor position must have 3 components [x, y, z]")
        if self.spin not in (1, -1):
            raise ValueError("rotor spin must be +1 (ccw) or -1 (cw)")
        position.setflags(write=False)
        # Frozen dataclass: fields can only be set through object.__setattr__.
        object.__setattr__(self, "position", position)


def allocation_matrix(rotors: Sequence[Rotor], moment_constant: float) -> np.ndarray:
    """Allocation matrix A (4 x n): wrench [T, Mx, My, Mz] = A @ rotor thrusts."""
    columns = [
        [1.0, -rotor.position[1], rotor.position[0], rotor.spin * moment_constant]
        for rotor in rotors
    ]
    return np.array(columns).T


@dataclass(frozen=True)
class MultirotorParams:
    """Physical parameters of a multirotor: only data, validated once.

    thrust_coefficient: k_T in T = k_T * Omega^2 [N/(rad/s)^2] (PX4 motorConstant).
    moment_constant: reaction torque / thrust, Q = moment_constant * T [m]
        (PX4 momentConstant).
    max_rotor_speed: Omega_max [rad/s].
    rotor_drag_coefficient: c_rd in the rotor drag -c_rd * Omega * v_perp
        [N/((rad/s)(m/s))] (Gazebo rotorDragCoefficient).
    body_drag_area: [CdA_x, CdA_y, CdA_z] of the airframe in body axes [m^2].
    reference_density: air density at which the rotor constants were measured
        [kg/m^3]; thrust, reaction torque and rotor drag scale with rho / this.
    motor_time_constant_up, motor_time_constant_down: the first-order lag of a
        rotor's speed behind its command, speeding up and slowing down [s] (0:
        not known; a vehicle with `motor_lag` needs both > 0).
    """

    mass: float  # [kg]
    inertia: np.ndarray  # 3x3 about the centre of mass, body axes [kg·m^2]
    rotors: tuple[Rotor, ...]
    thrust_coefficient: float
    moment_constant: float
    max_rotor_speed: float
    rotor_drag_coefficient: float
    body_drag_area: np.ndarray
    reference_density: float = 1.225  # ISA sea level
    motor_time_constant_up: float = 0.0
    motor_time_constant_down: float = 0.0
    allocation: np.ndarray = field(init=False, repr=False)
    allocation_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        positive = (
            "mass",
            "thrust_coefficient",
            "moment_constant",
            "max_rotor_speed",
            "reference_density",
        )
        for name in positive:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.rotor_drag_coefficient < 0.0:
            raise ValueError("rotor_drag_coefficient must be >= 0")
        if not (
            self.motor_time_constant_up >= 0.0 and self.motor_time_constant_down >= 0.0
        ):
            raise ValueError("motor time constants must be >= 0")
        body_drag_area = np.array(self.body_drag_area, dtype=float)
        if body_drag_area.shape != (3,) or np.any(body_drag_area < 0.0):
            raise ValueError(
                "body_drag_area must be 3 values >= 0 [CdA_x, CdA_y, CdA_z]"
            )
        body_drag_area.setflags(write=False)
        rotors = tuple(self.rotors)
        allocation = allocation_matrix(rotors, self.moment_constant)
        if np.linalg.matrix_rank(allocation) < 4:
            raise ValueError(
                "rotor layout cannot control thrust, roll, pitch and yaw independently"
            )
        # Pseudo-inverse: the exact inverse for 4 rotors, the minimum-effort
        # solution for more (hexacopter, octocopter).
        allocation_inv = np.linalg.pinv(allocation)
        allocation.setflags(write=False)
        allocation_inv.setflags(write=False)
        object.__setattr__(self, "inertia", inertia_tensor(self.inertia))
        object.__setattr__(self, "rotors", rotors)
        object.__setattr__(self, "body_drag_area", body_drag_area)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "allocation_inv", allocation_inv)

    @property
    def max_rotor_thrust(self) -> float:
        """Thrust of one rotor at full speed [N]."""
        return self.thrust_coefficient * self.max_rotor_speed**2

    def wrench(self, thrusts: npt.ArrayLike) -> np.ndarray:
        """Total effect [T, Mx, My, Mz] of the given rotor thrusts [N]."""
        return self.allocation @ np.asarray(thrusts, dtype=float)

    def rotor_thrusts(self, wrench: npt.ArrayLike) -> np.ndarray:
        """Rotor thrusts [N] that produce the requested wrench [T, Mx, My, Mz].

        Each thrust is clipped to [0, max_rotor_thrust]: a rotor cannot pull down
        or spin faster than its maximum. If clipping happens, the wrench actually
        produced differs from the one requested.
        """
        thrusts = self.allocation_inv @ np.asarray(wrench, dtype=float)
        return np.clip(thrusts, 0.0, self.max_rotor_thrust)

    def rotor_speeds_for(self, thrusts: npt.ArrayLike) -> np.ndarray:
        """Rotor speeds [rad/s] that give these thrusts [N] at the reference density.

        Inverse of T = k_T * Omega^2. Negative thrusts are treated as zero.
        """
        thrusts = np.clip(np.asarray(thrusts, dtype=float), 0.0, None)
        return np.sqrt(thrusts / self.thrust_coefficient)


def motor_derivative(
    params: MultirotorParams, command: np.ndarray, speeds: np.ndarray
) -> np.ndarray:
    """d(rotor speeds)/dt [rad/s^2] of rotors lagging behind their command: each
    speed approaches its command (clipped to [0, max_rotor_speed]) as a first
    order, (command - speed) / tau, with tau the time constant up while the
    command is above the speed and down otherwise (the motor model of PX4's
    Gazebo simulator). Works on one vehicle (rotors,) or many (n, rotors)."""
    command = np.clip(command, 0.0, params.max_rotor_speed)
    tau = np.where(
        command > speeds,
        params.motor_time_constant_up,
        params.motor_time_constant_down,
    )
    return (command - speeds) / tau


@dataclass
class Multirotor:
    """A multirotor in the environment: provides the force/moment model of a RigidBody.

    Input: `rotor_speed_command` [rad/s], one value per rotor, held between updates
    and clipped to [0, max_rotor_speed].

    Motors. Without `motor_lag` (the default) the rotors spin at their command at
    once, and the state is the rigid body's 13 elements. With it, each rotor's
    speed follows its command with a first-order lag (the profile's
    `motor_time_constant_up` / `_down`; 12.5 / 25 ms for the Iris): the rotor
    speeds are part of the state, after the body's 13 elements
    (`state_size` = 13 + rotors), integrated with the rest, and the forces use
    them instead of the command. `initial_state` adds them to a body state.

    Forces and moments in body axes (gravity is added by the RigidBody):
    - thrust and reaction torque of each rotor, through the mixer, scaled with
      rho / reference_density (T = C_T * rho * n^2 * D^4 from propeller theory);
    - rotor drag at each rotor hub, -(rho / rho_ref) * c_rd * Omega_i * v_perp_i,
      where v_perp_i is the airspeed of the hub in the rotor plane, including the
      rotation of the body (omega x r_i). Same form as the Gazebo motor model; the
      density factor makes it grow like sqrt(rho) at constant thrust, as in the
      PX4 EKF2 drag model;
    - body drag, -0.5 * rho * CdA_i * u_i * |u| on each body axis (PX4 EKF2 drag
      model; with equal CdA on all axes it is the usual isotropic quadratic drag);
    - contact, if a `contact` model is given (e.g. `GroundContact` with the
      vehicle's landing gear): its force and moment are added as they come.
    """

    params: MultirotorParams
    density: DensityModel = sea_level_density
    wind: WindModel = no_wind
    contact: ForceMomentModel | None = None
    motor_lag: bool = False
    rotor_speed_command: np.ndarray = field(init=False)
    body: RigidBody = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.motor_lag and not (
            self.params.motor_time_constant_up > 0.0
            and self.params.motor_time_constant_down > 0.0
        ):
            raise ValueError("motor_lag needs motor time constants > 0")
        self.rotor_speed_command = np.zeros(len(self.params.rotors))
        self.body = RigidBody(
            mass=self.params.mass,
            inertia=self.params.inertia,
            force_moment=self.force_moment,
        )

    @property
    def state_size(self) -> int:
        """13 (the rigid body), + one rotor speed per rotor with `motor_lag`."""
        return 13 + (len(self.params.rotors) if self.motor_lag else 0)

    def initial_state(
        self, body_state: np.ndarray, rotor_speeds: npt.ArrayLike = 0.0
    ) -> np.ndarray:
        """The vehicle's state from a rigid-body state (13,): with `motor_lag` the
        rotor speeds (one for all or one per rotor; stopped by default) are
        added after it; without, it is the body state."""
        body_state = np.asarray(body_state, dtype=float)[:13]
        if not self.motor_lag:
            return body_state.copy()
        speeds = np.broadcast_to(
            np.asarray(rotor_speeds, dtype=float), (len(self.params.rotors),)
        )
        return np.concatenate([body_state, speeds])

    def rotor_speeds(self, state: np.ndarray) -> np.ndarray:
        """The speeds the rotors spin at [rad/s]: the state's with `motor_lag`,
        the command's without (both clipped to [0, max_rotor_speed])."""
        speeds = state[13:] if self.motor_lag else self.rotor_speed_command
        return np.clip(speeds, 0.0, self.params.max_rotor_speed)

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return d(state)/dt: the rigid body's 13 elements, then the rotor
        speeds' with `motor_lag`."""
        body = self.body.derivative(t, state)
        if not self.motor_lag:
            return body
        motors = motor_derivative(self.params, self.rotor_speed_command, state[13:])
        return np.concatenate([body, motors])

    def force_moment(
        self, t: float, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Force and moment in body axes [N], [N·m]: rotors, drag and contact (no
        gravity: the RigidBody adds it)."""
        params = self.params
        speeds = self.rotor_speeds(state)
        rho = self.density(-state[2])
        density_ratio = rho / params.reference_density

        # Rotor thrust and reaction torque, through the mixer.
        thrusts = density_ratio * params.thrust_coefficient * speeds**2
        total_thrust, mx, my, mz = params.wrench(thrusts)
        force = np.array([0.0, 0.0, -total_thrust])
        moment = np.array([mx, my, mz])

        # Rotor drag at each hub, opposing the in-plane airspeed of the hub.
        airspeed = body_airspeed(t, state, self.wind)
        omega = state[10:13]
        for rotor, speed in zip(params.rotors, speeds, strict=True):
            hub_airspeed = airspeed + cross(omega, rotor.position)
            in_plane = np.array([hub_airspeed[0], hub_airspeed[1], 0.0])
            drag = -density_ratio * params.rotor_drag_coefficient * speed * in_plane
            force = force + drag
            moment = moment + cross(rotor.position, drag)

        # Body drag on the airframe.
        speed_through_air = np.linalg.norm(airspeed)
        force = force - 0.5 * rho * params.body_drag_area * airspeed * speed_through_air

        if self.contact is not None:
            contact_force, contact_moment = self.contact(t, state)
            force = force + contact_force
            moment = moment + contact_moment

        return force, moment
