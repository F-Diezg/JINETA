"""Multirotors for many drones at once: `jinete.multirotor.Multirotor` on a stack.

N drones of the same profile (`MultirotorParams`), each with its own rotor speed
commands, one row per drone in `rotor_speed_command` (N, n rotors). The forces are
those of the reference vehicle, term by term and in the same order:

- thrust and reaction torque of each rotor through the mixer, scaled with
  rho / reference_density;
- rotor drag at each hub, -(rho / rho_ref) c_rd Omega_i v_perp_i, with the hub's
  in-plane airspeed including omega x r_i;
- body drag, -1/2 rho CdA_i u_i |u| on each body axis;
- contact, if a batch `contact` model is given (e.g. `BatchGroundContact`).

With `motor_lag` the rotor speeds are part of the states, after the bodies' 13
columns (states (N, 13 + rotors)), and follow their commands with the profile's
first-order lag, as in the reference. With a `ground_effect` each rotor's
thrust is multiplied by its factor (`jinete.proximity`), as in the reference.

Drones of one profile, each with its own numbers (phase 12): `set_parameters`
gives chosen drones their own mass, inertia, thrust and moment constants,
maximum rotor speed, drag and motor time constants (`PER_DRONE`); from then on
those are arrays with one value per drone, and drone i flies as the reference
`Multirotor(drone_params(i))`. The rotors' places, the landing gear and the
propeller radius stay the profile's. Whoever controls the drones keeps the
profile's numbers (`params`): a controller does not know the drone it flies
exactly.

The sums over the rotors are written as a loop over the rotors (four for the Iris)
in which each operation acts on all the drones: the rotors are added one after the
other as in the reference, and no operation mixes the drones.

Environment models are the reference ones, called with arrays: `density` gets the
altitudes (N,) and may return one density for all or one per drone (indoors it is
constant, `sea_level_density`); `wind` gets the time and the positions (N, 3) and
may return one wind (3,) for all or one per drone (N, 3) (`no_wind`,
`ConstantWind`). Models written for one number at a time (the ISA functions, loose
end 2) do not work here.
"""

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.environment import DensityModel, WindModel, no_wind, sea_level_density
from jineta.core.vector import cross, matvec
from jineta.escuadron import quaternion
from jineta.escuadron.rigid_body import BatchForceMomentModel, BatchRigidBody
from jineta.jinete.multirotor import MultirotorParams, lag_derivative
from jineta.jinete.proximity import GroundEffect, ground_effect_factor

# The numbers each drone of a batch can have of its own (`set_parameters`).
PER_DRONE = (
    "mass",
    "inertia",
    "thrust_coefficient",
    "moment_constant",
    "max_rotor_speed",
    "rotor_drag_coefficient",
    "body_drag_area",
    "motor_time_constant_up",
    "motor_time_constant_down",
)


@dataclass
class BatchMultirotor:
    """N multirotors of one profile: the force model of a `BatchRigidBody`.

    Input: `rotor_speed_command` [rad/s], shape (n, number of rotors), held between
    updates and clipped to [0, max_rotor_speed]; it starts at zero (motors off).
    """

    params: MultirotorParams
    n: int  # number of drones
    density: DensityModel = sea_level_density
    wind: WindModel = no_wind
    contact: BatchForceMomentModel | None = None
    motor_lag: bool = False
    ground_effect: GroundEffect | None = None
    rotor_speed_command: np.ndarray = field(init=False)
    body: BatchRigidBody = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        if self.ground_effect is not None and not self.params.propeller_radius > 0.0:
            raise ValueError("a ground effect needs the propeller radius")
        if self.motor_lag and not (
            self.params.motor_time_constant_up > 0.0
            and self.params.motor_time_constant_down > 0.0
        ):
            raise ValueError("motor_lag needs motor time constants > 0")
        self.rotor_speed_command = np.zeros((self.n, len(self.params.rotors)))
        self.body = BatchRigidBody(
            mass=self.params.mass,
            inertia=self.params.inertia,
            force_moment=self.force_moment,
        )
        # The drones' own numbers: the profile's (one for all) until
        # set_parameters gives them one each, (n, 1), (n,) or (n, 3).
        p = self.params
        self.thrust_coefficient: float | np.ndarray = p.thrust_coefficient
        self.moment_constant: float | np.ndarray = p.moment_constant
        self.max_rotor_speed: float | np.ndarray = p.max_rotor_speed
        self.rotor_drag_coefficient: float | np.ndarray = p.rotor_drag_coefficient
        self.body_drag_area: np.ndarray = p.body_drag_area
        self.motor_time_constant_up: float | np.ndarray = p.motor_time_constant_up
        self.motor_time_constant_down: float | np.ndarray = p.motor_time_constant_down
        self.yaw_coefficients: np.ndarray | None = None  # (n, rotors) per drone

    @property
    def per_drone(self) -> bool:
        """Whether the drones have their own numbers (`set_parameters`)."""
        return self.yaw_coefficients is not None

    def set_parameters(self, drones: npt.ArrayLike, **values: npt.ArrayLike) -> None:
        """Give these drones (a mask (n,) or indices) their own numbers: any of
        `PER_DRONE` as keywords, one value per chosen drone (inertia (k, 3, 3),
        body_drag_area (k, 3), the others (k,))."""
        unknown = set(values) - set(PER_DRONE)
        if unknown:
            raise ValueError(f"not a per-drone parameter: {sorted(unknown)}")
        n, rotors = self.n, len(self.params.rotors)
        if not self.per_drone:
            column = (n, 1)
            self.thrust_coefficient = np.full(column, self.params.thrust_coefficient)
            self.moment_constant = np.full(n, self.params.moment_constant)
            self.max_rotor_speed = np.full(column, self.params.max_rotor_speed)
            self.rotor_drag_coefficient = np.full(n, self.params.rotor_drag_coefficient)
            self.body_drag_area = np.tile(self.params.body_drag_area, (n, 1))
            self.motor_time_constant_up = np.full(
                column, self.params.motor_time_constant_up
            )
            self.motor_time_constant_down = np.full(
                column, self.params.motor_time_constant_down
            )
            self.yaw_coefficients = np.zeros((n, rotors))
            self.body.set_bodies(n, np.arange(0), [], np.zeros((0, 3, 3)))
        rows = np.arange(n)[np.asarray(drones)]
        k = len(rows)
        if "mass" in values or "inertia" in values:
            mass = values.get("mass", self.body.mass[rows])
            inertia = values.get("inertia", self.body.inertia[rows])
            self.body.set_bodies(n, rows, mass, inertia)
        for name in ("thrust_coefficient", "max_rotor_speed"):
            if name in values:
                value = np.broadcast_to(np.asarray(values[name], float), (k,))
                if np.any(~(value > 0.0)):
                    raise ValueError(f"{name} must be > 0")
                getattr(self, name)[rows, 0] = value
        for name in ("motor_time_constant_up", "motor_time_constant_down"):
            if name in values:
                value = np.broadcast_to(np.asarray(values[name], float), (k,))
                if np.any(~(value >= 0.0)):
                    raise ValueError(f"{name} must be >= 0")
                getattr(self, name)[rows, 0] = value
        if "moment_constant" in values:
            value = np.broadcast_to(np.asarray(values["moment_constant"], float), (k,))
            if np.any(~(value > 0.0)):
                raise ValueError("moment_constant must be > 0")
            self.moment_constant[rows] = value
        if "rotor_drag_coefficient" in values:
            value = np.broadcast_to(
                np.asarray(values["rotor_drag_coefficient"], float), (k,)
            )
            if np.any(~(value >= 0.0)):
                raise ValueError("rotor_drag_coefficient must be >= 0")
            self.rotor_drag_coefficient[rows] = value
        if "body_drag_area" in values:
            value = np.broadcast_to(np.asarray(values["body_drag_area"], float), (k, 3))
            if np.any(~(value >= 0.0)):
                raise ValueError("body_drag_area must be >= 0")
            self.body_drag_area[rows] = value
        if self.motor_lag and not (
            np.all(self.motor_time_constant_up > 0.0)
            and np.all(self.motor_time_constant_down > 0.0)
        ):
            raise ValueError("motor_lag needs motor time constants > 0")
        for i, rotor in enumerate(self.params.rotors):
            self.yaw_coefficients[:, i] = rotor.spin * self.moment_constant

    def drone_params(self, i: int) -> MultirotorParams:
        """Drone i's numbers as a profile: the reference `Multirotor` with them
        flies as drone i does."""
        if not self.per_drone:
            return self.params
        return dataclasses.replace(
            self.params,
            mass=float(self.body.mass[i]),
            inertia=self.body.inertia[i].copy(),
            thrust_coefficient=float(self.thrust_coefficient[i, 0]),
            moment_constant=float(self.moment_constant[i]),
            max_rotor_speed=float(self.max_rotor_speed[i, 0]),
            rotor_drag_coefficient=float(self.rotor_drag_coefficient[i]),
            body_drag_area=self.body_drag_area[i].copy(),
            motor_time_constant_up=float(self.motor_time_constant_up[i, 0]),
            motor_time_constant_down=float(self.motor_time_constant_down[i, 0]),
        )

    @property
    def state_size(self) -> int:
        """13 (the rigid body), + one rotor speed per rotor with `motor_lag`."""
        return 13 + (len(self.params.rotors) if self.motor_lag else 0)

    def initial_state(
        self, body_state: np.ndarray, rotor_speeds: float = 0.0
    ) -> np.ndarray:
        """One drone's state (state_size,) from a rigid-body state (13,): the
        rotor speeds added after it with `motor_lag` (stopped by default)."""
        body_state = np.asarray(body_state, dtype=float)[:13]
        if not self.motor_lag:
            return body_state.copy()
        speeds = np.full(len(self.params.rotors), float(rotor_speeds))
        return np.concatenate([body_state, speeds])

    def rotor_speeds(self, states: np.ndarray) -> np.ndarray:
        """The speeds the rotors spin at (n, rotors) [rad/s]: the states' with
        `motor_lag`, the commands' without (clipped to [0, max_rotor_speed])."""
        speeds = states[:, 13:] if self.motor_lag else self.rotor_speed_command
        return np.clip(speeds, 0.0, self.max_rotor_speed)

    def derivative(self, t: float | np.ndarray, states: np.ndarray) -> np.ndarray:
        """d(states)/dt for the states (n, state_size)."""
        body = self.body.derivative(t, states)
        if not self.motor_lag:
            return body
        motors = lag_derivative(
            self.rotor_speed_command,
            states[:, 13:],
            self.max_rotor_speed,
            self.motor_time_constant_up,
            self.motor_time_constant_down,
        )
        return np.concatenate([body, motors], axis=1)

    def force_moment(
        self, t: float | np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forces and moments in body axes, (n, 3) each [N], [N·m]: rotors, drag
        and contact (no gravity: the BatchRigidBody adds it)."""
        params = self.params
        n = len(states)
        speeds = self.rotor_speeds(states)
        rho = np.broadcast_to(np.asarray(self.density(-states[:, 2]), float), (n,))
        density_ratio = rho / params.reference_density

        # Rotor thrust and reaction torque, through the mixer: the wrench
        # [T, Mx, My, Mz] = A @ thrusts, adding the rotors one after the other.
        thrusts = density_ratio[:, None] * self.thrust_coefficient * speeds**2
        rotation = quaternion.to_matrix(states[:, 6:10])
        air_ned = states[:, 3:6] - self.wind(t, states[:, 0:3])
        airspeed = matvec(rotation.swapaxes(-1, -2), air_ned)
        omega = states[:, 10:13]
        allocation = params.allocation
        if self.ground_effect is None:
            lifts = thrusts
        else:
            lifts = thrusts * self._ground_factors(
                states, rotation, thrusts, airspeed, rho
            )
        wrench = lifts[:, 0:1] * allocation[:, 0]
        for i in range(1, len(params.rotors)):
            wrench = wrench + lifts[:, i : i + 1] * allocation[:, i]
        if self.ground_effect is not None or self.per_drone:
            # The reaction torques: free of the floor, with each drone's moment
            # constant.
            yaw = allocation[3] if not self.per_drone else self.yaw_coefficients.T
            torque = thrusts[:, 0] * yaw[0]
            for i in range(1, len(params.rotors)):
                torque = torque + thrusts[:, i] * yaw[i]
            wrench[:, 3] = torque
        force = np.zeros((n, 3))
        force[:, 2] = -wrench[:, 0]
        moment = wrench[:, 1:4].copy()

        # Rotor drag at each hub, opposing the in-plane airspeed of the hub.
        drag_factor = -density_ratio * self.rotor_drag_coefficient
        for i, rotor in enumerate(params.rotors):
            hub_airspeed = airspeed + cross(omega, rotor.position)
            in_plane = hub_airspeed.copy()
            in_plane[:, 2] = 0.0
            drag = (drag_factor * speeds[:, i])[:, None] * in_plane
            force = force + drag
            moment = moment + cross(rotor.position, drag)

        # Body drag on the airframe.
        speed_through_air = np.sqrt(
            airspeed[:, 0] * airspeed[:, 0]
            + airspeed[:, 1] * airspeed[:, 1]
            + airspeed[:, 2] * airspeed[:, 2]
        )
        force = (
            force
            - 0.5
            * rho[:, None]
            * self.body_drag_area
            * airspeed
            * speed_through_air[:, None]
        )

        if self.contact is not None:
            contact_force, contact_moment = self.contact(t, states)
            force = force + contact_force
            moment = moment + contact_moment

        return force, moment

    def _ground_factors(
        self,
        states: np.ndarray,
        rotation: np.ndarray,
        thrusts: np.ndarray,
        airspeed: np.ndarray,
        rho: np.ndarray,
    ) -> np.ndarray:
        """Each rotor's ground-effect factor (n, rotors), as the reference's."""
        omega = states[:, 10:13]
        heights = np.zeros_like(thrusts)
        speeds = np.zeros_like(thrusts)
        for i, rotor in enumerate(self.params.rotors):
            r = rotor.position
            hub_z = states[:, 2] + (
                rotation[:, 2, 0] * r[0]
                + rotation[:, 2, 1] * r[1]
                + rotation[:, 2, 2] * r[2]
            )
            heights[:, i] = self.ground_effect.floor_z - hub_z
            hub_airspeed = airspeed + cross(omega, r)
            speeds[:, i] = np.hypot(hub_airspeed[:, 0], hub_airspeed[:, 1])
        return ground_effect_factor(
            self.params.propeller_radius,
            heights,
            thrusts,
            speeds,
            rho[:, None],
            self.ground_effect.min_height,
        )
