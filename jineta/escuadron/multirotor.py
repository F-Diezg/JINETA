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

from dataclasses import dataclass, field

import numpy as np

from jineta.core.environment import DensityModel, WindModel, no_wind, sea_level_density
from jineta.core.vector import cross, matvec
from jineta.escuadron import quaternion
from jineta.escuadron.rigid_body import BatchForceMomentModel, BatchRigidBody
from jineta.jinete.multirotor import MultirotorParams, motor_derivative
from jineta.jinete.proximity import GroundEffect, ground_effect_factor


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
        return np.clip(speeds, 0.0, self.params.max_rotor_speed)

    def derivative(self, t: float | np.ndarray, states: np.ndarray) -> np.ndarray:
        """d(states)/dt for the states (n, state_size)."""
        body = self.body.derivative(t, states)
        if not self.motor_lag:
            return body
        motors = motor_derivative(self.params, self.rotor_speed_command, states[:, 13:])
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
        thrusts = density_ratio[:, None] * params.thrust_coefficient * speeds**2
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
        if self.ground_effect is not None:
            # The reaction torques are those free of the floor.
            torque = thrusts[:, 0] * allocation[3, 0]
            for i in range(1, len(params.rotors)):
                torque = torque + thrusts[:, i] * allocation[3, i]
            wrench[:, 3] = torque
        force = np.zeros((n, 3))
        force[:, 2] = -wrench[:, 0]
        moment = wrench[:, 1:4].copy()

        # Rotor drag at each hub, opposing the in-plane airspeed of the hub.
        drag_factor = -density_ratio * params.rotor_drag_coefficient
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
            * params.body_drag_area
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
