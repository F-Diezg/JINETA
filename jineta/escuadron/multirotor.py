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
from jineta.jinete.multirotor import MultirotorParams


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
    rotor_speed_command: np.ndarray = field(init=False)
    body: BatchRigidBody = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        self.rotor_speed_command = np.zeros((self.n, len(self.params.rotors)))
        self.body = BatchRigidBody(
            mass=self.params.mass,
            inertia=self.params.inertia,
            force_moment=self.force_moment,
        )

    def derivative(self, t: float | np.ndarray, states: np.ndarray) -> np.ndarray:
        """d(states)/dt for the states (n, 13)."""
        return self.body.derivative(t, states)

    def force_moment(
        self, t: float | np.ndarray, states: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forces and moments in body axes, (n, 3) each [N], [N·m]: rotors, drag
        and contact (no gravity: the BatchRigidBody adds it)."""
        params = self.params
        n = len(states)
        speeds = np.clip(self.rotor_speed_command, 0.0, params.max_rotor_speed)
        rho = np.broadcast_to(np.asarray(self.density(-states[:, 2]), float), (n,))
        density_ratio = rho / params.reference_density

        # Rotor thrust and reaction torque, through the mixer: the wrench
        # [T, Mx, My, Mz] = A @ thrusts, adding the rotors one after the other.
        thrusts = density_ratio[:, None] * params.thrust_coefficient * speeds**2
        allocation = params.allocation
        wrench = thrusts[:, 0:1] * allocation[:, 0]
        for i in range(1, len(params.rotors)):
            wrench = wrench + thrusts[:, i : i + 1] * allocation[:, i]
        force = np.zeros((n, 3))
        force[:, 2] = -wrench[:, 0]
        moment = wrench[:, 1:4].copy()

        # Rotor drag at each hub, opposing the in-plane airspeed of the hub.
        rotation = quaternion.to_matrix(states[:, 6:10])
        air_ned = states[:, 3:6] - self.wind(t, states[:, 0:3])
        airspeed = matvec(rotation.swapaxes(-1, -2), air_ned)
        omega = states[:, 10:13]
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
