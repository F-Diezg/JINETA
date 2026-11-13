"""Ground effect: a rotor close to the floor pushes harder.

A rotor pushes air down (its wake). Far from the floor the wake leaves freely;
close to it, the floor stops the wake, the air under the rotor slows down and
its pressure rises, and at the same rotor speed the rotor gives more thrust.
Cheeseman and Bennett (1955, ARC R&M 3021) modelled the floor as a mirror image
of the rotor (a source of air below the floor that cancels the flow through
it) and found, for a rotor of radius R whose disc is a height z above the
floor, hovering:

    T_ground / T_free = 1 / (1 - (R / 4z)^2)

and in forward flight, where the wake is swept back and touches the floor
farther behind,

    T_ground / T_free = 1 / (1 - (R / 4z)^2 / (1 + (V / v_i)^2))

with V the speed of the air along the rotor disc and v_i the rotor's induced
velocity in hover, from momentum theory v_i = sqrt(T / (2 rho A)), A = pi R^2.

At z = R the thrust is 16/15 of the free one (+6.7 %), at z = 2R +1.6 %, at
z = 4R +0.4 %: the effect matters within about two rotor radii of the floor.
The formula grows without limit as z falls to R/4, where it no longer holds;
here the height is not taken below `min_height` R (0.5 R by default: at most
4/3 of the free thrust).

How it is used (`Multirotor(..., ground_effect=GroundEffect())`): each rotor's
thrust is multiplied by its own factor, with z the height of its hub above the
floor (so a tilted drone close to the floor pushes harder on its low side, which
levels it), V the airspeed of its hub in the rotor plane, and T its thrust free
of the floor. Simplifications, on purpose: the rotor disc is taken as parallel
to the floor whatever the tilt; the reaction torque is left as it is free of
the floor (no open data); neighbouring rotors and the body are not counted
(multirotor corrections exist, Sanchez-Cuevas et al. 2017, and add to the
effect); only the room's floor counts (not the top of a crate); and the ceiling
effect (a rotor sucks harder close to a ceiling) is not modelled.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class GroundEffect:
    """The floor under the rotors: its height `floor_z` (NED z of the floor
    [m], 0 for a room) and the lowest height the formula is used at, as a
    fraction of the propeller radius."""

    floor_z: float = 0.0
    min_height: float = 0.5

    def __post_init__(self) -> None:
        if not self.min_height > 0.25:
            raise ValueError("min_height must be > 0.25 (the formula's pole)")


def ground_effect_factor(
    radius: float,
    height: npt.ArrayLike,
    thrust: npt.ArrayLike,
    inplane_speed: npt.ArrayLike,
    density: npt.ArrayLike,
    min_height: float = 0.5,
) -> np.ndarray:
    """T_ground / T_free for rotors of `radius` [m] whose hubs are `height`
    above the floor [m], each giving `thrust` free of the floor [N], with the
    air moving at `inplane_speed` along the disc [m/s], in air of `density`
    [kg/m^3] (Cheeseman and Bennett; see the module). Any shapes that
    broadcast; elementwise."""
    height = np.maximum(np.asarray(height, dtype=float), min_height * radius)
    ratio = (radius / (4.0 * height)) ** 2
    # (V / v_i)^2 with v_i^2 = T / (2 rho A): ratio / (1 + V^2 / v_i^2) =
    # ratio * v_i^2 / (v_i^2 + V^2), which is 0 without thrust (no wake).
    area = np.pi * radius * radius
    induced_squared = np.maximum(np.asarray(thrust, dtype=float), 0.0) / (
        2.0 * np.asarray(density, dtype=float) * area
    )
    speed_squared = np.asarray(inplane_speed, dtype=float) ** 2
    total = induced_squared + speed_squared
    weight = np.where(
        total > 0.0, induced_squared / np.where(total > 0.0, total, 1.0), 0.0
    )
    return 1.0 / (1.0 - ratio * weight)
