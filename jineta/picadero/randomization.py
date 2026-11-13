"""Domain randomisation: every episode a slightly different drone.

The numbers of a simulated drone are only approximately those of the real one:
its mass with or without a camera, an inertia estimated from a drawing, the
thrust of propellers that wear and of a battery that runs down, drag coefficients
that nobody measured. A policy trained on one exact drone can learn to rely on
numbers that are wrong (phase 11's policy flew on a knife's edge). Domain
randomisation draws the numbers anew at the start of every episode, inside
ranges that cover what the real drone could be, so the policy learns to fly all
of them, and with luck the real one among them (Tobin et al. 2017 named it, for
images; for the dynamics of robots it is the usual way from a simulator to a
real vehicle).

`Randomization` gives the ranges: factors on the profile's numbers, drawn
uniformly (or uniformly in their logarithm, for the ranges that span a factor of
two either way, so that half and double are as likely), and the latency in
seconds. The controller (the rate loop) keeps the profile's numbers: it does not
know which drone it flies, as PX4 does not.

The ranges are chosen here, not measured (the real drone is measured in phase
14); each says why:
  - mass x [0.9, 1.1]: a payload or another battery;
  - inertia x [0.8, 1.2], one factor for the whole tensor (scaling each axis on
    its own breaks the triangle inequality of a flat drone like the Iris, whose
    I_zz is almost I_xx + I_yy);
  - thrust coefficient x [0.9, 1.1]: propellers, their wear, the air;
  - maximum rotor speed x [0.85, 1.0]: the battery running down (a LiPo cell
    from 4.2 V full to ~3.7 V, -12 %), which is how the battery is covered in
    this phase;
  - moment constant x [0.8, 1.2];
  - rotor drag and body drag x [0.5, 2] (log): the profile's weakest numbers
    (the body drag is a generic PX4 default);
  - motor time constants x [0.5, 2] (log), one factor for both: the Iris's are
    the simulator's, not a measurement;
  - latency [0, 40] ms: 20 ms is the chosen nominal (`realism.LAB`).
"""

from dataclasses import dataclass

import numpy as np

from jineta.jinete.multirotor import MultirotorParams

Range = tuple[float, float]


@dataclass(frozen=True)
class Randomization:
    """The ranges drawn at the start of every episode (see the module):
    factors on the profile's numbers, and the latency [s]."""

    mass: Range = (0.9, 1.1)
    inertia: Range = (0.8, 1.2)
    thrust_coefficient: Range = (0.9, 1.1)
    max_rotor_speed: Range = (0.85, 1.0)
    moment_constant: Range = (0.8, 1.2)
    rotor_drag: Range = (0.5, 2.0)
    body_drag: Range = (0.5, 2.0)
    motor_time_constant: Range = (0.5, 2.0)
    latency: Range = (0.0, 0.04)

    # Ranges drawn uniformly in their logarithm.
    LOG = ("rotor_drag", "body_drag", "motor_time_constant")

    def __post_init__(self) -> None:
        for name in (
            "mass",
            "inertia",
            "thrust_coefficient",
            "max_rotor_speed",
            "moment_constant",
            "rotor_drag",
            "body_drag",
            "motor_time_constant",
        ):
            low, high = getattr(self, name)
            if not 0.0 < low <= high:
                raise ValueError(f"{name} must be a range 0 < low <= high")
        low, high = self.latency
        if not 0.0 <= low <= high:
            raise ValueError("latency must be a range 0 <= low <= high")

    def _draw(self, name: str, rng: np.random.Generator, k: int) -> np.ndarray:
        low, high = getattr(self, name)
        if name in self.LOG:
            return np.exp(rng.uniform(np.log(low), np.log(high), k))
        return rng.uniform(low, high, k)

    def draw(
        self, params: MultirotorParams, rng: np.random.Generator, k: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """The numbers of k new drones around the profile `params` (keywords of
        `BatchMultirotor.set_parameters`) and their latencies (k,) [s]."""
        tau = self._draw("motor_time_constant", rng, k)
        values = {
            "mass": params.mass * self._draw("mass", rng, k),
            "inertia": params.inertia * self._draw("inertia", rng, k)[:, None, None],
            "thrust_coefficient": params.thrust_coefficient
            * self._draw("thrust_coefficient", rng, k),
            "max_rotor_speed": params.max_rotor_speed
            * self._draw("max_rotor_speed", rng, k),
            "moment_constant": params.moment_constant
            * self._draw("moment_constant", rng, k),
            "rotor_drag_coefficient": params.rotor_drag_coefficient
            * self._draw("rotor_drag", rng, k),
            "body_drag_area": params.body_drag_area
            * self._draw("body_drag", rng, k)[:, None],
            "motor_time_constant_up": params.motor_time_constant_up * tau,
            "motor_time_constant_down": params.motor_time_constant_down * tau,
        }
        return values, rng.uniform(*self.latency, k)

    @property
    def max_latency(self) -> float:
        return self.latency[1]


# No randomisation at all: every range a single point (the profile's numbers,
# no latency); a check that randomising changes nothing by itself.
NONE = Randomization(
    mass=(1.0, 1.0),
    inertia=(1.0, 1.0),
    thrust_coefficient=(1.0, 1.0),
    max_rotor_speed=(1.0, 1.0),
    moment_constant=(1.0, 1.0),
    rotor_drag=(1.0, 1.0),
    body_drag=(1.0, 1.0),
    motor_time_constant=(1.0, 1.0),
    latency=(0.0, 0.0),
)
