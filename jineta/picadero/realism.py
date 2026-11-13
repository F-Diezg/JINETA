"""What separates the simulator from a real room: the effects of phase 12.

`Realism` gathers the switches of the training environment (`CourseEnv(...,
realism=...)`). Every effect is off by default, so an environment without it
flies exactly as before phase 12, and each effect can be measured on its own.

- motor_lag: each rotor follows its speed command with the profile's first-order
  lag (12.5 ms speeding up, 25 ms slowing down for the Iris) instead of at once;
  the rotor speeds become part of the drones' states.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Realism:
    """The effects of a real flight the environment simulates (all off by
    default; see the module)."""

    motor_lag: bool = False


IDEAL = Realism()  # nothing on: the simulator of phase 11
