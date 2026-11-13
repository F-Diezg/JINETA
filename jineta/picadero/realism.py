"""What separates the simulator from a real room: the effects of phase 12.

`Realism` gathers the switches of the training environment (`CourseEnv(...,
realism=...)`). Every effect is off by default, so an environment without it
flies exactly as before phase 12, and each effect can be measured on its own.

- motor_lag: each rotor follows its speed command with the profile's first-order
  lag (12.5 ms speeding up, 25 ms slowing down for the Iris) instead of at once;
  the rotor speeds become part of the drones' states.
- ground_effect: each rotor pushes harder close to the room's floor
  (Cheeseman and Bennett's factor from the height of its hub; see
  `jinete.proximity`).
- sensors: the policy sees what the drone estimates from its sensors (motion
  capture, gyroscope, accelerometer; `atalaya.estimator`) instead of the true
  state; None: the true state.
- latency: the time from the moment the policy sees its observation to the
  moment its command reaches the rate loop [s] (rounded to physics steps): the
  companion computer's work, the link to the autopilot. The command applied
  during that time is the one sent before (zero thrust and rates at the start
  of an episode).

`LAB` switches every effect on with the values of a motion-capture lab
(`atalaya.estimator.MOCAP_LAB`) and 20 ms of latency. No open measurement of
the latency from a companion computer to PX4's rate loop was found: 20 ms is a
chosen value, to be measured on the real drone (phase 14) and randomised in
training.
"""

from dataclasses import dataclass

from jineta.atalaya.estimator import MOCAP_LAB, Sensors


@dataclass(frozen=True)
class Realism:
    """The effects of a real flight the environment simulates (all off by
    default; see the module)."""

    motor_lag: bool = False
    ground_effect: bool = False
    sensors: Sensors | None = None
    latency: float = 0.0

    def __post_init__(self) -> None:
        if not self.latency >= 0.0:
            raise ValueError("latency must be >= 0")

    def latency_steps(self, dt: float) -> int:
        """The latency in whole physics steps."""
        return round(self.latency / dt)


IDEAL = Realism()  # nothing on: the simulator of phase 11
LAB = Realism(motor_lag=True, ground_effect=True, sensors=MOCAP_LAB, latency=0.02)
