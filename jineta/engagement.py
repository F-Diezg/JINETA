"""Engagement loop: one interceptor flying against one scripted target.

This module sits above `core`, `jinete` and (later) `campo`, because it composes
them: it needs the simulation loop, a vehicle with its autopilot, a guidance law
and a target. That is why it lives at the top of the package and not inside any of
the layers it uses (the dependency arrows only point downwards).

The target is a closed-form track `t -> (position, velocity)`, so it is evaluated
at the time of each step and needs no state of its own: only the interceptor is
integrated by `run`.
"""

from dataclasses import dataclass, field

import numpy as np

from jineta.core.events import ClosestApproach, closest_approach
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import Autopilot
from jineta.jinete.guidance import pure_pursuit_velocity
from jineta.jinete.targets import TargetTrack, track_positions


@dataclass(frozen=True)
class Engagement:
    """Result of one engagement."""

    times: np.ndarray  # (n,) [s]
    states: np.ndarray  # (n, 13) interceptor state
    target_positions: np.ndarray  # (n, 3) NED [m]
    closest: ClosestApproach  # smallest separation, interpolated inside its step
    captured: bool  # True: it got within the capture radius
    pursuit_feasible: bool  # False: commanded speed <= target speed, see `engage`


@dataclass
class EndOfPursuit:
    """Stop condition for `run`, with memory: `stop(t, y)` -> bool.

    Ends the run when either
      - the interceptor is within `capture_radius` of the target (it got there), or
      - it has closed in at some point and the separation has since grown more than
        `margin` above the smallest seen (it went past the target and is opening up
        again).

    Both are needed. After a capture the separation never grows again (the pursuer
    keeps circling the target), and after a miss it never gets below the radius.
    The "closed in at some point" condition matters at the start: an interceptor
    taking off from rest behind a fleeing target sees the separation grow for a few
    seconds before it starts to close, and that is not a pass.
    """

    target: TargetTrack
    capture_radius: float  # [m]
    margin: float  # [m]
    initial_distance: float  # separation at t = 0 [m]
    smallest: float = field(init=False)  # smallest separation seen so far [m]
    closed_in: bool = field(default=False, init=False)  # dropped below the start

    def __post_init__(self) -> None:
        if self.capture_radius <= 0.0 or self.margin <= 0.0:
            raise ValueError("capture_radius and margin must be > 0")
        self.smallest = self.initial_distance

    def __call__(self, t: float, state: np.ndarray) -> bool:
        distance = float(np.linalg.norm(self.target(t)[0] - state[:3]))
        if distance < self.smallest:
            self.smallest = distance
            self.closed_in = True
        captured = distance < self.capture_radius
        passed = self.closed_in and distance > self.smallest + self.margin
        return captured or passed


def engage(
    autopilot: Autopilot,
    target: TargetTrack,
    initial_state: np.ndarray,
    speed: float,
    t_end: float,
    capture_radius: float = 0.5,
    margin: float = 10.0,
) -> Engagement:
    """Fly the interceptor against the target with pure pursuit.

    Every step (`pre_step`) the guidance reads the target position at that time and
    sets the autopilot velocity command; the autopilot does the rest. The step is
    the autopilot period (`autopilot.dt`), so control and simulation cannot
    disagree.

    Args:
        autopilot: the interceptor's autopilot (it holds the vehicle).
        target: scripted track, `t -> (position, velocity)`.
        initial_state: 13-element interceptor state at t = 0.
        speed: commanded pursuit speed [m/s]. The vehicle may not reach it.
        t_end: maximum duration [s].
        capture_radius: separation that counts as reaching the target [m].
        margin: growth above the smallest separation that ends a miss [m].

    Returns:
        The `Engagement`. When it is captured, `closest.distance` is below the
        capture radius by construction (the run stops there), so it says "it got
        there", not how well. `pursuit_feasible` is False when `speed` does not
        exceed the target speed: pure pursuit cannot catch a target that is at
        least as fast (see tests/test_guidance.py). It is a statement about the
        commanded speed; a speed the vehicle cannot fly is not detected here.
    """
    autopilot.velocity_loop.reset()
    autopilot.rate_loop.reset()

    def guide(t: float, state: np.ndarray) -> None:
        autopilot.velocity_command = pure_pursuit_velocity(
            state[:3], target(t)[0], speed
        )
        autopilot(t, state)

    initial_distance = float(np.linalg.norm(target(0.0)[0] - initial_state[:3]))
    stop = EndOfPursuit(target, capture_radius, margin, initial_distance)
    times, states = run(
        autopilot.vehicle.derivative,
        initial_state,
        dt=autopilot.dt,
        t_end=t_end,
        stop=stop,
        post_step=normalize_attitude,
        pre_step=guide,
    )
    target_positions = track_positions(target, times)
    closest = closest_approach(times, target_positions - states[:, :3])
    target_speed = float(np.linalg.norm(target(0.0)[1]))
    return Engagement(
        times=times,
        states=states,
        target_positions=target_positions,
        closest=closest,
        captured=closest.distance < capture_radius,
        pursuit_feasible=speed > target_speed,
    )
