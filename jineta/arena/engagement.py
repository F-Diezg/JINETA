"""Engagements in the world: interceptors against their targets, judged per pair.

A scenario says who chases whom (a `Pairing`: interceptor -> target; choosing the
pairs is phase 11). The world flies everyone together (`world.py`); a `Referee`
watches every pair after each step:

  - capture: the separation is inside the pair's capture radius. Both are
    destroyed (they collide): they stop where they are and leave the fight.
  - pass: the interceptor has closed in at some point and the separation has since
    grown more than `margin` above the smallest seen. It keeps chasing (a capture
    afterwards still counts), but the pair is decided.
  - lost: one of the two was destroyed in another pair (two interceptors on one
    target: the second one loses its target).
  - escaped: the pair defends a dome (`Pairing.dome`) and the target, after being
    inside it, has left it: past that point the site no longer cares about it.

The run ends when every pair is decided, or at `t_end`.

`engage` is the one-interceptor, one-scripted-target case of the earlier phases,
written on top of the world: one pair, the same result as before to the bit.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from jineta.arena.launch import Dome
from jineta.arena.world import (
    Control,
    Flight,
    Snapshot,
    TrackEntity,
    VehicleEntity,
    World,
)
from jineta.core.events import ClosestApproach, closest_approach
from jineta.jinete.control import Autopilot
from jineta.jinete.guidance import pure_pursuit_velocity
from jineta.jinete.targets import TargetTrack

OPEN, CAPTURED, PASSED = "open", "captured", "passed"
TARGET_LOST, INTERCEPTOR_LOST = "target lost", "interceptor lost"
ESCAPED = "escaped"


@dataclass(frozen=True)
class Pairing:
    """One interceptor assigned to one target."""

    interceptor: str  # entity name
    target: str  # entity name
    capture_radius: float = 0.5  # [m]
    margin: float = 10.0  # growth over the smallest separation that is a pass [m]
    dome: Dome | None = None  # the area it defends: leaving it ends the pair

    def __post_init__(self) -> None:
        if self.interceptor == self.target:
            raise ValueError("an entity cannot chase itself")
        if not self.capture_radius > 0.0 or not self.margin > 0.0:
            raise ValueError("capture_radius and margin must be > 0")


@dataclass
class Duel:
    """The judge of one pair: it is told the separation after every step.

    Both conditions of the end of a pursuit are needed. After a capture the
    separation never grows again, and after a miss it never gets below the radius.
    The "closed in at some point" condition matters at the start: an interceptor
    taking off from rest behind a fleeing target sees the separation grow for a few
    seconds before it starts to close, and that is not a pass.
    """

    pairing: Pairing
    initial_distance: float  # separation at t = 0 [m]
    smallest: float = field(init=False)  # smallest separation seen so far [m]
    closed_in: bool = field(default=False, init=False)  # dropped below the start
    outcome: str = field(default=OPEN, init=False)
    end_step: int | None = field(default=None, init=False)  # capture, loss, escape
    seen_inside: bool = field(default=False, init=False)  # target inside the dome

    def __post_init__(self) -> None:
        self.smallest = self.initial_distance

    @property
    def decided(self) -> bool:
        return self.outcome != OPEN

    @property
    def finished(self) -> bool:
        """Nothing can change any more (captured or lost; a pass can still turn
        into a capture)."""
        return self.outcome not in (OPEN, PASSED)

    def update(self, step: int, distance: float) -> None:
        """Judge the separation after step `step`."""
        if self.finished:
            return
        if distance < self.smallest:
            self.smallest = distance
            self.closed_in = True
        if distance < self.pairing.capture_radius:
            self.outcome, self.end_step = CAPTURED, step
        elif (
            self.outcome == OPEN
            and self.closed_in
            and distance > self.smallest + self.pairing.margin
        ):
            self.outcome = PASSED

    def lose(self, step: int, outcome: str) -> None:
        """One of the two was destroyed in another pair, or the target escaped."""
        if not self.finished:
            self.outcome, self.end_step = outcome, step

    def watch_dome(self, step: int, target_position: np.ndarray) -> None:
        """Track the target against the dome: out again after being in, it escaped."""
        dome = self.pairing.dome
        if dome is None or self.finished:
            return
        if dome.contains(target_position):
            self.seen_inside = True
        elif self.seen_inside:
            self.lose(step, ESCAPED)


@dataclass
class Referee:
    """Stop condition for `World.fly` that judges every pair after each step.

    A capture destroys both entities at once, so a pair later in the list can lose
    its target in the same step (the pairs are judged in their order).
    """

    world: World
    duels: Sequence[Duel]
    step: int = field(default=0, init=False)  # index of the sample just computed

    def __call__(self, t: float, state: np.ndarray) -> bool:
        self.step += 1
        snapshot = self.world.snapshot(t, state)
        for duel in self.duels:
            if duel.finished:
                continue
            pair = duel.pairing
            target = snapshot.position(pair.target)
            gap = target - snapshot.position(pair.interceptor)
            duel.update(self.step, float(np.linalg.norm(gap)))
            duel.watch_dome(self.step, target)
            if duel.outcome == CAPTURED:
                self.world.destroy(pair.interceptor, t)
                self.world.destroy(pair.target, t)
                self.lost_with(pair)
        return all(duel.decided for duel in self.duels)

    def lost_with(self, captured: Pairing) -> None:
        """The other pairs of the two entities just destroyed are lost."""
        for duel in self.duels:
            pair = duel.pairing
            if pair is captured:
                continue
            if pair.target in (captured.interceptor, captured.target):
                duel.lose(self.step, TARGET_LOST)
            elif pair.interceptor in (captured.interceptor, captured.target):
                duel.lose(self.step, INTERCEPTOR_LOST)


@dataclass(frozen=True)
class DuelResult:
    """How one pair ended."""

    pairing: Pairing
    outcome: str  # "captured", "passed", "open", "escaped", "target lost",
    # "interceptor lost"
    closest: ClosestApproach  # up to the capture, loss or escape (else whole run)
    end_time: float  # when the pair ended (the end of the run if it never did) [s]

    @property
    def captured(self) -> bool:
        return self.outcome == CAPTURED


@dataclass(frozen=True)
class Battle:
    """Result of `fight`: the whole flight and one result per pair, in order."""

    flight: Flight
    results: tuple[DuelResult, ...]


def fight(world: World, pairings: Sequence[Pairing], dt: float, t_end: float) -> Battle:
    """Fly the world until every pair is decided (or t_end) and judge each pair.

    Controllers with memory are not reset here (see `World.fly`).
    """
    pairings = tuple(pairings)
    if not pairings:
        raise ValueError("a fight needs at least one pairing")
    for pair in pairings:
        world.entity(pair.interceptor)  # a clear error for an unknown name
        world.entity(pair.target)
    world.reset()
    start = world.snapshot(0.0, world.initial_state())
    duels = []
    for pair in pairings:
        target = start.position(pair.target)
        gap = target - start.position(pair.interceptor)
        duel = Duel(pair, float(np.linalg.norm(gap)))
        duel.seen_inside = pair.dome is not None and pair.dome.contains(target)
        duels.append(duel)
    flight = world.fly(dt, t_end, stop=Referee(world, duels))

    results = []
    for duel in duels:
        pair = duel.pairing
        end = len(flight.times) if duel.end_step is None else duel.end_step + 1
        relative = flight.positions(pair.target) - flight.positions(pair.interceptor)
        closest = closest_approach(flight.times[:end], relative[:end])
        results.append(
            DuelResult(pair, duel.outcome, closest, float(flight.times[end - 1]))
        )
    return Battle(flight, tuple(results))


def pursuit(autopilot: Autopilot, target: str, speed: float) -> Control:
    """Control of a multirotor flying pure pursuit at `speed` against `target`:
    every step the guidance points the velocity command at where the target is now,
    then the autopilot runs."""
    if not speed >= 0.0:
        raise ValueError("speed must be >= 0")

    def control(t: float, state: np.ndarray, snapshot: Snapshot) -> None:
        autopilot.velocity_command = pure_pursuit_velocity(
            state[:3], snapshot.position(target), speed
        )
        autopilot(t, state)

    return control


@dataclass(frozen=True)
class Engagement:
    """Result of one engagement (one interceptor, one scripted target)."""

    times: np.ndarray  # (n,) [s]
    states: np.ndarray  # (n, 13) interceptor state
    target_positions: np.ndarray  # (n, 3) NED [m]
    closest: ClosestApproach  # smallest separation, interpolated inside its step
    captured: bool  # True: it got within the capture radius
    pursuit_feasible: bool  # False: commanded speed <= target speed, see `engage`


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

    A world with two entities ("interceptor" and "target") and one pair. The step
    is the autopilot period (`autopilot.dt`), so control and simulation cannot
    disagree. The loop integrators are reset first, so one autopilot can fly
    several engagements with identical results.

    Args:
        autopilot: the interceptor's autopilot (it holds the vehicle).
        target: scripted track, `t -> (position, velocity)`.
        initial_state: 13-element interceptor state at t = 0.
        speed: commanded pursuit speed [m/s]. The vehicle may not reach it.
        t_end: maximum duration [s].
        capture_radius: separation that counts as reaching the target [m].
        margin: growth above the smallest separation that ends a miss [m]
            (`math.inf`: a pass never ends the run).

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
    world = World(
        [
            VehicleEntity(
                "interceptor",
                autopilot.vehicle.derivative,
                initial_state,
                control=pursuit(autopilot, "target", speed),
            ),
            TrackEntity("target", target),
        ]
    )
    pair = Pairing("interceptor", "target", capture_radius, margin)
    battle = fight(world, [pair], autopilot.dt, t_end)
    result = battle.results[0]
    return Engagement(
        times=battle.flight.times,
        states=battle.flight.state("interceptor"),
        target_positions=battle.flight.positions("target"),
        closest=result.closest,
        captured=result.closest.distance < capture_radius,
        pursuit_feasible=speed > float(np.linalg.norm(target(0.0)[1])),
    )
