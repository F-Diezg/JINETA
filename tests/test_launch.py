"""Tests for launches and for threats flown in the same loop (jineta/arena/launch.py).

A threat that reacts, the analytic check. A pursuer at speed vp chases an evader
that sits still until the pursuer is within a trigger range, and from then on flies
straight away from it at ve < vp. Both always move along the line between them, so
the problem is one-dimensional: each step of length dt shortens the gap by
    vp dt         while the gap is above the trigger range (the evader is still),
    (vp - ve) dt  once it is inside (the evader flees),
exactly, step by step. The pursuer reads where the evader is and the evader reads
where the pursuer is: the coupling goes both ways, which a recorded threat cannot do.
"""

import math

import numpy as np
import pytest

from jineta.arena.engagement import (
    CAPTURED,
    ESCAPED,
    Engagement,
    Pairing,
    engage,
    fight,
    pursuit,
)
from jineta.arena.launch import Dome, LaunchAt, LaunchOnAlert
from jineta.arena.world import PointEntity, TrackEntity, VehicleEntity, World
from jineta.core.events import ClosestApproach
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.control import Autopilot
from jineta.jinete.fixed_wing import FixedWing
from jineta.jinete.fixed_wing_control import FixedWingAutopilot
from jineta.jinete.guidance import pure_pursuit_velocity
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_GAINS,
    SMALL_FIXED_WING,
    SMALL_FIXED_WING_AUTOPILOT,
    SMALL_FIXED_WING_PROPULSION,
)
from jineta.jinete.targets import (
    RecordedTrack,
    StraightLineTarget,
    TurningTarget,
    track_positions,
)
from jineta.jinete.trim import trim_level_flight

DT = 0.01
PAD = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)  # at rest


def waiting_iris(target, speed):
    """An Iris at rest on the pad at the origin, waiting, set to pursue `target`."""
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    control = pursuit(autopilot, target, speed)
    return VehicleEntity(
        "iris", autopilot.vehicle.derivative, PAD, control=control, waiting=True
    )


# --- Dome and launch rules -----------------------------------------------------------


def test_the_dome_contains_its_surface_and_rejects_bad_input():
    dome = Dome([10.0, 0.0, 0.0], 5.0)
    assert dome.contains([10.0, 0.0, -5.0])  # on the surface
    assert dome.contains([12.0, 1.0, -1.0])
    assert not dome.contains([10.0, 5.1, 0.0])
    with pytest.raises(ValueError):
        Dome([0.0, 0.0], 5.0)
    with pytest.raises(ValueError):
        Dome([0.0, 0.0, 0.0], 0.0)


def test_launch_at_a_time():
    body = PointEntity(
        "dot", [0.0, 0.0, 0.0], waiting=True, velocity_command=[1.0, 0.0, 0.0]
    )
    flight = World([body], rules=[LaunchAt("dot", 0.5)]).fly(DT, 1.0)
    assert flight.launched_at["dot"] == pytest.approx(0.5)
    x = flight.positions("dot")[:, 0]
    assert np.all(x[flight.times <= 0.5 + 1e-9] == 0.0)
    assert x[-1] == pytest.approx(0.5)  # 0.5 s at 1 m/s


def test_launch_on_alert_at_the_first_step_the_threat_is_inside():
    """A threat flies straight at the pad at 20 m/s from 503.1 m: it is inside the
    400 m dome from t = 5.155 s, so the alert is the step at t = 5.16 s (away from
    the edge: a crossing exactly on a sample would depend on rounding). The Iris
    sits exactly still until that step."""
    threat = StraightLineTarget([503.1, 0.0, 0.0], [-20.0, 0.0, 0.0])
    dome = Dome([0.0, 0.0, 0.0], 400.0)
    world = World(
        [waiting_iris("threat", 10.0), TrackEntity("threat", threat)],
        rules=[LaunchOnAlert("iris", dome, ["threat"])],
    )
    flight = world.fly(DT, 6.0)
    t_launch = flight.launched_at["iris"]
    assert t_launch == pytest.approx(5.16, abs=1e-9)
    k = int(np.flatnonzero(flight.times == t_launch)[0])
    ranges = np.linalg.norm(flight.positions("threat"), axis=1)
    assert ranges[k] <= 400.0 < ranges[k - 1]
    iris = flight.state("iris")
    assert np.array_equal(iris[: k + 1], np.tile(PAD, (k + 1, 1)))
    assert iris[-1, 2] < 0.0  # it took off


def test_launch_on_alert_watches_several_threats_and_ignores_destroyed_ones():
    far = StraightLineTarget([800.1, 0.0, 0.0], [-20.0, 0.0, 0.0])  # in at 20.005 s
    near = StraightLineTarget([0.0, 300.0, 0.0], [0.0, 0.0, 0.0])  # already inside
    dome = Dome([0.0, 0.0, 0.0], 400.0)

    def world_with(rules):
        return World(
            [
                waiting_iris("near", 10.0),
                TrackEntity("near", near),
                TrackEntity("far", far),
            ],
            rules=rules,
        )

    alert = LaunchOnAlert("iris", dome, ["far", "near"])
    assert world_with([alert]).fly(DT, 0.1).launched_at["iris"] == 0.0

    def shot_down(snapshot):  # "near" is gone before the alert looks
        if not snapshot.world.entity("near").destroyed:
            snapshot.world.destroy("near", snapshot.time)

    flight = world_with([shot_down, alert]).fly(DT, 25.0)
    assert flight.launched_at["iris"] == pytest.approx(20.01, abs=1e-9)  # "far"


def test_a_destroyed_entity_cannot_be_launched():
    world = World([PointEntity("dot", [0.0, 0.0, 0.0], waiting=True)])
    world.reset()
    world.destroy("dot", 0.0)
    with pytest.raises(ValueError):
        world.launch("dot", 1.0)


# --- The dome ends a pair ------------------------------------------------------------


def test_the_pair_ends_when_the_threat_leaves_the_dome():
    """A fast threat crosses the 200 m dome; the slow interceptor never catches it.
    The pair ends at the first sample outside the dome after being inside."""
    threat = StraightLineTarget([300.0, 50.0, -100.0], [-40.0, 0.0, 0.0])
    dome = Dome([0.0, 0.0, 0.0], 200.0)
    chaser = PointEntity("chaser", [0.0, 0.0, -100.0])

    def control(t, position, snapshot):
        chaser.velocity_command = pure_pursuit_velocity(
            position, snapshot.position("threat"), 10.0
        )

    chaser.control = control
    world = World([chaser, TrackEntity("threat", threat)])
    pair = Pairing("chaser", "threat", margin=math.inf, dome=dome)
    battle = fight(world, [pair], DT, 60.0)
    result = battle.results[0]

    assert result.outcome == ESCAPED
    ranges = np.linalg.norm(battle.flight.positions("threat"), axis=1)
    assert ranges[-1] > 200.0 >= ranges[-2]  # the run ends at the first sample out
    assert result.end_time == battle.flight.times[-1] < 60.0


# --- The world reproduces the earlier way of doing it ------------------------------


def test_launch_in_the_world_is_waiting_offline_then_engaging():
    """Until phase 8 the scramble kept the Iris out of the simulation until the
    alert, then ran `engage` against the threat seen from the launch time and joined
    the two. A launch rule in the world must give the same, to rounding error."""
    threat = TurningTarget(
        start=[420.0, -250.0, -110.0],
        speed=8.0,
        heading=np.radians(155.0),
        turn_rate=np.radians(-1.5),
    )
    dome = Dome([0.0, 0.0, 0.0], 400.0)
    world = World(
        [waiting_iris("threat", 16.0), TrackEntity("threat", threat)],
        rules=[LaunchOnAlert("iris", dome, ["threat"])],
    )
    battle = fight(
        world, [Pairing("iris", "threat", margin=math.inf, dome=dome)], DT, 60.0
    )
    in_world = battle.flight

    # the earlier way: first step inside, standby rows, then engage from there
    grid = DT * np.arange(round(60.0 / DT) + 1)
    inside = np.linalg.norm(track_positions(threat, grid), axis=1) <= 400.0
    waiting = int(np.flatnonzero(inside)[0])
    t_launch = grid[waiting]
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    pursuit_only = engage(
        autopilot,
        lambda t: threat(t + t_launch),
        PAD,
        16.0,
        60.0 - t_launch,
        margin=math.inf,
    )
    offline = Engagement(
        times=np.concatenate([grid[:waiting], t_launch + pursuit_only.times]),
        states=np.vstack([np.tile(PAD, (waiting, 1)), pursuit_only.states]),
        target_positions=np.zeros((0, 3)),  # not compared
        closest=ClosestApproach(
            pursuit_only.closest.time + t_launch,
            pursuit_only.closest.distance,
            pursuit_only.closest.step + waiting,
        ),
        captured=pursuit_only.captured,
        pursuit_feasible=True,
    )

    assert battle.results[0].outcome == CAPTURED and offline.captured
    assert in_world.launched_at["iris"] == pytest.approx(t_launch, abs=1e-9)
    assert np.allclose(in_world.times, offline.times, rtol=0, atol=1e-9)
    assert np.allclose(in_world.state("iris"), offline.states, rtol=0, atol=1e-6)
    result = battle.results[0].closest
    assert result.step == offline.closest.step
    assert result.time == pytest.approx(offline.closest.time, abs=1e-6)
    assert result.distance == pytest.approx(offline.closest.distance, abs=1e-6)


def fixed_wing_threat():
    """The fixed wing trimmed at 18 m/s and 40 m up, flying South over the pad from
    150 m North, held on course and altitude by its autopilot."""
    aircraft = FixedWing(SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION)
    trim = trim_level_flight(aircraft, 18.0, altitude=40.0, course=np.pi)
    autopilot = FixedWingAutopilot(aircraft, trim, SMALL_FIXED_WING_AUTOPILOT, DT)
    start = np.array(trim.state)
    start[0:2] = [150.0, 0.0]
    return autopilot, start


def test_a_fixed_wing_in_the_same_loop_gives_the_recorded_engagement():
    """Loose end 19. The fixed wing does not react to the Iris, so flying it in the
    same loop must give what flying it first and replaying it gave (phase 6b): the
    guidance reads the target only at the start of each step, where the recording
    has its samples."""
    dome = Dome([0.0, 0.0, 0.0], 120.0)

    def scramble(threat):
        world = World(
            [waiting_iris("threat", 25.0), threat],
            rules=[LaunchOnAlert("iris", dome, ["threat"])],
        )
        pair = Pairing("iris", "threat", margin=math.inf, dome=dome)
        return fight(world, [pair], DT, 30.0)

    autopilot, start = fixed_wing_threat()
    live = scramble(
        VehicleEntity(
            "threat",
            autopilot.vehicle.derivative,
            start,
            control=lambda t, state, snapshot: autopilot(t, state),
        )
    )

    autopilot, start = fixed_wing_threat()
    times, states = run(
        autopilot.vehicle.derivative,
        start,
        DT,
        15.0,
        pre_step=autopilot,
        post_step=normalize_attitude,
    )
    recorded = TrackEntity(
        "threat", RecordedTrack(times, states[:, 0:3], states[:, 3:6])
    )
    replayed = scramble(recorded)

    a, b = live.results[0], replayed.results[0]
    assert a.outcome == b.outcome == ESCAPED  # it flew over the pad and away
    assert len(live.flight.times) == len(replayed.flight.times)
    assert np.allclose(
        live.flight.state("iris"), replayed.flight.state("iris"), rtol=0, atol=1e-6
    )
    assert a.closest.distance == pytest.approx(b.closest.distance, abs=1e-6)
    assert a.closest.distance < 5.0  # a real engagement: it got close


# --- A threat that reacts ------------------------------------------------------------


def test_a_threat_that_flees_from_its_pursuer():
    vp, ve, trigger, d0 = 10.0, 6.0, 50.0, 120.05
    pursuer = PointEntity("pursuer", [0.0, 0.0, -100.0])
    evader = PointEntity("evader", [d0, 0.0, -100.0])

    def chase(t, position, snapshot):
        pursuer.velocity_command = pure_pursuit_velocity(
            position, snapshot.position("evader"), vp
        )

    def flee(t, position, snapshot):
        away = position - snapshot.position("pursuer")
        distance = np.linalg.norm(away)
        evader.velocity_command = (
            ve * away / distance if distance <= trigger else np.zeros(3)
        )

    pursuer.control, evader.control = chase, flee
    battle = fight(World([pursuer, evader]), [Pairing("pursuer", "evader")], DT, 60.0)

    expected = [d0]  # the 1-D recursion of the module docstring
    while expected[-1] >= 0.5:
        d = expected[-1]
        expected.append(d - (vp if d > trigger else vp - ve) * DT)
    gap = battle.flight.positions("evader") - battle.flight.positions("pursuer")
    distance = np.linalg.norm(gap, axis=1)

    assert battle.results[0].outcome == CAPTURED
    assert len(distance) == len(expected)  # captured at the same step
    assert np.allclose(distance, expected, rtol=0, atol=1e-9)
    # it waited until the pursuer came within the trigger range, then fled East
    evader_x = battle.flight.positions("evader")[:, 0]
    still = distance > trigger
    k = int(np.flatnonzero(~still)[0])  # first sample inside the trigger range
    assert np.all(evader_x[: k + 1] == d0)
    assert np.all(np.diff(evader_x[k:]) > 0)
    # analytic capture time, to one step
    t_trigger = (d0 - trigger) / vp
    t_capture = t_trigger + (trigger - 0.5) / (vp - ve)
    assert battle.flight.times[-1] == pytest.approx(t_capture, abs=DT)
