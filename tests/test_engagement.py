"""Tests for engagements: the judge of a pair, `engage` and fights of several pairs.

The judge of one pair is tested alone first (cheap, no dynamics). The closed-loop
tests fly the full cascade (6-DOF Iris + autopilot) and compare it with the
closed-form pure pursuit of an ideal point (see test_guidance.py): the real vehicle
can only be LATER than the ideal one, because its velocity lags the command. The
fights of several pairs use ideal points where the vehicle does not matter.
"""

import numpy as np
import pytest

from jineta.arena.engagement import (
    CAPTURED,
    OPEN,
    PASSED,
    TARGET_LOST,
    Duel,
    Pairing,
    engage,
    fight,
    pursuit,
)
from jineta.arena.world import PointEntity, TrackEntity, VehicleEntity, World
from jineta.jinete.control import Autopilot
from jineta.jinete.guidance import pure_pursuit_velocity
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import (
    RecordedTrack,
    StraightLineTarget,
    TurningTarget,
    track_positions,
)

DT = 0.01  # control and simulation step [s]


def judge(margin=10.0, initial_distance=50.0):
    return Duel(Pairing("i", "t", capture_radius=0.5, margin=margin), initial_distance)


def feed(duel, *distances):
    """Give the judge one separation per step; return the outcomes it reached."""
    outcomes = []
    for step, distance in enumerate(distances, start=1):
        duel.update(step, distance)
        outcomes.append(duel.outcome)
    return outcomes


def hover_at(position):
    """Iris state hovering (level, at rest) at a NED position."""
    return np.array([*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


def fly(target, start, speed, t_end, **kwargs):
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    return engage(autopilot, target, hover_at(start), speed, t_end, **kwargs)


def separation(result):
    return np.linalg.norm(result.target_positions - result.states[:, :3], axis=1)


# --- The judge of one pair alone ------------------------------------------------------


def test_a_capture_is_inside_the_capture_radius():
    duel = judge()
    assert feed(duel, 5.0, 0.6, 0.4) == [OPEN, OPEN, CAPTURED]
    assert duel.end_step == 3


def test_a_pass_is_growing_by_the_margin_over_the_smallest():
    duel = judge()
    feed(duel, 40.0, 30.0, 20.0, 12.0)  # closing in: smallest is 12
    assert feed(duel, 21.9) == [OPEN]  # opened up by 9.9 m: not yet
    assert feed(duel, 22.1) == [PASSED]  # 10.1 m above the smallest: it passed
    assert duel.end_step is None  # nothing was destroyed


def test_growing_separation_is_not_a_pass_until_it_has_closed_in():
    """Taking off from rest behind a fleeing target the separation grows at first;
    that is not a pass, however far it grows."""
    duel = judge()
    assert feed(duel, 52.0, 60.0, 75.0, 100.0) == [OPEN] * 4
    assert feed(duel, 49.0) == [OPEN]  # now it closes in...
    assert feed(duel, 60.0) == [PASSED]  # ...and then opens: a pass


def test_a_capture_after_a_pass_still_counts_and_a_capture_is_final():
    duel = judge()
    feed(duel, 10.0, 25.0)  # passed
    assert feed(duel, 3.0, 0.3) == [PASSED, CAPTURED]  # came round again
    assert feed(duel, 50.0) == [CAPTURED]  # nothing changes after a capture


def test_infinite_margin_never_passes():
    duel = judge(margin=np.inf)
    assert feed(duel, 1.0, 1e6) == [OPEN, OPEN]


@pytest.mark.parametrize(
    "interceptor, target, radius, margin",
    [
        ("i", "t", 0.0, 10.0),
        ("i", "t", -1.0, 10.0),
        ("i", "t", 0.5, 0.0),
        ("i", "i", 0.5, 10.0),  # chasing itself
    ],
)
def test_pairing_rejects_invalid_parameters(interceptor, target, radius, margin):
    with pytest.raises(ValueError):
        Pairing(interceptor, target, radius, margin)


# --- Closed loop with the Iris --------------------------------------------------------


def test_catches_a_slower_target_in_three_dimensions():
    """Target 40 m East and 20 m above, flying East at 4 m/s; interceptor at 10 m/s.
    The ideal point pursuer of the closed form (T = r0 (k + cos phi0) / (Vt (k^2 - 1)))
    catches it at T. The Iris starts from rest and follows the velocity command with
    a lag, so it can only take longer: between T and T + 3 s (the +3 s is a margin
    over the ~1.8 s seen when this test was written, not a derived bound)."""
    target = StraightLineTarget(start=[0.0, 40.0, -120.0], velocity=[0.0, 4.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=10.0, t_end=40.0)

    r0 = np.array([0.0, 40.0, -20.0])
    k = 10.0 / 4.0
    ideal = np.linalg.norm(r0) * (k + r0[1] / np.linalg.norm(r0)) / (4.0 * (k**2 - 1))

    assert result.pursuit_feasible
    assert result.captured
    assert result.closest.distance < 0.5
    assert ideal - 0.2 < result.closest.time < ideal + 3.0
    assert result.times[-1] < 40.0  # it stopped at the capture, not at t_end


def test_catches_a_fleeing_target_it_starts_behind():
    """Regression: from rest, behind a target that runs away at 10 m/s, the
    separation grows for the first seconds. With a margin of 5 m it grows by more
    than that, and it must not end the run as a miss: it has not closed in yet."""
    target = StraightLineTarget(start=[40.0, 0.0, -100.0], velocity=[10.0, 0.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=16.0, t_end=40.0, margin=5.0)
    distance = separation(result)
    assert distance.max() > distance[0] + 5.0  # it opened up by more than the margin
    assert result.captured


def test_a_faster_target_is_reported_infeasible_and_never_approached():
    target = StraightLineTarget(start=[50.0, 0.0, -100.0], velocity=[25.0, 0.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=20.0, t_end=10.0)
    assert not result.pursuit_feasible
    assert not result.captured
    assert result.closest.time == 0.0  # the closest it ever was is the start
    assert np.isclose(result.closest.distance, 50.0)
    assert separation(result)[-1] > 100.0


def test_a_pass_without_capture_ends_the_run_after_the_closest_approach():
    """A target crossing at 15 m/s against 10 m/s: the interceptor gets closer, goes
    past and falls behind; the run ends once the separation has grown by the margin
    over its smallest value."""
    target = StraightLineTarget(start=[100.0, -60.0, -100.0], velocity=[0.0, 15.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=10.0, t_end=30.0, margin=10.0)
    distance = separation(result)

    assert not result.pursuit_feasible
    assert not result.captured
    assert result.times[-1] < 30.0  # stopped early
    assert 0.0 < result.closest.time < result.times[-1]
    assert result.closest.distance < distance[0] - 10.0  # it did close in
    assert distance[-1] > distance.min() + 10.0  # and then opened by the margin


def test_the_same_autopilot_can_be_reused():
    """The integrators of the loops are reset at the start of each engagement."""
    target = StraightLineTarget(start=[0.0, 25.0, -100.0], velocity=[0.0, 3.0, 0.0])
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    first = engage(autopilot, target, hover_at([0.0, 0.0, -100.0]), 10.0, 20.0)
    second = engage(autopilot, target, hover_at([0.0, 0.0, -100.0]), 10.0, 20.0)
    assert first.captured
    assert np.array_equal(first.times, second.times)
    assert np.array_equal(first.states, second.states)


def test_negative_speed_is_rejected():
    target = StraightLineTarget(start=[10.0, 0.0, -100.0], velocity=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fly(target, start=[0.0, 0.0, -100.0], speed=-1.0, t_end=1.0)


def test_a_recorded_target_gives_the_same_engagement():
    # A simulated threat (e.g. the fixed wing) is flown first, recorded at the step
    # rate and replayed through `RecordedTrack`. Replaying a recorded turn must give
    # the engagement of the turn itself: the samples fall on the step grid.
    turn = TurningTarget(
        start=[30.0, 30.0, -110.0], speed=5.0, heading=1.0, turn_rate=0.1
    )
    times = DT * np.arange(round(40.0 / DT) + 1)
    velocities = np.array([turn(t)[1] for t in times])
    recorded = RecordedTrack(times, track_positions(turn, times), velocities)
    start = [0.0, 0.0, -100.0]
    original = fly(turn, start=start, speed=10.0, t_end=30.0)
    replayed = fly(recorded, start=start, speed=10.0, t_end=30.0)
    assert original.captured and replayed.captured
    assert np.allclose(replayed.states, original.states, atol=1e-9)
    assert np.isclose(replayed.closest.time, original.closest.time, atol=1e-9)


# --- Several pairs in one world ---------------------------------------------------


def chase(point, target, speed):
    """Pure pursuit control for an ideal point."""

    def control(t, position, snapshot):
        point.velocity_command = pure_pursuit_velocity(
            position, snapshot.position(target), speed
        )

    return control


def ideal_pursuer(name, start, target, speed):
    point = PointEntity(name, start)
    point.control = chase(point, target, speed)
    return point


def test_two_pairs_in_one_world_fly_each_as_if_alone():
    """Two Iris interceptors, two targets. Until its own capture each pair is the
    engagement it would be alone (to the bit); after it, both of the pair stay where
    they collided while the other pair goes on."""
    near = StraightLineTarget(start=[0.0, 40.0, -120.0], velocity=[0.0, 4.0, 0.0])
    far = StraightLineTarget(start=[90.0, 0.0, -100.0], velocity=[3.0, 0.0, 0.0])
    starts = {
        "near": hover_at([0.0, 0.0, -100.0]),
        "far": hover_at([0.0, 20.0, -100.0]),
    }
    speeds = {"near": 10.0, "far": 12.0}
    tracks = {"near": near, "far": far}

    alone = {
        key: fly(tracks[key], starts[key][:3], speeds[key], t_end=60.0)
        for key in tracks
    }

    entities, pairings = [], []
    for key in ("near", "far"):
        autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
        control = pursuit(autopilot, f"threat-{key}", speeds[key])
        entities.append(
            VehicleEntity(
                f"iris-{key}", autopilot.vehicle.derivative, starts[key], control
            )
        )
        entities.append(TrackEntity(f"threat-{key}", tracks[key]))
        pairings.append(Pairing(f"iris-{key}", f"threat-{key}"))
    battle = fight(World(entities), pairings, DT, 60.0)

    assert [r.outcome for r in battle.results] == [CAPTURED, CAPTURED]
    flight = battle.flight
    assert len(alone["near"].times) < len(alone["far"].times) == len(flight.times)
    for key, result in zip(("near", "far"), battle.results, strict=True):
        n = len(alone[key].times)  # samples up to its capture
        assert np.array_equal(flight.state(f"iris-{key}")[:n], alone[key].states)
        assert result.closest == alone[key].closest
        # after the capture both stay where they collided
        assert np.all(flight.state(f"iris-{key}")[n:] == alone[key].states[-1])
        hit = flight.positions(f"threat-{key}")[n - 1]
        assert np.all(flight.positions(f"threat-{key}")[n:] == hit)


def test_two_interceptors_on_one_target_the_second_loses_it():
    target = StraightLineTarget(start=[100.0, 0.0, -100.0], velocity=[0.0, 5.0, 0.0])
    world = World(
        [
            ideal_pursuer("near", [60.0, 0.0, -100.0], "threat", 10.0),
            ideal_pursuer("far", [0.0, 0.0, -100.0], "threat", 10.0),
            TrackEntity("threat", target),
        ]
    )
    battle = fight(
        world, [Pairing("near", "threat"), Pairing("far", "threat")], DT, 60.0
    )
    near, far = battle.results
    assert (near.outcome, far.outcome) == (CAPTURED, TARGET_LOST)
    assert battle.flight.times[-1] < 60.0  # nothing left to judge: the run ended
    assert far.closest.distance > 30.0  # it never got close


def test_the_fight_ends_when_every_pair_is_decided():
    """One pair captures early; the other is a crossing target faster than its
    pursuer, so it ends as a pass. The run ends when that pass is seen, and the
    first pair stays still meanwhile."""
    slow = StraightLineTarget(start=[30.0, 0.0, -100.0], velocity=[0.0, 2.0, 0.0])
    crossing = StraightLineTarget(
        start=[100.0, -60.0, -100.0], velocity=[0.0, 15.0, 0.0]
    )
    world = World(
        [
            ideal_pursuer("a", [0.0, 0.0, -100.0], "slow", 10.0),
            TrackEntity("slow", slow),
            ideal_pursuer("b", [0.0, 0.0, -110.0], "crossing", 10.0),
            TrackEntity("crossing", crossing),
        ]
    )
    battle = fight(world, [Pairing("a", "slow"), Pairing("b", "crossing")], DT, 60.0)
    first, second = battle.results
    assert (first.outcome, second.outcome) == (CAPTURED, PASSED)
    assert first.closest.time < second.closest.time < battle.flight.times[-1] < 60.0
    gap = battle.flight.positions("crossing") - battle.flight.positions("b")
    distance = np.linalg.norm(gap, axis=1)
    assert distance[-1] > second.closest.distance + 10.0  # opened by the margin


def test_a_fight_rejects_unknown_names_and_no_pairings():
    world = World([PointEntity("a", [0.0, 0.0, 0.0])])
    with pytest.raises(KeyError):
        fight(world, [Pairing("a", "ghost")], DT, 1.0)
    with pytest.raises(ValueError):
        fight(world, [], DT, 1.0)
