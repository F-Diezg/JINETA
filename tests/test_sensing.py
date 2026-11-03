"""Tests for sensors in the world: radar stations, knowledge and launch on track."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from jineta.arena.engagement import CAPTURED, Pairing, fight, pursuit
from jineta.arena.launch import Dome, LaunchOnAlert, LaunchOnTrack
from jineta.arena.scenario import at_rest
from jineta.arena.sensing import (
    RadarStation,
    TrackSettings,
    perfect_knowledge,
)
from jineta.arena.world import TrackEntity, VehicleEntity, World
from jineta.atalaya.profiles import SHORT_RANGE_RADAR
from jineta.atalaya.radar import Radar, RadarParams
from jineta.atalaya.tracker import Track
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import StraightLineTarget, TurningTarget

DT = 0.01
SETTINGS = TrackSettings(process_noise=1.0, speed_prior=30.0)
# Almost no noise (exactly none would leave the filter a singular S).
NEAR_PERFECT = RadarParams(1e-4, 1e-7, 1e-7, rate=10.0, max_range=5000.0)
# 8 m/s towards the origin, slightly across: inside 400 m at 6.74 s.
THREAT = StraightLineTarget([400.0, -150.0, -110.0], [-6.0, 2.0, 0.0])


def station(params=NEAR_PERFECT, seed=1, targets=("threat",)) -> RadarStation:
    return RadarStation(
        Radar(params, np.zeros(3), seed=seed), {name: SETTINGS for name in targets}
    )


def iris(knowledge, start, waiting=False) -> VehicleEntity:
    """An Iris flying pure pursuit at 16 m/s against "threat"."""
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    control = pursuit(autopilot, "threat", 16.0, knowledge)
    return VehicleEntity(
        "iris", autopilot.vehicle.derivative, at_rest(start), control, waiting
    )


def crossing_time(radius: float) -> float:
    """When THREAT comes within `radius` of the origin (closed form, solved)."""
    return brentq(lambda t: np.linalg.norm(THREAT(t)[0]) - radius, 0.0, 60.0)


def test_the_station_in_the_loop_is_the_radar_offline():
    """In the world, the station measures the truth at its radar's rate and feeds
    the track; the same radar and track run by hand over the same times give
    exactly the same measurements and the same estimate."""
    turn = TurningTarget(
        [900.0, 300.0, -120.0], speed=18.0, heading=2.5, turn_rate=0.05
    )
    radar_station = station(SHORT_RANGE_RADAR, seed=7, targets=("drone",))
    world = World([TrackEntity("drone", turn)], [radar_station])
    flight = world.fly(DT, 6.0)

    radar = Radar(SHORT_RANGE_RADAR, np.zeros(3), seed=7)
    track = Track("drone", SETTINGS.process_noise, SETTINGS.speed_prior)
    measured = []
    for t in flight.times[:-1]:  # the rules run at the start of every step
        for m in radar.scan(t, {"drone": turn(t)[0]}):
            track.update(m)
            measured.append(m.values)
    assert len(measured) == len(radar_station.measurements) == 60
    found = [m.values for m in radar_station.measurements]
    assert np.array_equal(np.array(found), np.array(measured))
    assert np.array_equal(radar_station.tracks["drone"].x, track.x)
    assert np.array_equal(radar_station.tracks["drone"].P, track.P)
    history = radar_station.history["drone"]
    assert len(history) == 60 and np.array_equal(history[-1].x, track.x)


def test_flying_again_repeats_the_run():
    """The world resets the station (tracks, measurements, the radar's noise) at the
    start of every flight."""
    radar_station = station(SHORT_RANGE_RADAR, seed=3)
    world = World([TrackEntity("threat", THREAT)], [radar_station])
    world.fly(DT, 3.0)
    first = np.array([m.values for m in radar_station.measurements])
    first_x = radar_station.tracks["threat"].x.copy()
    world.fly(DT, 3.0)
    again = np.array([m.values for m in radar_station.measurements])
    assert np.array_equal(again, first)
    assert np.array_equal(radar_station.tracks["threat"].x, first_x)


def test_perfect_knowledge_is_the_truth_bit_for_bit():
    """`perfect_knowledge` reads the snapshot, as the guidance did before phase 9:
    the engagement is the same to the last bit."""
    flights = []
    for knowledge in (None, perfect_knowledge):
        world = World(
            [iris(knowledge, [0.0, 0.0, -100.0]), TrackEntity("threat", THREAT)]
        )
        flights.append(fight(world, [Pairing("iris", "threat")], DT, 40.0).flight)
    assert np.array_equal(flights[0].states, flights[1].states)


def test_a_near_perfect_radar_gives_the_perfect_engagement():
    """Guided by the track of an almost noise-free radar, the Iris flies the
    engagement it flies with perfect knowledge: the same capture time and the
    closest approach within 1 mm."""
    results = []
    for use_radar in (False, True):
        radar_station = station()
        knowledge = radar_station.knowledge if use_radar else None
        world = World(
            [iris(knowledge, [0.0, 0.0, -100.0]), TrackEntity("threat", THREAT)],
            [radar_station] if use_radar else [],
        )
        results.append(fight(world, [Pairing("iris", "threat")], DT, 60.0).results[0])
    perfect, tracked = results
    assert perfect.outcome == tracked.outcome == CAPTURED
    assert tracked.end_time == perfect.end_time
    assert abs(tracked.closest.distance - perfect.closest.distance) < 1e-3


def test_without_a_track_the_interceptor_holds_still():
    """A hovering Iris whose radar sees only 300 m: until the first scan after the
    threat comes within 300 m (23.44 s, so the scan at 23.5 s) it has nothing to
    chase and holds its position; then it goes."""
    short = RadarParams(1e-4, 1e-7, 1e-7, rate=10.0, max_range=300.0)
    radar_station = station(short)
    world = World(
        [
            iris(radar_station.knowledge, [0.0, 0.0, -100.0]),
            TrackEntity("threat", THREAT),
        ],
        [radar_station],
    )
    flight = world.fly(DT, 25.0)
    assert radar_station.measurements[0].time == pytest.approx(23.5)
    assert crossing_time(300.0) == pytest.approx(23.44, abs=0.005)
    positions, times = flight.positions("iris"), flight.times
    before = times < 23.5 - 1e-9
    assert np.allclose(positions[before], [0.0, 0.0, -100.0], rtol=0, atol=1e-9)
    assert np.hypot(*positions[-1, :2]) > 1.0  # moving towards the threat


def test_launch_on_track_with_a_near_perfect_radar_is_the_launch_on_alert():
    """With an almost perfect radar the estimate crosses the 400 m dome when the
    truth does: the same launch step as launching on the truth (6.75 s, the first
    step after the crossing at 6.744 s)."""
    launches = []
    for on_track in (False, True):
        radar_station = station()
        dome = Dome([0.0, 0.0, 0.0], 400.0)
        if on_track:
            rule = LaunchOnTrack("iris", dome, radar_station, ["threat"])
        else:
            rule = LaunchOnAlert("iris", dome, ["threat"])
        world = World(
            [
                iris(radar_station.knowledge if on_track else None, [0, 0, 0], True),
                TrackEntity("threat", THREAT),
            ],
            [radar_station, rule],
        )
        launches.append(world.fly(DT, 8.0).launched_at["iris"])
    assert launches[0] == launches[1] == pytest.approx(6.75)
    assert 6.74 < crossing_time(400.0) < 6.75


def test_launch_on_track_waits_for_the_radar():
    """A radar that sees only 300 m cannot raise the alert of a 400 m dome when
    the threat crosses it: the launch waits for the first scan that sees it, the
    first one at or after it comes within 300 m (23.5 s)."""
    short = RadarParams(1e-4, 1e-7, 1e-7, rate=10.0, max_range=300.0)
    radar_station = station(short)
    dome = Dome([0.0, 0.0, 0.0], 400.0)
    world = World(
        [iris(radar_station.knowledge, [0, 0, 0], True), TrackEntity("threat", THREAT)],
        [radar_station, LaunchOnTrack("iris", dome, radar_station, ["threat"])],
    )
    assert world.fly(DT, 25.0).launched_at["iris"] == pytest.approx(23.5)
    first_scan_inside = math.ceil(crossing_time(300.0) * 10.0) / 10.0
    assert first_scan_inside == pytest.approx(23.5)


def test_a_destroyed_target_is_no_longer_measured():
    radar_station = station(targets=("a", "b"))

    def destroy_a(snapshot) -> None:
        if snapshot.time >= 1.0 - 1e-9 and not snapshot.world.entity("a").destroyed:
            snapshot.world.destroy("a", snapshot.time)

    world = World(
        [
            TrackEntity("a", StraightLineTarget([500.0, 0.0, -100.0], [-5.0, 0, 0])),
            TrackEntity("b", StraightLineTarget([0.0, 500.0, -100.0], [0, -5.0, 0])),
        ],
        [destroy_a, radar_station],
    )
    world.fly(DT, 2.0)
    times = {
        name: [m.time for m in radar_station.measurements if m.target == name]
        for name in "ab"
    }
    assert len(times["a"]) == 10 and max(times["a"]) < 1.0
    assert len(times["b"]) == 20


def test_station_rejects_unknown_targets_and_an_empty_list():
    radar_station = station()
    world = World([TrackEntity("threat", THREAT)], [radar_station])
    snapshot = world.snapshot(0.0, world.initial_state())
    assert radar_station.knowledge("threat", snapshot) is None  # no scan yet
    with pytest.raises(KeyError):
        radar_station.knowledge("other", snapshot)
    with pytest.raises(ValueError):
        RadarStation(Radar(NEAR_PERFECT, np.zeros(3)), {})
