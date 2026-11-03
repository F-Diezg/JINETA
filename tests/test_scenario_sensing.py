"""Tests for scenarios with radars (phase 9): what `build` makes of a radar, a
radar-guided interceptor and the seed, and that a radar nobody uses changes nothing.
"""

from pathlib import Path

import numpy as np
import yaml

from jineta.arena.launch import LaunchOnAlert, LaunchOnTrack
from jineta.arena.scenario import build, play, radar_seeds
from jineta.arena.sensing import RadarStation, TrackSettings
from jineta.arena.spec import load_scenario, parse_scenario
from jineta.atalaya.profiles import SHORT_RANGE_RADAR

ROOT = Path(__file__).resolve().parents[1]

TWO_RADARS = """
name: "two radars"
duration: 30
seed: 5
sites:
  - {id: north, position: [300, 0], alert_range: 400, radar: short_range}
  - {id: south, position: [-300, 0], alert_range: 400, radar: short_range}
  - {id: field, position: [0, 500]}
interceptors:
  - {id: n, vehicle: iris, site: north, target: a, speed: 16, knowledge: radar}
  - {id: s, vehicle: iris, site: south, target: b, speed: 16}
threats:
  - {id: a, kind: straight, position: [900, 0], altitude: 100, speed: 8,
     course_deg: 180}
  - {id: b, kind: straight, position: [-900, 0], altitude: 100, speed: 8,
     course_deg: 0, tracking: {process_noise: 3, speed_prior: 15}}
"""


def spec(text: str = TWO_RADARS, **changes):
    data = yaml.safe_load(text)
    data.update(changes)
    return parse_scenario(data)


def test_a_site_with_a_radar_becomes_a_station_that_tracks_every_threat():
    setup = build(spec())
    assert list(setup.stations) == ["north", "south"]  # the field has no radar
    north = setup.stations["north"]
    assert north.radar.params is SHORT_RANGE_RADAR
    assert np.array_equal(north.radar.site, [300.0, 0.0, 0.0])
    assert list(north.targets) == ["a", "b"]  # every threat, in the file's order
    assert north.targets["b"] == TrackSettings(process_noise=3.0, speed_prior=15.0)


def test_the_stations_come_first_and_each_interceptor_gets_its_launch_rule():
    """The radar-guided interceptor launches on its station's track, the other on
    the truth; the stations run before both."""
    setup = build(spec())
    kinds = [type(rule) for rule in setup.world.rules]
    assert kinds == [RadarStation, RadarStation, LaunchOnTrack, LaunchOnAlert]
    on_track = setup.world.rules[2]
    assert (on_track.interceptor, on_track.station) == ("n", setup.stations["north"])


def test_every_radar_gets_its_own_seed_from_the_scenario_seed():
    seeds = radar_seeds(5, 2)
    assert seeds == radar_seeds(5, 2)  # the same file, the same noise
    assert seeds[0] != seeds[1]  # independent streams
    assert radar_seeds(6, 2) != seeds  # another seed, another run
    setup = build(spec())
    assert [s.radar.seed for s in setup.stations.values()] == seeds


def test_the_seed_decides_the_noise_and_repeats_it():
    def first_scans(seed: int) -> np.ndarray:
        setup = build(spec(seed=seed))
        setup.world.fly(0.01, 0.5)
        station = setup.stations["north"]
        return np.array([m.values for m in station.measurements])

    assert np.array_equal(first_scans(5), first_scans(5))
    assert not np.array_equal(first_scans(5), first_scans(6))


def test_the_radar_guided_interceptor_reads_its_station():
    """After a short flight the radar-guided interceptor's knowledge of its target
    is that station's track; the truth-guided one never read any track."""
    setup = build(spec())
    flight = setup.world.fly(0.01, 0.35)
    track = setup.stations["north"].tracks["a"]
    assert track.ready and track.updates == 4  # scans at 0, 0.1, 0.2 and 0.3 s
    assert flight.launched_at["n"] is None  # 900 m away: no alert yet


def test_a_radar_that_nobody_uses_changes_nothing():
    """radar-scramble case A is generic-scramble case A with a radar on the pad and
    the Iris still guided by the truth: the same flight to the last bit (the radar
    runs, but its tracks are not read)."""
    _, plain = play(load_scenario(ROOT / "scenarios/generic-scramble/a-slow.yaml"))
    setup, with_radar = play(
        load_scenario(ROOT / "scenarios/radar-scramble/a-slow-perfect.yaml")
    )
    assert np.array_equal(plain.flight.states, with_radar.flight.states)
    assert plain.results[0].closest == with_radar.results[0].closest
    # It did measure: every 0.1 s from 0 to 36.8 s (the capture is at 36.87 s).
    assert round(plain.results[0].end_time, 2) == 36.87
    assert len(setup.stations["pad"].measurements) == 369


def test_a_radar_guided_case_flies_the_same_way_twice():
    """The file fixes the seed, so two builds of radar-scramble case B fly the
    same 15 s (radar alert at 13.2 s and the first seconds of the chase)."""
    path = ROOT / "scenarios/radar-scramble/b-slow-radar.yaml"
    flights = [build(load_scenario(path)).world.fly(0.01, 15.0) for _ in range(2)]
    assert np.array_equal(flights[0].states, flights[1].states)
    assert flights[0].launched_at["iris"] is not None
