"""Tests for building and flying a scenario from its spec (jineta/arena/scenario.py).

The engagement in wind (loose end 16). An Iris at 10 m/s chases a target flying
East at 4 m/s, 40 m away and 20 m above, in ISA air and a 5 m/s wind. The ideal
point pursuer of the closed form catches it at T = r0 (k + cos phi0) / (Vt (k^2 - 1))
(test_guidance.py); the real Iris, with its lag, between T and T + 3 s, as in still
air (test_engagement.py). The velocity loop holds the ground velocity whatever the
wind, so the wind only shows while the Iris accelerates: a headwind (chasing into
it) makes it later, a tailwind earlier.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from jineta.arena.engagement import CAPTURED, Pairing, fight, pursuit
from jineta.arena.launch import Dome, LaunchOnAlert
from jineta.arena.scenario import build, play
from jineta.arena.spec import load_scenario, parse_scenario
from jineta.arena.world import TrackEntity, VehicleEntity, World
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import TurningTarget

ROOT = Path(__file__).resolve().parents[1]


def scenario(text, **changes):
    data = yaml.safe_load(text)
    data.update(changes)
    return parse_scenario(data)


def test_a_scenario_file_flies_exactly_the_hand_built_world():
    """The file of the phase 7 scramble, case A (scenarios/generic-scramble), against
    the same world built by hand from the library pieces: identical to the bit."""
    _, from_file = play(load_scenario(ROOT / "scenarios/generic-scramble/a-slow.yaml"))

    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, 0.01)
    pad = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    threat = TurningTarget(
        [420.0, -250.0, -110.0], 8.0, math.radians(155.0), math.radians(-1.5)
    )
    dome = Dome([0.0, 0.0, 0.0], 400.0)
    world = World(
        [
            VehicleEntity(
                "iris",
                autopilot.vehicle.derivative,
                pad,
                control=pursuit(autopilot, "threat", 16.0),
                waiting=True,
            ),
            TrackEntity("threat", threat),
        ],
        rules=[LaunchOnAlert("iris", dome, ["threat"])],
    )
    pair = Pairing("iris", "threat", 0.5, math.inf, dome)
    by_hand = fight(world, [pair], 0.01, 120.0)

    assert from_file.results[0].outcome == CAPTURED
    assert np.array_equal(from_file.flight.states, by_hand.flight.states)
    assert from_file.results[0].closest == by_hand.results[0].closest
    assert from_file.flight.launched_at == by_hand.flight.launched_at


def test_build_turns_each_entry_into_its_entity_rule_and_pair():
    setup = build(
        scenario(
            """
name: build
duration: 5
sites:
  - {id: pad, position: [0, 0], alert_range: 300}
  - {id: field, position: [-50, 0]}
interceptors:
  - {id: a, vehicle: iris, site: pad, target: drone, speed: 20}
  - {id: b, vehicle: iris, site: field, target: dart, speed: 20, margin: 10}
  - {id: c, vehicle: iris, start: {position: [0, 0], altitude: 80}, target: dart,
     speed: 20}
threats:
  - {id: dart, kind: straight, position: [500, 0], altitude: 50, speed: 20,
     course_deg: 180}
  - {id: drone, kind: fixed_wing, vehicle: small_fixed_wing, position: [600, 100],
     altitude: 120, speed: 22, course_deg: 200}
"""
        )
    )
    world = setup.world
    assert [e.name for e in world.entities] == ["a", "b", "c", "dart", "drone"]
    a, b, c = (world.entity(name) for name in "abc")
    assert (a.waiting, b.waiting, c.waiting) == (True, False, False)
    assert b.initial_state[0] == -50.0  # on its field
    assert c.initial_state[2] == -80.0  # hovering at 80 m
    assert isinstance(world.entity("dart"), TrackEntity)
    drone = world.entity("drone").initial_state
    assert np.linalg.norm(drone[3:6]) == pytest.approx(22.0, rel=1e-3)  # trimmed
    assert np.allclose(drone[0:3], [600.0, 100.0, -120.0], rtol=0, atol=1e-6)
    # only the pad with a dome has an alert, and it watches every threat
    (rule,) = world.rules
    assert (rule.interceptor, list(rule.threats)) == ("a", ["dart", "drone"])
    pa, pb, pc = setup.pairings
    assert (pa.target, pa.margin, pa.dome.radius) == ("drone", math.inf, 300.0)
    assert (pb.margin, pb.dome) == (10.0, None)
    assert pc.dome is None


CHASE = """
name: wind
duration: 40
environment: {atmosphere: isa}
interceptors:
  - {id: iris, vehicle: iris, start: {position: [0, 0], altitude: 100}, target: t,
     speed: 10}
threats:
  - {id: t, kind: straight, position: [0, 40], altitude: 120, speed: 4,
     course_deg: 90}
"""


def capture_time(wind_from=None):
    environment = {"atmosphere": "isa"}
    if wind_from is not None:
        environment["wind"] = {"speed": 5, "from_deg": wind_from}
    _, battle = play(scenario(CHASE, environment=environment))
    result = battle.results[0]
    assert result.outcome == CAPTURED
    return result.closest.time


def test_an_engagement_in_isa_and_wind():
    r0 = np.array([0.0, 40.0, -20.0])
    k = 10.0 / 4.0
    ideal = np.linalg.norm(r0) * (k + r0[1] / np.linalg.norm(r0)) / (4.0 * (k**2 - 1))

    still = capture_time()
    headwind = capture_time(wind_from=90)  # from the East: it chases into it
    tailwind = capture_time(wind_from=270)
    for t in (still, headwind, tailwind):
        assert ideal < t < ideal + 3.0
    assert tailwind < still < headwind
    assert headwind - still > 0.5  # the wind does reach the vehicle
