"""Tests for scenario files (jineta/arena/spec.py): reading, conversions, errors.

The errors matter as much as the happy path: the files are written by hand, so a
typo must stop the run with a message that says where it is and what was meant.
"""

import copy
import math

import pytest
import yaml

from jineta.arena.spec import ScenarioError, load_scenario, parse_scenario

FULL = """
name: "Two pads"
detail: "a test file"
duration: 60
step: 0.02
environment:
  atmosphere: sea_level
  wind: {speed: 5, from_deg: 270}
sites:
  - {id: north-pad, position: [100, 0], altitude: 10, alert_range: 400}
  - {id: south-pad, position: [-100, 0]}
interceptors:
  - id: iris-1
    vehicle: iris
    site: north-pad
    target: drone
    speed: 25
  - id: iris-2
    vehicle: iris
    start: {position: [0, 0], altitude: 100}
    target: dart
    speed: 16
    capture_radius: 1.0
    margin: 10
threats:
  - {id: dart, kind: straight, position: [-300, -250], altitude: 120, speed: 25,
     course_deg: 45}
  - {id: curl, kind: turning, position: [150, 60], altitude: 120, speed: 8,
     course_deg: 90, turn_rate_deg_s: 2}
  - id: drone
    kind: fixed_wing
    vehicle: small_fixed_wing
    position: [650, -350]
    altitude: 100
    speed: 18
    course_deg: 140
    orbit: {center: [60, 40], radius: 150, clockwise: false}
"""


def base():
    return yaml.safe_load(FULL)


def test_reads_every_kind_of_entry_in_si_and_ned():
    spec = parse_scenario(base())
    assert (spec.name, spec.detail, spec.duration, spec.step) == (
        "Two pads",
        "a test file",
        60.0,
        0.02,
    )
    assert spec.environment.atmosphere == "sea_level"
    assert (spec.environment.wind.speed, spec.environment.wind.from_deg) == (5, 270)

    north, south = spec.sites
    assert north.position == (100.0, 0.0, -10.0)  # altitude up -> down = -10
    assert north.alert_range == 400.0
    assert south.position == (-100.0, 0.0, 0.0) and south.alert_range is None

    first, second = spec.interceptors
    assert (first.site, first.start, first.target) == ("north-pad", None, "drone")
    assert (first.capture_radius, first.margin) == (0.5, None)  # defaults
    assert (second.site, second.start) == (None, (0.0, 0.0, -100.0))
    assert (second.capture_radius, second.margin) == (1.0, 10.0)

    dart, curl, drone = spec.threats
    assert dart.position == (-300.0, -250.0, -120.0)
    assert dart.course == pytest.approx(math.pi / 4)
    assert dart.turn_rate == 0.0
    assert curl.turn_rate == pytest.approx(math.radians(2.0))
    assert drone.vehicle == "small_fixed_wing"
    assert drone.orbit.center == (60.0, 40.0, 0.0)
    assert (drone.orbit.radius, drone.orbit.clockwise) == (150.0, False)


def test_defaults_of_a_minimal_file():
    spec = parse_scenario(
        {
            "name": "minimal",
            "duration": 10,
            "interceptors": [
                {
                    "id": "i",
                    "vehicle": "iris",
                    "start": {"position": [0, 0], "altitude": 50},
                    "target": "t",
                    "speed": 10,
                }
            ],
            "threats": [
                {
                    "id": "t",
                    "kind": "straight",
                    "position": [100, 0],
                    "altitude": 50,
                    "speed": 0,
                    "course_deg": 0,
                }
            ],
        }
    )
    assert spec.step == 0.01
    assert spec.detail == ""
    assert spec.environment.atmosphere == "isa"
    assert spec.environment.wind is None
    assert spec.sites == ()


def edit(path, value):
    """A change to the full file: set (or with value=DELETE, remove) one key."""

    def apply(data):
        *parents, last = path
        node = data
        for key in parents:
            node = node[key]
        if value is DELETE:
            del node[last]
        else:
            node[last] = value

    return apply


DELETE = object()


@pytest.mark.parametrize(
    "change, message",
    [
        (edit(["duraton"], 60), "unknown key 'duraton'; did you mean 'duration'?"),
        (edit(["name"], DELETE), "top level: missing key 'name'"),
        (
            edit(["interceptors", 0, "vehicle"], "irs"),
            "interceptors[0].vehicle: unknown vehicle 'irs'; did you mean 'iris'?",
        ),
        (
            edit(["interceptors", 0, "target"], "dron"),
            "interceptors[0]: unknown target 'dron'; did you mean 'drone'?",
        ),
        (edit(["interceptors", 0, "site"], "east-pad"), "unknown site 'east-pad'"),
        (
            edit(["interceptors", 1, "site"], "north-pad"),
            "interceptors[1]: needs either 'site'",
        ),
        (edit(["interceptors", 0, "site"], DELETE), "needs either 'site'"),
        (edit(["interceptors", 1, "id"], "iris-1"), "repeated interceptor/threat id"),
        (edit(["sites", 1, "id"], "north-pad"), "repeated site id"),
        (edit(["interceptors", 0, "speed"], -1), "interceptors[0].speed: must be >= 0"),
        (edit(["interceptors", 0, "speed"], True), "speed: must be a number"),
        (edit(["interceptors", 0, "margin"], 0), "margin: must be > 0"),
        (edit(["threats", 0, "kind"], "balloon"), "unknown kind 'balloon'"),
        (
            edit(["threats", 1, "turn_rate_deg_s"], DELETE),
            "threats[1]: missing key 'turn_rate_deg_s'",
        ),
        (edit(["threats", 0, "turn_rate_deg_s"], 2), "threats[0]: unknown key"),
        (edit(["threats", 0, "position"], [1, 2, 3]), "must be [north, east]"),
        (edit(["threats", 2, "speed"], 0), "threats[2].speed: must be > 0"),
        (edit(["threats", 2, "orbit", "radious"], 1), "did you mean 'radius'?"),
        (edit(["environment", "atmosphere"], "mars"), "unknown atmosphere 'mars'"),
        (edit(["environment", "wind", "from"], 270), "environment.wind: unknown key"),
        (edit(["interceptors"], []), "interceptors: needs at least one entry"),
        (edit(["threats"], {"id": "x"}), "threats: must be a list"),
        (edit(["duration"], 0.001), "shorter than one step"),
        (edit(["detail"], 5), "detail: must be a text"),
    ],
)
def test_every_mistake_says_where_and_what(change, message):
    data = copy.deepcopy(base())
    change(data)
    with pytest.raises(ScenarioError) as error:
        parse_scenario(data, "case.yaml")
    assert str(error.value).startswith("case.yaml: ")
    assert message in str(error.value)


def test_load_from_a_file_and_reject_broken_yaml(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text(FULL, encoding="utf-8")
    assert load_scenario(good).source == str(good)
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScenarioError, match="not valid YAML"):
        load_scenario(broken)
    with pytest.raises(ScenarioError, match="must be a mapping"):
        parse_scenario(["a", "list"])
