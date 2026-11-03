"""Scenario files: YAML read into frozen, validated specs (SI units, NED).

A scenario file is data, written by hand. Units: m, m/s, s; angles in degrees, and
only in the file (the key says so: `_deg`, `_deg_s`). Positions are written as on a
map, `[north, east]` plus an `altitude` (up); they become NED here (down = -altitude).

    name: "A · slow threat"
    detail: "8 m/s, turning 1.5 deg/s"     # optional
    duration: 120                           # longest run [s]
    step: 0.01                              # optional, default 0.01 [s]
    environment:                            # optional
      atmosphere: sea_level                 # isa (default) or sea_level
      wind: {speed: 5, from_deg: 270}       # optional, uniform; FROM where
    sites:                                  # optional
      - id: pad
        position: [0, 0]
        altitude: 0                         # optional, default 0
        alert_range: 400                    # optional: the alert dome
    interceptors:
      - id: iris
        vehicle: iris                       # a name in MULTIROTOR_PROFILES
        site: pad                           # at rest on the pad, or instead
        # start: {position: [0, 0], altitude: 100}   # hovering, flying at t = 0
        target: threat
        speed: 16                           # ordered pursuit speed
        capture_radius: 0.5                 # optional, default 0.5
        margin: 10                          # optional: a pass ends the pair
    threats:
      - id: threat
        kind: turning                       # straight, turning or fixed_wing
        position: [420, -250]
        altitude: 110
        speed: 8
        course_deg: 155                     # clockwise from North
        turn_rate_deg_s: -1.5               # turning only; > 0 turns right
      - id: drone
        kind: fixed_wing
        vehicle: small_fixed_wing           # a name in FIXED_WING_PROFILES
        position: [650, -350]
        altitude: 100
        speed: 18                           # trimmed airspeed
        course_deg: 140
        orbit: {center: [60, 40], radius: 150, clockwise: true}   # optional

An interceptor on a site with an `alert_range` waits until a threat comes within
that range (any threat), and its pair ends if its target leaves the dome; on a site
without one it takes off at t = 0. Without `margin`, a pass never ends a pair.

Every mistake is an error that says where it is: an unknown or missing key (with a
suggestion when the name is close to a valid one), a wrong type, a value out of
range, an unknown vehicle, site or target, a repeated id.
"""

import difflib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from jineta.jinete.profiles import FIXED_WING_PROFILES, MULTIROTOR_PROFILES

Vector = tuple[float, float, float]  # NED [m]
THREAT_KINDS = ("straight", "turning", "fixed_wing")
ATMOSPHERES = ("isa", "sea_level")


class ScenarioError(ValueError):
    """A scenario file that cannot be used, with where and why."""


# --- Specs -------------------------------------------------------------------------


@dataclass(frozen=True)
class WindSpec:
    speed: float  # [m/s]
    from_deg: float  # where it comes FROM, clockwise from North [deg]


@dataclass(frozen=True)
class EnvironmentSpec:
    atmosphere: str = "isa"  # "isa" or "sea_level"
    wind: WindSpec | None = None


@dataclass(frozen=True)
class SiteSpec:
    id: str
    position: Vector  # NED [m]
    alert_range: float | None  # radius of the alert dome [m]


@dataclass(frozen=True)
class InterceptorSpec:
    id: str
    vehicle: str  # key of MULTIROTOR_PROFILES
    site: str | None  # starts at rest on this site, or
    start: Vector | None  # starts hovering here (NED [m]), flying from t = 0
    target: str
    speed: float  # ordered pursuit speed [m/s]
    capture_radius: float  # [m]
    margin: float | None  # None: a pass never ends the pair [m]


@dataclass(frozen=True)
class OrbitSpec:
    center: Vector  # NED [m]; only North and East matter
    radius: float  # [m]
    clockwise: bool


@dataclass(frozen=True)
class ThreatSpec:
    id: str
    kind: str  # "straight", "turning" or "fixed_wing"
    position: Vector  # NED [m] at t = 0
    speed: float  # [m/s]
    course: float  # direction of travel, clockwise from North [rad]
    turn_rate: float = 0.0  # turning only [rad/s]
    vehicle: str | None = None  # fixed_wing only: key of FIXED_WING_PROFILES
    orbit: OrbitSpec | None = None  # fixed_wing only


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    detail: str
    duration: float  # [s]
    step: float  # [s]
    environment: EnvironmentSpec
    sites: tuple[SiteSpec, ...]
    interceptors: tuple[InterceptorSpec, ...]
    threats: tuple[ThreatSpec, ...]
    source: str  # where it came from (a file path, or "<text>")


# --- Reading helpers ------------------------------------------------------------------


class _Reader:
    """Reads one mapping of the file, remembering where it is for the errors."""

    def __init__(self, data: object, where: str, source: str) -> None:
        self.where, self.source = where, source
        if not isinstance(data, Mapping):
            self.fail(f"must be a mapping of keys (key: value), not {data!r}")
        self.data = data
        self.used: set[str] = set()

    def fail(self, message: str, key: str | None = None):
        place = self.where if key is None else f"{self.where}.{key}".lstrip(".")
        raise ScenarioError(f"{self.source}: {place or 'top level'}: {message}")

    def get(self, key: str, default: object = ..., kind: str = "any") -> object:
        self.used.add(key)
        if key not in self.data:
            if default is ...:
                self.fail(f"missing key '{key}'")
            return default
        value = self.data[key]
        if kind == "number":
            # bool is an int in Python (and YAML reads yes/no as booleans)
            if isinstance(value, bool) or not isinstance(value, int | float):
                self.fail(f"must be a number, not {value!r}", key)
            return float(value)
        if kind == "text":
            if not isinstance(value, str) or not value:
                self.fail(f"must be a non-empty text, not {value!r}", key)
        if kind == "bool" and not isinstance(value, bool):
            self.fail(f"must be true or false, not {value!r}", key)
        if kind == "list" and not isinstance(value, list):
            self.fail(f"must be a list (lines starting with '-'), not {value!r}", key)
        return value

    def number(self, key, default=..., low=None, positive=False):
        value = self.get(key, default, "number")
        if value is None or value is default:
            return value
        if positive and not value > 0.0:
            self.fail(f"must be > 0, not {value:g}", key)
        if low is not None and value < low:
            self.fail(f"must be >= {low:g}, not {value:g}", key)
        return value

    def text(self, key, default=...):
        return self.get(key, default, "text")

    def choice(self, key, options, default=...):
        value = self.text(key, default)
        if value not in options:
            self.fail(f"unknown {key} '{value}'{_suggest(value, options)}", key)
        return value

    def map_position(self, key="position", altitude_key="altitude", altitude=...):
        """[north, east] + altitude -> NED."""
        value = self.get(key, kind="list")
        if len(value) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int | float) for v in value
        ):
            self.fail(f"must be [north, east] in m, not {value!r}", key)
        up = self.number(altitude_key, altitude)
        return (float(value[0]), float(value[1]), -up)

    def done(self) -> None:
        """Every key of the mapping must have been read: anything else is a typo or
        something this format does not have."""
        for key in self.data:
            if key not in self.used:
                options = sorted(self.used)
                self.fail(f"unknown key '{key}'{_suggest(str(key), options)}")


def _suggest(word: str, options) -> str:
    close = difflib.get_close_matches(word, list(options), n=1)
    hint = f"; did you mean '{close[0]}'?" if close else ""
    return f"{hint} (valid: {', '.join(sorted(options))})"


def _items(reader: _Reader, key: str, required: bool) -> list[_Reader]:
    values = reader.get(key, [] if not required else ..., "list")
    if required and not values:
        reader.fail("needs at least one entry", key)
    return [
        _Reader(value, f"{key}[{i}]", reader.source) for i, value in enumerate(values)
    ]


# --- Parsing -----------------------------------------------------------------------


def _environment(top: _Reader) -> EnvironmentSpec:
    if "environment" not in top.data:
        top.used.add("environment")
        return EnvironmentSpec()
    r = _Reader(top.get("environment"), "environment", top.source)
    atmosphere = r.choice("atmosphere", ATMOSPHERES, "isa")
    wind = None
    if "wind" in r.data:
        w = _Reader(r.get("wind"), "environment.wind", r.source)
        wind = WindSpec(w.number("speed", low=0.0), w.number("from_deg"))
        w.done()
    else:
        r.used.add("wind")
    r.done()
    return EnvironmentSpec(atmosphere, wind)


def _site(r: _Reader) -> SiteSpec:
    spec = SiteSpec(
        id=r.text("id"),
        position=r.map_position(altitude=0.0),
        alert_range=r.number("alert_range", None, positive=True),
    )
    r.done()
    return spec


def _interceptor(r: _Reader) -> InterceptorSpec:
    site = r.text("site", None)
    start = None
    if "start" in r.data:
        s = _Reader(r.get("start"), f"{r.where}.start", r.source)
        start = s.map_position()
        s.done()
    else:
        r.used.add("start")
    if (site is None) == (start is None):
        r.fail("needs either 'site' (at rest on a pad) or 'start' (hovering)")
    spec = InterceptorSpec(
        id=r.text("id"),
        vehicle=r.choice("vehicle", MULTIROTOR_PROFILES),
        site=site,
        start=start,
        target=r.text("target"),
        speed=r.number("speed", low=0.0),
        capture_radius=r.number("capture_radius", 0.5, positive=True),
        margin=r.number("margin", None, positive=True),
    )
    r.done()
    return spec


def _threat(r: _Reader) -> ThreatSpec:
    kind = r.choice("kind", THREAT_KINDS)
    common = dict(
        id=r.text("id"),
        kind=kind,
        position=r.map_position(),
        speed=r.number("speed", positive=kind == "fixed_wing", low=0.0),
        course=math.radians(r.number("course_deg")),
    )
    if kind == "turning":
        common["turn_rate"] = math.radians(r.number("turn_rate_deg_s"))
    if kind == "fixed_wing":
        common["vehicle"] = r.choice("vehicle", FIXED_WING_PROFILES)
        if "orbit" in r.data:
            o = _Reader(r.get("orbit"), f"{r.where}.orbit", r.source)
            common["orbit"] = OrbitSpec(
                center=o.map_position("center", altitude=0.0),
                radius=o.number("radius", positive=True),
                clockwise=o.get("clockwise", True, "bool"),
            )
            o.done()
        else:
            r.used.add("orbit")
    r.done()
    return ThreatSpec(**common)


def parse_scenario(data: object, source: str = "<text>") -> ScenarioSpec:
    """A scenario from already-parsed YAML (dicts and lists), validated."""
    top = _Reader(data, "", source)
    name = top.text("name")
    detail = top.get("detail", "")
    if not isinstance(detail, str):
        top.fail(f"must be a text, not {detail!r}", "detail")
    duration = top.number("duration", positive=True)
    step = top.number("step", 0.01, positive=True)
    environment = _environment(top)
    sites = tuple(_site(r) for r in _items(top, "sites", required=False))
    interceptors = tuple(_interceptor(r) for r in _items(top, "interceptors", True))
    threats = tuple(_threat(r) for r in _items(top, "threats", True))
    top.done()

    # Cross-references: unique ids, and every name points at something that exists.
    site_ids = [s.id for s in sites]
    entity_ids = [e.id for e in interceptors] + [t.id for t in threats]
    for label, ids in (("site", site_ids), ("interceptor/threat", entity_ids)):
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            top.fail(f"repeated {label} id {repeated}")
    threat_ids = [t.id for t in threats]
    for i, spec in enumerate(interceptors):
        where = f"interceptors[{i}]"
        if spec.site is not None and spec.site not in site_ids:
            top.fail(
                f"unknown site '{spec.site}'{_suggest(spec.site, site_ids)}", where
            )
        if spec.target not in threat_ids:
            top.fail(
                f"unknown target '{spec.target}'{_suggest(spec.target, threat_ids)}",
                where,
            )
    if not duration >= step:
        top.fail(f"duration ({duration:g} s) is shorter than one step ({step:g} s)")

    return ScenarioSpec(
        name=name,
        detail=detail,
        duration=duration,
        step=step,
        environment=environment,
        sites=sites,
        interceptors=interceptors,
        threats=threats,
        source=source,
    )


def load_scenario(path: Path | str) -> ScenarioSpec:
    """Read and validate one scenario file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ScenarioError(f"{path}: not valid YAML: {error}") from error
    return parse_scenario(data, str(path))
