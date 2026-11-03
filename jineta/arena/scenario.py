"""From a scenario spec (`spec.py`) to a world ready to fly, and the fight itself.

Each spec entry becomes an entity of the world, in the order of the file
(interceptors first, then threats):
  - an interceptor: a multirotor with its autopilot and pure pursuit guidance;
    at rest on its site (waiting for the alert when the site has a dome) or
    hovering at its start point;
  - a threat: a scripted track (straight, turning) or a simulated fixed wing with
    its autopilot (trimmed for level flight; optional orbit guidance).
Every vehicle gets the scenario's atmosphere and wind. Each interceptor pursues the
target the file gives it; the pairs, rules and domes follow from the sites.

A site with a radar becomes a `RadarStation` that tracks every threat; the stations
come first among the rules, so the launch rules and the controls of a step use
that step's scan. Each radar gets its own seed, drawn from the scenario's seed
(`numpy.random.SeedSequence`): independent noise streams, and the same ones every
time the file is flown.
"""

import math
from dataclasses import dataclass

import numpy as np

from jineta.arena.engagement import Battle, Pairing, fight, pursuit
from jineta.arena.launch import Dome, LaunchOnAlert, LaunchOnTrack
from jineta.arena.sensing import Knowledge, RadarStation
from jineta.arena.spec import InterceptorSpec, ScenarioSpec, ThreatSpec
from jineta.arena.world import Control, Entity, TrackEntity, VehicleEntity, World
from jineta.atalaya.profiles import RADAR_PROFILES
from jineta.atalaya.radar import Radar
from jineta.campo.atmosphere import isa_density
from jineta.campo.wind import ConstantWind, meteorological_wind
from jineta.core.environment import DensityModel, WindModel, no_wind, sea_level_density
from jineta.jinete.control import Autopilot
from jineta.jinete.fixed_wing import FixedWing
from jineta.jinete.fixed_wing_control import FixedWingAutopilot, orbit_course
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import FIXED_WING_PROFILES, MULTIROTOR_PROFILES
from jineta.jinete.targets import StraightLineTarget, TurningTarget
from jineta.jinete.trim import trim_level_flight


@dataclass(frozen=True)
class Setup:
    """A scenario built and ready to fly."""

    spec: ScenarioSpec
    world: World
    pairings: tuple[Pairing, ...]
    stations: dict[str, RadarStation]  # by site id (sites with a radar)


def at_rest(position) -> np.ndarray:
    """13-element state: level, still, at a NED position (on a pad or hovering)."""
    return np.array([*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


def environment(spec: ScenarioSpec) -> tuple[DensityModel, WindModel]:
    """The density and wind models every vehicle of the scenario flies in."""
    density = isa_density if spec.environment.atmosphere == "isa" else sea_level_density
    wind = spec.environment.wind
    if wind is None or wind.speed == 0.0:
        return density, no_wind
    return density, ConstantWind(meteorological_wind(wind.speed, wind.from_deg))


def threat_entity(
    spec: ThreatSpec, density: DensityModel, wind: WindModel, dt: float
) -> Entity:
    if spec.kind == "straight":
        velocity = spec.speed * np.array([np.cos(spec.course), np.sin(spec.course), 0])
        return TrackEntity(spec.id, StraightLineTarget(spec.position, velocity))
    if spec.kind == "turning":
        track = TurningTarget(spec.position, spec.speed, spec.course, spec.turn_rate)
        return TrackEntity(spec.id, track)

    profile = FIXED_WING_PROFILES[spec.vehicle]
    aircraft = FixedWing(profile.params, profile.propulsion, density=density, wind=wind)
    altitude = -spec.position[2]
    trim = trim_level_flight(
        aircraft, spec.speed, altitude=altitude, course=spec.course
    )
    autopilot = FixedWingAutopilot(aircraft, trim, profile.autopilot, dt)
    start = np.array(trim.state)
    start[0:2] = spec.position[0:2]
    orbit = spec.orbit

    def control(t: float, state: np.ndarray, snapshot) -> None:
        if orbit is not None:
            autopilot.course_command = orbit_course(
                state[:3], np.array(orbit.center), orbit.radius, orbit.clockwise
            )
        autopilot(t, state)

    return VehicleEntity(spec.id, aircraft.derivative, start, control=control)


def interceptor_entity(
    spec: InterceptorSpec,
    start: tuple[float, float, float],
    waiting: bool,
    density: DensityModel,
    wind: WindModel,
    dt: float,
    knowledge: Knowledge | None = None,
) -> VehicleEntity:
    profile = MULTIROTOR_PROFILES[spec.vehicle]
    vehicle = Multirotor(profile.params, density=density, wind=wind)
    autopilot = Autopilot(vehicle, profile.gains, dt)
    control: Control = pursuit(autopilot, spec.target, spec.speed, knowledge)
    return VehicleEntity(
        spec.id, vehicle.derivative, at_rest(start), control=control, waiting=waiting
    )


def radar_seeds(seed: int, count: int) -> list[int]:
    """`count` independent seeds drawn from a scenario's seed, one per radar."""
    children = np.random.SeedSequence(seed).spawn(count)
    return [int(child.generate_state(1)[0]) for child in children]


def stations(spec: ScenarioSpec) -> dict[str, RadarStation]:
    """One radar station per site with a radar, tracking every threat (in the order
    of the file), each with its own seed."""
    sites = [site for site in spec.sites if site.radar is not None]
    targets = {threat.id: threat.tracking for threat in spec.threats}
    return {
        site.id: RadarStation(
            Radar(RADAR_PROFILES[site.radar], site.position, seed), targets
        )
        for site, seed in zip(sites, radar_seeds(spec.seed, len(sites)), strict=True)
    }


def build(spec: ScenarioSpec) -> Setup:
    """The world, rules and pairs of a scenario (fresh vehicles every time)."""
    density, wind = environment(spec)
    sites = {site.id: site for site in spec.sites}
    radar_stations = stations(spec)

    entities: list[Entity] = []
    rules: list = list(radar_stations.values())  # first: the others use their scans
    pairings = []
    for interceptor in spec.interceptors:
        site = sites.get(interceptor.site)
        dome = None
        if site is not None and site.alert_range is not None:
            dome = Dome(site.position, site.alert_range)
        start = site.position if site is not None else interceptor.start
        station = None
        if interceptor.knowledge == "radar":
            station = radar_stations[interceptor.site]
        entities.append(
            interceptor_entity(
                interceptor,
                start,
                dome is not None,
                density,
                wind,
                spec.step,
                None if station is None else station.knowledge,
            )
        )
        if dome is not None and station is not None:
            rules.append(
                LaunchOnTrack(interceptor.id, dome, station, [interceptor.target])
            )
        elif dome is not None:
            rules.append(LaunchOnAlert(interceptor.id, dome, [interceptor.target]))
        margin = math.inf if interceptor.margin is None else interceptor.margin
        pairings.append(
            Pairing(
                interceptor.id,
                interceptor.target,
                interceptor.capture_radius,
                margin,
                dome,
            )
        )
    for threat in spec.threats:
        entities.append(threat_entity(threat, density, wind, spec.step))

    return Setup(spec, World(entities, rules), tuple(pairings), radar_stations)


def play(spec: ScenarioSpec) -> tuple[Setup, Battle]:
    """Build the scenario and fight it to the end."""
    setup = build(spec)
    battle = fight(setup.world, setup.pairings, spec.step, spec.duration)
    return setup, battle
