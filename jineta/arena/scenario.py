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
"""

import math
from dataclasses import dataclass

import numpy as np

from jineta.arena.engagement import Battle, Pairing, fight, pursuit
from jineta.arena.launch import Dome, LaunchOnAlert
from jineta.arena.spec import InterceptorSpec, ScenarioSpec, ThreatSpec
from jineta.arena.world import Control, Entity, TrackEntity, VehicleEntity, World
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
) -> VehicleEntity:
    profile = MULTIROTOR_PROFILES[spec.vehicle]
    vehicle = Multirotor(profile.params, density=density, wind=wind)
    autopilot = Autopilot(vehicle, profile.gains, dt)
    control: Control = pursuit(autopilot, spec.target, spec.speed)
    return VehicleEntity(
        spec.id, vehicle.derivative, at_rest(start), control=control, waiting=waiting
    )


def build(spec: ScenarioSpec) -> Setup:
    """The world, rules and pairs of a scenario (fresh vehicles every time)."""
    density, wind = environment(spec)
    sites = {site.id: site for site in spec.sites}
    threat_ids = [threat.id for threat in spec.threats]

    entities: list[Entity] = []
    rules = []
    pairings = []
    for interceptor in spec.interceptors:
        site = sites.get(interceptor.site)
        dome = None
        if site is not None and site.alert_range is not None:
            dome = Dome(site.position, site.alert_range)
        start = site.position if site is not None else interceptor.start
        entities.append(
            interceptor_entity(
                interceptor, start, dome is not None, density, wind, spec.step
            )
        )
        if dome is not None:
            rules.append(LaunchOnAlert(interceptor.id, dome, threat_ids))
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

    return Setup(spec, World(entities, rules), tuple(pairings))


def play(spec: ScenarioSpec) -> tuple[Setup, Battle]:
    """Build the scenario and fight it to the end."""
    setup = build(spec)
    battle = fight(setup.world, setup.pairings, spec.step, spec.duration)
    return setup, battle
