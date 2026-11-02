"""Tests for the world loop (jineta/arena/world.py): many entities, one 'run'.

Cyclic pursuit, the analytic check. N pursuers start on the corners of a regular
polygon of side a, and each one flies at speed v straight at the next one. By
symmetry the polygon stays regular while it turns and shrinks. Along one side, the
pursuer closes at v and its target runs away at v cos(theta), with
theta = 2 pi / N the turn from one side to the next, so the side shrinks at a
constant rate:
    d(t) = a - v (1 - cos(theta)) t,   they meet at T = a / (v (1 - cos(theta))).
(N = 4 is the classic "four mice in a square": T = a / v.)

In the simulation every pursuer reads where the others are at the start of the step
and flies straight for dt (zero-order hold). One step of length h = v dt then takes a
side d to exactly
    d' = sqrt((d - h)^2 + h^2 + 2 (d - h) h cos(theta)),
which the world must reproduce to rounding error. It only holds if every control
reads the same snapshot: a pursuer that saw its target already moved would break the
symmetry. The continuous law is the limit dt -> 0; to first order the discrete side
runs ahead of it by about h sin^2(theta) / (2 (1 - cos(theta))) ln(a / d), which is
0.1-0.4 m at d = 1 m here, so the time to shrink to 1 m is late by under 0.4 %.
"""

import numpy as np
import pytest

from jineta.arena.world import PointEntity, TrackEntity, VehicleEntity, World
from jineta.campo.atmosphere import isa_density
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import RigidBody, normalize_attitude
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
from jineta.jinete.targets import StraightLineTarget, track_positions
from jineta.jinete.trim import trim_level_flight

DT = 0.01
AT_REST = np.array([0, 0, -100.0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])


# --- Independent bodies --------------------------------------------------------------


def iris_climbing():
    """An Iris under its autopilot, ordered to 6 m/s North, 2 West, 1 up."""
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    autopilot.velocity_command = np.array([6.0, -2.0, -1.0])
    return autopilot, AT_REST


def fixed_wing_turning():
    """The fixed wing trimmed at 25 m/s heading North, ordered to turn East."""
    aircraft = FixedWing(
        SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION, density=isa_density
    )
    trim = trim_level_flight(aircraft, 25.0, altitude=150.0, course=0.0)
    autopilot = FixedWingAutopilot(aircraft, trim, SMALL_FIXED_WING_AUTOPILOT, DT)
    autopilot.course_command = np.radians(90.0)
    return autopilot, np.array(trim.state)


def test_bodies_flown_together_are_bit_for_bit_the_same_as_flown_alone():
    """Stacking independent bodies changes nothing: RK4 works component by
    component, so each slice sees exactly the arithmetic it would see alone."""
    alone = {}
    for name, make in (("iris", iris_climbing), ("wing", fixed_wing_turning)):
        autopilot, start = make()
        alone[name] = run(
            autopilot.vehicle.derivative,
            start,
            DT,
            5.0,
            pre_step=autopilot,
            post_step=normalize_attitude,
        )[1]

    iris, iris_start = iris_climbing()
    wing, wing_start = fixed_wing_turning()
    world = World(
        [
            VehicleEntity(
                "iris",
                iris.vehicle.derivative,
                iris_start,
                control=lambda t, state, snapshot: iris(t, state),
            ),
            TrackEntity("line", StraightLineTarget([0, 50, -60.0], [3.0, 0, 0])),
            VehicleEntity(
                "wing",
                wing.vehicle.derivative,
                wing_start,
                control=lambda t, state, snapshot: wing(t, state),
            ),
        ]
    )
    flight = world.fly(DT, 5.0)
    assert np.array_equal(flight.state("iris"), alone["iris"])
    assert np.array_equal(flight.state("wing"), alone["wing"])


def test_a_track_is_where_its_track_says_and_a_point_flies_its_command():
    line = StraightLineTarget([0, 50, -60.0], [3.0, 0, 0])
    world = World(
        [
            TrackEntity("line", line),
            PointEntity("dot", [10.0, 10.0, -10.0], velocity_command=[1.0, 2.0, 3.0]),
        ]
    )
    flight = world.fly(DT, 2.0)
    assert np.array_equal(flight.positions("line"), track_positions(line, flight.times))
    expected = np.array([10.0, 10.0, -10.0]) + np.outer(flight.times, [1.0, 2.0, 3.0])
    assert np.allclose(flight.positions("dot"), expected, rtol=0, atol=1e-9)
    assert flight.state("line").shape == (len(flight.times), 0)  # no state


# --- Cyclic pursuit ------------------------------------------------------------------


def cyclic_pursuit(n, side=100.0, speed=10.0, duration=None):
    """N ideal points on a regular polygon, each chasing the next. Returns the
    flight, the corners and the closed-form meeting time."""
    theta = 2 * np.pi / n
    meet = side / (speed * (1 - np.cos(theta)))
    radius = side / (2 * np.sin(np.pi / n))
    angles = theta * np.arange(n)
    center = np.array([50.0, -30.0, -80.0])
    corners = center + radius * np.stack(
        [np.cos(angles), np.sin(angles), np.zeros(n)], axis=1
    )
    agents = [PointEntity(f"p{i}", corner) for i, corner in enumerate(corners)]

    def chase(agent, target):
        def control(t, position, snapshot):
            agent.velocity_command = pure_pursuit_velocity(
                position, snapshot.position(target), speed
            )

        return control

    for i, agent in enumerate(agents):
        agent.control = chase(agent, f"p{(i + 1) % n}")
    flight = World(agents).fly(DT, duration if duration is not None else 0.995 * meet)
    return flight, corners, meet


def sides_of(flight, n):
    """(n, samples) length of every side of the polygon."""
    positions = [flight.positions(f"p{i}") for i in range(n)]
    return np.array(
        [
            np.linalg.norm(positions[(i + 1) % n] - positions[i], axis=1)
            for i in range(n)
        ]
    )


@pytest.mark.parametrize("n", [3, 4, 6])
def test_cyclic_pursuit_follows_the_exact_discrete_law(n):
    side, speed = 100.0, 10.0
    flight, corners, _ = cyclic_pursuit(n, side, speed)
    sides = sides_of(flight, n)

    h, theta = speed * DT, 2 * np.pi / n
    expected = [side]
    for _ in range(len(flight.times) - 1):
        d = expected[-1]
        expected.append(np.sqrt((d - h) ** 2 + h**2 + 2 * (d - h) * h * np.cos(theta)))
    assert np.allclose(sides[0], expected, rtol=0, atol=1e-9)
    # still a regular polygon, around the same centre
    assert np.allclose(sides, sides[0], rtol=0, atol=1e-9)
    centroid = np.mean([flight.positions(f"p{i}") for i in range(n)], axis=0)
    assert np.allclose(centroid, corners.mean(axis=0), rtol=0, atol=1e-9)


@pytest.mark.parametrize("n", [3, 4, 6])
def test_cyclic_pursuit_meets_at_the_closed_form_time(n):
    side, speed = 100.0, 10.0
    flight, _, meet = cyclic_pursuit(n, side, speed)
    gap = sides_of(flight, n)[0]
    k = int(np.flatnonzero(gap < 1.0)[0])  # first sample closer than 1 m
    t_1m = np.interp(1.0, gap[[k, k - 1]], flight.times[[k, k - 1]])
    expected = (side - 1.0) / side * meet  # d(t) = 1 m on the continuous law
    assert t_1m == pytest.approx(expected, rel=0.01)
    assert t_1m >= expected  # the discrete side runs ahead of the continuous one


# --- Waiting, rules, snapshots -------------------------------------------------------


def test_a_waiting_entity_does_not_move_until_a_rule_launches_it():
    """A bare rigid body (gravity only) waits for 1 s, then falls from rest:
    z = z0 + g (t - t_launch)^2 / 2, exact for RK4 (a quadratic)."""
    body = RigidBody(mass=2.0, inertia=np.diag([0.1, 0.2, 0.3]))
    calls = []
    ball = VehicleEntity(
        "ball",
        body.derivative,
        AT_REST,
        control=lambda t, state, snapshot: calls.append(t),
        waiting=True,
    )
    launched = []

    def launch_at_one_second(snapshot):
        # Sample times accumulate rounding error (0.01 + 0.01 + ...), hence 1e-9.
        if not ball.active and snapshot.time >= 1.0 - 1e-9:
            ball.active = True
            launched.append(snapshot.time)

    flight = World([ball], rules=[launch_at_one_second]).fly(DT, 2.0)
    t_launch = launched[0]
    assert t_launch == pytest.approx(1.0)
    assert calls[0] == t_launch  # its control only runs once it is active

    states = flight.state("ball")
    waiting = flight.times <= t_launch
    assert np.array_equal(states[waiting], np.tile(AT_REST, (waiting.sum(), 1)))
    falling = ~waiting
    expected_z = -100.0 + 0.5 * GRAVITY * (flight.times[falling] - t_launch) ** 2
    assert np.allclose(states[falling, 2], expected_z, rtol=0, atol=1e-9)


def test_flying_again_starts_from_the_same_status():
    body = RigidBody(mass=1.0, inertia=np.eye(3))
    ball = VehicleEntity("ball", body.derivative, AT_REST, waiting=True)

    def launch(snapshot):
        ball.active = True

    world = World([ball], rules=[launch])
    first = world.fly(DT, 1.0)
    world.rules = ()  # no rule this time: it must wait the whole flight
    second = world.fly(DT, 1.0)
    assert first.state("ball")[-1, 2] > AT_REST[2]  # it fell
    assert np.array_equal(second.state("ball")[-1], AT_REST)  # it waited


def test_controls_cannot_change_the_world_they_read():
    def vandal(t, position, snapshot):
        position[0] = 1e6  # tries to move itself by hand

    world = World([PointEntity("dot", [0.0, 0.0, 0.0], control=vandal)])
    with pytest.raises(ValueError):  # numpy: "assignment destination is read-only"
        world.fly(DT, 0.1)


def test_snapshot_reads_every_kind_of_entity():
    line = StraightLineTarget([1.0, 2.0, -3.0], [4.0, 0.0, 0.0])
    world = World(
        [
            VehicleEntity("body", RigidBody(1.0, np.eye(3)).derivative, AT_REST),
            TrackEntity("line", line),
            PointEntity("dot", [7.0, 8.0, 9.0], velocity_command=[0.0, 1.0, 0.0]),
        ]
    )
    snapshot = world.snapshot(2.0, world.initial_state())
    assert np.array_equal(snapshot.position("body"), AT_REST[0:3])
    assert np.array_equal(snapshot.position("line"), [9.0, 2.0, -3.0])
    assert np.array_equal(snapshot.velocity("line"), [4.0, 0.0, 0.0])
    assert np.array_equal(snapshot.position("dot"), [7.0, 8.0, 9.0])
    assert np.array_equal(snapshot.velocity("dot"), [0.0, 1.0, 0.0])
    assert world.size == 13 + 0 + 3


def test_rejects_invalid_worlds():
    body = RigidBody(1.0, np.eye(3)).derivative
    with pytest.raises(ValueError):
        World([])  # empty
    with pytest.raises(ValueError):
        World([PointEntity("a", [0, 0, 0]), PointEntity("a", [1, 1, 1])])  # repeated
    with pytest.raises(ValueError):
        PointEntity("", [0, 0, 0])  # no name
    with pytest.raises(ValueError):
        PointEntity("a", [0, 0])  # 2 components
    with pytest.raises(ValueError):
        VehicleEntity("a", body, np.zeros(12))  # 12 elements
    with pytest.raises(ValueError):
        VehicleEntity("a", body, np.zeros(13))  # zero quaternion
    with pytest.raises(KeyError):
        World([PointEntity("a", [0, 0, 0])]).slice_of("b")  # unknown name


def test_a_destroyed_entity_stays_where_it_was_hit_until_the_next_flight():
    """A point and a scripted track, destroyed by a rule at 1 s: from then on both
    stay where they were (the track stops on its path), with zero velocity. Flying
    again starts them afresh."""
    line = StraightLineTarget([0.0, 0.0, -50.0], [4.0, 0.0, 0.0])
    world = World(
        [
            PointEntity("dot", [0.0, 0.0, -10.0], velocity_command=[0.0, 2.0, 0.0]),
            TrackEntity("line", line),
        ]
    )
    hit = []

    def shoot_down(snapshot):
        if not hit and snapshot.time >= 1.0 - 1e-9:
            hit.append(snapshot.time)
            world.destroy("dot", snapshot.time)
            world.destroy("line", snapshot.time)

    world.rules = (shoot_down,)
    flight = world.fly(DT, 2.0)
    after = flight.times >= hit[0]
    for name, speed in (("dot", [0.0, 2.0, 0.0]), ("line", [4.0, 0.0, 0.0])):
        positions = flight.positions(name)
        assert np.array_equal(
            positions[after], np.tile(positions[after][0], (after.sum(), 1))
        )
        assert np.allclose(positions[after][0], positions[0] + hit[0] * np.array(speed))
        assert world.entity(name).destroyed
    snapshot = world.snapshot(1.5, flight.states[-1])
    assert np.array_equal(snapshot.velocity("line"), np.zeros(3))
    assert np.array_equal(snapshot.velocity("dot"), np.zeros(3))

    line_hit = flight.positions("line")[-1]
    world.rules = ()
    again = world.fly(DT, 2.0)
    assert not world.entity("line").destroyed
    assert np.array_equal(again.positions("line"), track_positions(line, again.times))
    assert np.allclose(again.positions("dot")[-1], [0.0, 4.0, -10.0])
    assert np.array_equal(flight.positions("line")[-1], line_hit)  # first one kept
