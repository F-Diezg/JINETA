"""Tests for collisions in flight: the vehicle's spheres against walls, ceiling and
obstacles, its landing gear against the floor once it has taken off.

Where the motion is known in closed form the moment of contact is checked exactly:
flying straight at a flat face the gap falls linearly, so the moment is exact; in
free fall onto the floor it is quadratic, and the interpolation error is below
g dt^2 / (8 v). What the samples alone miss (a thin wall jumped, a graze between
two samples) is measured against its formulas on straight sampled paths, and then
the swept check is shown to find every one of those contacts.

Most tests use the Iris's bounding sphere (one sphere at the centre of mass, the
same reach every way), so the expected moments are simple; the last section flies
the Iris's own 13 spheres, whose reach depends on the attitude.
"""

import numpy as np
import pytest

from jineta.campo import collision
from jineta.campo.collision import (
    SWEEP_TOLERANCE,
    TAKEOFF_CLEARANCE,
    CollisionMonitor,
)
from jineta.campo.indoor import Box, Cylinder, Room
from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.collision import BatchCollisionMonitor
from jineta.jinete.contact import GroundContact
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_BOUNDING_SPHERE,
    IRIS_GAINS,
    IRIS_HITBOX,
    IRIS_LANDING_GEAR,
)

DT = 0.01
SPHERE = IRIS_BOUNDING_SPHERE
R = SPHERE.radii[0]  # 0.384 m
ROOM = Room(size=[10.0, 8.0, 3.0])


def iris_monitor(obstacles=(), room=ROOM, centers=None, radii=None, hitbox=SPHERE):
    return CollisionMonitor(
        room,
        obstacles,
        hitbox.centers if centers is None else centers,
        hitbox.radii if radii is None else radii,
        IRIS_LANDING_GEAR.points,
    )


def state_at(position, velocity=(0.0, 0.0, 0.0), attitude=(1.0, 0.0, 0.0, 0.0)):
    y = np.zeros(13)
    y[0:3] = position
    y[3:6] = velocity
    y[6:10] = attitude
    return y


def straight_line(t, y):
    """Constant velocity, no rotation: the position moves, nothing else."""
    dy = np.zeros(13)
    dy[0:3] = y[3:6]
    return dy


def free_fall(t, y):
    dy = straight_line(t, y)
    dy[5] = GRAVITY
    return dy


# ---- walls, ceiling and obstacles, flying straight --------------------------------


def test_flying_into_a_wall_is_found_at_the_exact_moment():
    monitor = iris_monitor()
    start = state_at([5.0, 4.0, -1.5], velocity=[2.0, 0.0, 0.0])
    times, states = run(straight_line, start, DT, 5.0, stop=monitor)
    hit = monitor.collision
    assert hit.kind == "room" and hit.index is None
    # The sphere reaches the north wall (x = 10) when x = 10 - R.
    assert hit.time == pytest.approx((5.0 - R) / 2.0, abs=1e-9)
    np.testing.assert_allclose(hit.position, [10.0 - R, 4.0, -1.5], atol=1e-9)
    # The run stops at the first sample past the contact.
    assert times[-2] < hit.time <= times[-1]
    assert monitor.margins(states[-1])[0] < 0.0 <= monitor.margins(states[-2])[0]


def test_climbing_into_the_ceiling():
    monitor = iris_monitor()
    start = state_at([5.0, 4.0, -1.0], velocity=[0.0, 0.0, -1.0])
    run(straight_line, start, DT, 5.0, stop=monitor)
    assert monitor.collision.kind == "room"
    # The ceiling is 3 m up: the sphere touches it at an altitude of 3 - R.
    assert monitor.collision.time == pytest.approx(2.0 - R, abs=1e-9)


def test_the_obstacle_hit_is_named_by_its_index():
    pillar = Cylinder(base=[5.0, 6.0, 0.0], radius=0.25, height=3.0)
    crate = Box(center=[8.0, 4.0, -1.0], size=[1.0, 1.0, 2.0])
    monitor = iris_monitor([pillar, crate])
    run(straight_line, state_at([2.0, 4.0, -1.0], [3.0, 0.0, 0.0]), DT, 5.0, monitor)
    hit = monitor.collision
    assert (hit.kind, hit.index) == ("obstacle", 1)
    # The crate's south face is at x = 7.5: contact at x = 7.5 - R.
    assert hit.time == pytest.approx((5.5 - R) / 3.0, abs=1e-9)


def test_flying_at_a_pillar_through_its_axis():
    pillar = Cylinder(base=[6.0, 4.0, 0.0], radius=0.25, height=3.0)
    monitor = iris_monitor([pillar])
    run(straight_line, state_at([2.0, 4.0, -1.0], [2.0, 0.0, 0.0]), DT, 5.0, monitor)
    assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
    assert monitor.collision.time == pytest.approx((4.0 - 0.25 - R) / 2.0, abs=1e-9)


def test_spheres_follow_the_attitude():
    """A sphere 1 m ahead of the nose, the vehicle headed East (yaw 90 deg) and
    flying East: the sphere is 1 m East of the centre of mass. Placed with R^T
    instead of R it would be 1 m West, and the contact 2/v later."""
    nose = CollisionMonitor(ROOM, (), [[1.0, 0.0, 0.0]], [0.1], [[0.0, 0.0, 0.1]])
    east = quaternion.from_euler(0.0, 0.0, np.pi / 2)
    start = state_at([5.0, 2.0, -1.5], [0.0, 2.0, 0.0], attitude=east)
    run(straight_line, start, DT, 5.0, stop=nose)
    # The east wall is y = 8: contact when y + 1 + 0.1 = 8.
    assert nose.collision.time == pytest.approx((8.0 - 1.1 - 2.0) / 2.0, abs=1e-9)


def test_the_earliest_contact_inside_the_step_wins():
    """Two faces reached inside the same step (0.50 to 0.51 s): the one touched
    first is reported, although the room comes first in the list of checks."""
    monitor = iris_monitor([Box(center=[9.85, 4.0, -1.5], size=[0.1, 8.0, 3.0])])
    # Going North-East: the obstacle face x = 9.8 at 0.503 s, the east wall (y = 8)
    # at 0.507 s.
    start = state_at([9.8 - R - 0.503, 8.0 - R - 0.507, -1.5], [1.0, 1.0, 0.0])
    run(straight_line, start, DT, 2.0, stop=monitor)
    assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
    assert monitor.collision.time == pytest.approx(0.503, abs=1e-9)


def test_starting_inside_a_wall_is_reported_at_the_first_sample():
    monitor = iris_monitor()
    run(straight_line, state_at([0.1, 4.0, -1.5]), DT, 1.0, stop=monitor)
    assert monitor.collision.kind == "room"
    assert monitor.collision.time == pytest.approx(DT)


# ---- the floor ----------------------------------------------------------------------


def test_free_fall_onto_the_floor_after_leaving_it():
    """Released level 1 m up: the gear (0.055 m below the centre of mass) reaches
    the floor after sqrt(2 * 0.945 / g). The margin is quadratic in time, so the
    linear interpolation errs by less than g dt^2 / (8 v) = 3e-5 s."""
    monitor = iris_monitor()
    run(free_fall, state_at([5.0, 4.0, -1.0]), DT, 2.0, stop=monitor)
    assert monitor.collision.kind == "floor"
    exact = np.sqrt(2.0 * (1.0 - 0.055) / GRAVITY)
    bound = GRAVITY * DT**2 / (8.0 * GRAVITY * exact)
    assert abs(monitor.collision.time - exact) < bound
    assert abs(monitor.collision.time - exact) < DT / 100


@pytest.mark.parametrize("heading_deg", [0.0, 30.0, 45.0, 90.0, 200.0])
def test_floor_margin_is_the_height_of_the_lowest_foot(heading_deg):
    """Nose up 20 deg, 1 m up: the lowest feet are the back ones, at
    1 - (0.235 sin 20 + 0.055 cos 20) whatever the heading. (Placed with R^T
    instead of R the feet would depend on the heading: at 45 deg they would reach
    0.332 sin 20 down instead of 0.235 sin 20.)"""
    pitch = np.radians(20.0)
    attitude = quaternion.from_euler(0.0, pitch, np.radians(heading_deg))
    margin = iris_monitor().margins(state_at([5.0, 4.0, -1.0], attitude=attitude))[-1]
    expected = 1.0 - (0.235 * np.sin(pitch) + 0.055 * np.cos(pitch))
    assert margin == pytest.approx(expected, abs=1e-12)


def fly_iris(schedule, t_end, monitor, landed=True, start=None):
    """An Iris with its gear and autopilot, the monitor as `stop`."""
    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    autopilot = Autopilot(vehicle, IRIS_GAINS, DT, landed=landed)

    def pre_step(t, y):
        autopilot.velocity_command = np.array(schedule(t), dtype=float)
        autopilot(t, y)

    if start is None:
        start = state_at([5.0, 4.0, -0.055 + 0.001])  # gear 1 mm into the floor
    return run(
        vehicle.derivative,
        start,
        DT,
        t_end,
        stop=monitor,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )


def test_standing_on_the_floor_is_not_a_collision():
    """At rest the gear is 1 mm into the floor and the sphere (0.384 m) reaches far
    below it: neither counts while the vehicle has not left the floor."""
    monitor = iris_monitor()
    times, _ = fly_iris(lambda t: [0.0, 0.0, 0.0], 3.0, monitor)
    assert monitor.collision is None and not monitor.airborne
    assert times[-1] == pytest.approx(3.0)


def test_take_off_and_hover_is_not_a_collision():
    monitor = iris_monitor()
    times, states = fly_iris(
        lambda t: [0.0, 0.0, -1.0] if 1.0 <= t < 2.0 else [0.0, 0.0, 0.0], 6.0, monitor
    )
    assert monitor.collision is None and monitor.airborne
    assert times[-1] == pytest.approx(6.0)
    assert -states[-1, 2] > 0.8  # hovering about 1 m up


def test_coming_back_down_after_take_off_is_a_floor_crash():
    monitor = iris_monitor()
    times, states = fly_iris(
        lambda t: [0.0, 0.0, -1.0] if t < 1.0 else [0.0, 0.0, 1.0], 5.0, monitor
    )
    hit = monitor.collision
    assert hit.kind == "floor"
    assert times[-2] < hit.time <= times[-1]
    # The lowest gear point is at the floor at the interpolated moment.
    before, after = (monitor.margins(s)[-1] for s in states[-2:])
    assert before >= 0.0 > after
    assert hit.time > 1.5


@pytest.mark.parametrize(
    "peak, crash",
    [
        (0.0002, False),
        (TAKEOFF_CLEARANCE - 1e-4, False),
        (TAKEOFF_CLEARANCE + 1e-4, True),
    ],
)
def test_a_hop_lower_than_the_takeoff_clearance_is_not_a_take_off(peak, crash):
    """The gear lifted straight up to `peak` above the floor and put back 1 mm into
    it: a crash only when the hop went higher than TAKEOFF_CLEARANCE. (A gear
    unloading with thrust just below the weight hops a fraction of a millimetre.)
    The batch monitor judges the same."""
    monitor = iris_monitor()
    batch = BatchCollisionMonitor(
        ROOM, (), SPHERE.centers, SPHERE.radii, IRIS_LANDING_GEAR.points, 1
    )
    foot = -monitor.margins(state_at([5.0, 4.0, 0.0]))[-1]  # feet below the CoM
    heights = np.concatenate(
        [np.linspace(-0.001, peak, 11), np.linspace(peak, -0.001, 11)]
    )
    for k, height in enumerate(heights):
        state = state_at([5.0, 4.0, -foot - height])
        monitor(k * DT, state)
        batch(k * DT, state[None, :])
    assert (monitor.collision is not None) == crash
    assert monitor.airborne == crash
    assert batch.collided[0] == crash and batch.airborne[0] == crash
    if crash:
        assert monitor.collision.kind == "floor"
        assert batch.collision(0).kind == "floor"
        assert batch.collision(0).time == pytest.approx(
            monitor.collision.time, abs=1e-12
        )


def test_reset_forgets_the_flight():
    monitor = iris_monitor()
    run(free_fall, state_at([5.0, 4.0, -1.0]), DT, 2.0, stop=monitor)
    monitor.reset()
    assert monitor.collision is None and not monitor.airborne
    run(free_fall, state_at([5.0, 4.0, -1.0]), DT, 2.0, stop=monitor)
    assert monitor.collision.kind == "floor"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(centers=np.zeros((0, 3)), radii=[]), "sphere_centers"),
        (dict(centers=[[0.0, 0.0, 0.0]], radii=[0.1, 0.2]), "one value per centre"),
        (dict(centers=[[0.0, 0.0, 0.0]], radii=[-0.1]), "> 0"),
    ],
)
def test_monitor_rejects_invalid_hitbox(kwargs, match):
    with pytest.raises(ValueError, match=match):
        iris_monitor(**kwargs)


def test_monitor_rejects_invalid_floor_points():
    with pytest.raises(ValueError, match="floor_points"):
        CollisionMonitor(ROOM, (), [[0.0, 0.0, 0.0]], [0.1], [0.0, 0.0, 0.1])


# ---- what the samples alone miss ---------------------------------------------------


def sampled_path_misses(monitor, check, start, direction, step, n):
    """True when no sample of a straight path touches: the margins of `check` at
    start + k * step * direction (k = 0 .. n - 1) are all >= 0."""
    for k in range(n):
        y = state_at(np.asarray(start) + k * step * np.asarray(direction))
        if monitor.margins(y)[check] < 0.0:
            return False
    return True


def test_a_thin_wall_is_only_missed_beyond_2r_plus_w_per_step():
    """A slab 2 cm thick across the room, crossed head on. Inflated by R it is
    2R + w long along the path, so steps shorter than that always land a sample
    inside it, and steps longer can jump it: the Iris at 100 Hz would need
    (2R + w) / dt = 79 m/s, almost four times its top speed (20.7 m/s)."""
    w = 0.02
    slab = Box(center=[5.0, 4.0, -1.5], size=[w, 8.0, 3.0])
    monitor = iris_monitor([slab])
    limit = 2.0 * R + w
    assert limit / DT == pytest.approx(78.8, abs=0.1)
    step = 0.99 * limit
    for phase in np.linspace(0.0, 1.0, 40, endpoint=False):
        start = [5.0 - 3.0 * step + phase * step, 4.0, -1.5]
        assert not sampled_path_misses(monitor, 1, start, [1, 0, 0], step, 7)
    # Just past the limit, a sample right before the slab's inflated band and the
    # next one right after it: the slab is jumped.
    step = 1.01 * limit
    start = [5.0 - limit / 2.0 - 1e-3 - 3.0 * step, 4.0, -1.5]
    assert sampled_path_misses(monitor, 1, start, [1, 0, 0], step, 7)


def test_grazing_a_pillar_misses_at_most_the_depth_of_the_chord():
    """Passing a pillar sideways, the inflated pillar (radius rho = R + 0.05) is
    crossed along a chord. When the chord is shorter than the step no sample falls
    on it: the deepest contact that can go unseen is rho - sqrt(rho^2 - s^2 / 4)
    ~ s^2 / (8 rho), 1.17 cm for the Iris at 20 m/s."""
    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    monitor = iris_monitor([pillar])
    rho = R + 0.05
    step = 20.0 * DT
    exact = rho - np.sqrt(rho**2 - step**2 / 4.0)
    assert exact == pytest.approx(step**2 / (8.0 * rho), rel=0.02)
    assert exact == pytest.approx(0.0117, abs=1e-4)
    worst = 0.0
    for depth in np.linspace(0.0, 1.5 * exact, 61):
        offset = rho - depth  # distance of the path from the pillar's axis
        for phase in np.linspace(0.0, 1.0, 20, endpoint=False):
            start = [5.0 - 3.0 * step + phase * step, 4.0 + offset, -1.5]
            if sampled_path_misses(monitor, 1, start, [1, 0, 0], step, 7):
                worst = max(worst, depth)
    assert worst <= exact + 1e-12
    assert worst > 0.95 * exact


# ---- the swept check: contacts between the samples -----------------------------------


def test_a_thin_wall_jumped_between_samples_is_found_at_the_exact_moment():
    """The slab the samples jump at 1.01 (2R + w) per step (80 m/s): the swept
    check finds it at every phase of the samples, at the exact moment (a flat face
    approached head on)."""
    w = 0.02
    slab = Box(center=[5.0, 4.0, -1.5], size=[w, 8.0, 3.0])
    speed = 1.01 * (2.0 * R + w) / DT
    for phase in np.linspace(0.0, 1.0, 40, endpoint=False):
        monitor = iris_monitor([slab])
        x0 = 5.0 - w / 2 - R - 3.0 * speed * DT + phase * speed * DT
        run(
            straight_line,
            state_at([x0, 4.0, -1.5], [speed, 0.0, 0.0]),
            DT,
            1.0,
            monitor,
        )
        assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
        exact = (5.0 - w / 2 - R - x0) / speed
        assert monitor.collision.time == pytest.approx(exact, abs=1e-9)


def first_touch_of_a_pillar(rho, offset, x0, speed):
    """Moment a centre moving along x at `offset` from a pillar's axis (x = 5)
    comes within rho of it, from x0 at t = 0."""
    return (5.0 - np.sqrt(rho**2 - offset**2) - x0) / speed


def test_grazes_between_samples_are_found_near_their_first_touch():
    """Passing the 5 cm pillar at 20 m/s, from 2 micrometres deep to well past the
    deepest graze the samples can miss, at 17 phases of the samples: every graze is
    found, as the obstacle, within 2e-5 s of the first touch (0.4 mm along the
    path). Grazes found between samples are reported where a sphere comes within
    SWEEP_TOLERANCE, a little early; contacts at a sample get their last bit
    interpolated."""
    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    rho = R + 0.05
    speed = 20.0
    step = speed * DT
    deepest_missed = rho - np.sqrt(rho**2 - step**2 / 4.0)  # 1.17 cm
    for depth in [2e-6, 1e-5, 1e-4, 1e-3, deepest_missed, 1.5 * deepest_missed, 0.1]:
        for phase in np.linspace(0.0, 1.0, 17, endpoint=False):
            monitor = iris_monitor([pillar])
            x0 = 3.0 + phase * step
            start = state_at([x0, 4.0 + rho - depth, -1.5], [speed, 0.0, 0.0])
            run(straight_line, start, DT, 0.2, stop=monitor)
            assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
            first = first_touch_of_a_pillar(rho, rho - depth, x0, speed)
            assert monitor.collision.time == pytest.approx(first, abs=2e-5)


def test_passing_within_the_sweep_tolerance_is_a_contact():
    """Coming within SWEEP_TOLERANCE (1 micrometre) counts as touching: it is what
    ends the advancement at a shape it only brushes, where the steps m / |move|
    would otherwise shrink without end."""
    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    monitor = iris_monitor([pillar])
    start = [3.07, 4.0 + R + 0.05 + 0.5 * SWEEP_TOLERANCE, -1.5]
    run(straight_line, state_at(start, [20.0, 0.0, 0.0]), DT, 0.2, stop=monitor)
    assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)


@pytest.mark.parametrize("clearance", [10 * SWEEP_TOLERANCE, 1e-3])
def test_passing_close_is_not_a_contact(clearance):
    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    for phase in np.linspace(0.0, 1.0, 9, endpoint=False):
        monitor = iris_monitor([pillar])
        start = [3.0 + phase * 0.2, 4.0 + R + 0.05 + clearance, -1.5]
        run(straight_line, state_at(start, [20.0, 0.0, 0.0]), DT, 0.2, stop=monitor)
        assert monitor.collision is None


def test_a_contact_between_samples_beats_one_at_the_sample():
    """In one step (80 m/s) the sphere grazes a pillar early and ends inside the
    north wall: the pillar, touched first, is reported."""
    pillar = Cylinder(base=[9.2, 4.4, 0.0], radius=0.05, height=3.0)
    monitor = iris_monitor([pillar])
    speed = 80.0
    x0 = 8.9 - 2 * speed * DT  # samples at x = 8.9 (free) and 9.7 (in the wall)
    run(straight_line, state_at([x0, 4.0, -1.5], [speed, 0.0, 0.0]), DT, 1.0, monitor)
    assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
    first = first_touch_of_a_pillar(R + 0.05, 0.4, x0, speed) + (9.2 - 5.0) / speed
    assert monitor.collision.time == pytest.approx(first, abs=2e-5)
    assert monitor.collision.time < 2 * DT + (10.0 - R - 8.9) / speed  # the wall


def test_the_move_between_samples_is_the_straight_line():
    """A sphere 1 m ahead of the centre of mass, turned 90 degrees in one step
    (far more than any vehicle turns in 10 ms, to make the shape of the move
    visible): it goes from (1, 0) to (0, 1) along the chord, which passes 0.71 m
    from the centre and touches a post there, which the samples and the true arc
    (1 m out) do not. The contact is where the chord comes within 0.15 m of the
    post: s = 0.5 - 0.15 / sqrt(2) of the step."""
    post = Cylinder(base=[5.5, 4.5, 0.0], radius=0.05, height=3.0)
    monitor = CollisionMonitor(
        ROOM, [post], [[1.0, 0.0, 0.0]], [0.1], IRIS_LANDING_GEAR.points
    )
    heading_north = state_at([5.0, 4.0, -1.5])
    heading_east = state_at(
        [5.0, 4.0, -1.5], attitude=quaternion.from_euler(0.0, 0.0, np.pi / 2)
    )
    assert not monitor(0.0, heading_north)
    assert np.all(monitor.margins(heading_east)[:2] > 0.0)  # free at the sample
    assert monitor(DT, heading_east)
    assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
    s = 0.5 - 0.15 / np.sqrt(2.0)
    assert monitor.collision.time == pytest.approx(s * DT, abs=1e-12)


def test_far_from_everything_nothing_is_advanced(monkeypatch):
    """Across the middle of the room at 10 m/s the margins at two samples add up to
    more than the move, so the swept check never runs its advancement."""
    advances = []
    advance = CollisionMonitor._advance

    def counting_advance(self, *args):
        advances.append(1)
        return advance(self, *args)

    monkeypatch.setattr(collision.CollisionMonitor, "_advance", counting_advance)
    monitor = iris_monitor([Cylinder(base=[5.0, 1.0, 0.0], radius=0.05, height=3.0)])
    start = state_at([2.0, 4.0, -1.5], [10.0, 0.0, 0.0])
    run(straight_line, start, DT, 0.5, stop=monitor)
    assert monitor.collision is None
    assert advances == []
    # passing 2 cm from the pillar it does run (and finds nothing)
    monitor = iris_monitor(
        [Cylinder(base=[5.0, 4.0 + R + 0.07, 0.0], radius=0.05, height=3.0)]
    )
    run(straight_line, start, DT, 0.5, stop=monitor)
    assert monitor.collision is None
    assert len(advances) > 0


# ---- the Iris's own spheres ---------------------------------------------------------


def climb_to_the_ceiling(hitbox, attitude=(1.0, 0.0, 0.0, 0.0)):
    """Climb straight up at 1 m/s from 1 m: the moment the ceiling (3 m) is touched."""
    monitor = iris_monitor(hitbox=hitbox)
    start = state_at([5.0, 4.0, -1.0], [0.0, 0.0, -1.0], attitude=attitude)
    run(straight_line, start, DT, 5.0, stop=monitor)
    assert monitor.collision.kind == "room"
    return monitor.collision.time


def test_the_iris_spheres_give_back_room_under_the_ceiling():
    """Level, the rotors reach 0.151 m up where the bounding sphere reaches 0.384 m:
    the Iris climbs 0.233 m closer to the ceiling before touching it."""
    up = IRIS_HITBOX.extent([0, 0, -1])
    assert up == pytest.approx(0.023 + np.hypot(0.128, 0.0025))
    assert climb_to_the_ceiling(IRIS_HITBOX) == pytest.approx(2.0 - up, abs=1e-9)
    gained = climb_to_the_ceiling(IRIS_HITBOX) - climb_to_the_ceiling(SPHERE)
    assert gained == pytest.approx(0.233, abs=1e-3)


def test_the_iris_spheres_follow_its_roll():
    """Rolled 90 deg (right side down), what reaches up is its left side: a rotor
    0.22 m out + 0.128 m, so it touches the ceiling 0.197 m sooner than level."""
    rolled = quaternion.from_euler(np.pi / 2, 0.0, 0.0)
    side = IRIS_HITBOX.extent([0, -1, 0])
    assert side == pytest.approx(0.22 + np.hypot(0.128, 0.0025))
    t = climb_to_the_ceiling(IRIS_HITBOX, rolled)
    assert t == pytest.approx(2.0 - side, abs=1e-9)
    assert climb_to_the_ceiling(IRIS_HITBOX) - t == pytest.approx(0.197, abs=1e-3)


def test_the_iris_spheres_nose_first_into_a_wall():
    """Nose first at the north wall the front body spheres lead (0.47/3 + 0.124 m
    ahead), not the rotors (their hubs are only 0.13 m forward): 0.280 m."""
    monitor = iris_monitor(hitbox=IRIS_HITBOX)
    run(straight_line, state_at([5.0, 4.0, -1.5], [2.0, 0.0, 0.0]), DT, 5.0, monitor)
    nose = IRIS_HITBOX.extent([1, 0, 0])
    assert nose == pytest.approx(0.2803, abs=1e-4)
    assert monitor.collision.time == pytest.approx((5.0 - nose) / 2.0, abs=1e-9)
    np.testing.assert_allclose(monitor.collision.position, [10 - nose, 4, -1.5])


def test_the_iris_spheres_take_off_and_land_without_a_crash_on_the_floor():
    """Standing still, the spheres reach 0.07 m below the floor: the floor is the
    gear's to judge, so a take-off and hover are still no crash."""
    assert IRIS_HITBOX.extent([0, 0, 1]) > 0.054 + 0.05
    monitor = iris_monitor(hitbox=IRIS_HITBOX)
    times, _ = fly_iris(
        lambda t: [0.0, 0.0, -1.0] if 1.0 <= t < 2.0 else [0.0, 0.0, 0.0], 4.0, monitor
    )
    assert monitor.collision is None and monitor.airborne
    assert times[-1] == pytest.approx(4.0)


def test_smaller_spheres_miss_deeper_grazes():
    """The price of the tighter shape: a graze is missed up to the depth of a chord
    of the sphere that grazes, and the rotor spheres (0.128 m) are smaller than the
    bounding sphere (0.384 m). Passing a 5 cm pillar sideways at 20 m/s, the
    deepest unseen graze is at most rho - sqrt(rho^2 - s^2/4) with rho = 0.128 +
    0.05 m: 3.1 cm (1.2 cm with the bounding sphere). Head on, a thin wall still
    needs a step longer than the drone's length along the path (front and back
    reach + w): 58 m/s."""
    w = 0.02
    length = IRIS_HITBOX.extent([1, 0, 0]) + IRIS_HITBOX.extent([-1, 0, 0]) + w
    assert length / DT == pytest.approx(58.1, abs=0.1)
    slab = Box(center=[5.0, 4.0, -1.5], size=[w, 8.0, 3.0])
    walls = iris_monitor([slab], hitbox=IRIS_HITBOX)
    step = 0.99 * length
    for phase in np.linspace(0.0, 1.0, 40, endpoint=False):
        start = [5.0 - 3.0 * step + phase * step, 4.0, -1.5]
        assert not sampled_path_misses(walls, 1, start, [1, 0, 0], step, 7)

    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    monitor = iris_monitor([pillar], hitbox=IRIS_HITBOX)
    rho = np.hypot(0.128, 0.0025) + 0.05
    step = 20.0 * DT
    bound = rho - np.sqrt(rho**2 - step**2 / 4.0)
    assert bound == pytest.approx(0.031, abs=1e-3)
    side = IRIS_HITBOX.extent([0, -1, 0])  # the side that passes the pillar
    worst = 0.0
    for depth in np.linspace(0.0, 1.5 * bound, 46):
        offset = side + 0.05 - depth
        for phase in np.linspace(0.0, 1.0, 20, endpoint=False):
            start = [5.0 - 3.0 * step + phase * step, 4.0 + offset, -1.5]
            if sampled_path_misses(monitor, 1, start, [1, 0, 0], step, 7):
                worst = max(worst, depth)
    assert worst <= bound + 1e-12
    assert worst > 0.0116  # deeper than anything the bounding sphere misses


def test_the_iris_spheres_grazes_are_found():
    """The 13 spheres passing the 5 cm pillar at 20 m/s, as deep as 1.5 times the
    deepest graze the samples can miss (3.1 cm): every one is found."""
    pillar = Cylinder(base=[5.0, 4.0, 0.0], radius=0.05, height=3.0)
    side = IRIS_HITBOX.extent([0, -1, 0])
    rho = np.hypot(0.128, 0.0025) + 0.05
    bound = rho - np.sqrt(rho**2 - 0.2**2 / 4.0)
    for depth in np.linspace(1e-4, 1.5 * bound, 7):
        for phase in np.linspace(0.0, 1.0, 10, endpoint=False):
            monitor = iris_monitor([pillar], hitbox=IRIS_HITBOX)
            start = [3.0 + phase * 0.2, 4.0 + side + 0.05 - depth, -1.5]
            run(straight_line, state_at(start, [20.0, 0.0, 0.0]), DT, 0.2, monitor)
            assert (monitor.collision.kind, monitor.collision.index) == ("obstacle", 0)
