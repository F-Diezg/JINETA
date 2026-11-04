"""Tests for collisions in flight: the vehicle's spheres against walls, ceiling and
obstacles, its landing gear against the floor once it has taken off.

Where the motion is known in closed form the moment of contact is checked exactly:
flying straight at a flat face the margin falls linearly, so the interpolated
moment is exact; in free fall onto the floor it is quadratic, and the interpolation
error is below g dt^2 / (8 v). The tunnelling limit (a contact missed between two
samples) is measured against its formulas on straight sampled paths.
"""

import numpy as np
import pytest

from jineta.campo.collision import CollisionMonitor
from jineta.campo.indoor import Box, Cylinder, Room
from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.contact import GroundContact
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_HITBOX, IRIS_LANDING_GEAR

DT = 0.01
R = IRIS_HITBOX.radii[0]  # 0.384 m
ROOM = Room(size=[10.0, 8.0, 3.0])


def iris_monitor(obstacles=(), room=ROOM, centers=None, radii=None):
    return CollisionMonitor(
        room,
        obstacles,
        IRIS_HITBOX.centers if centers is None else centers,
        IRIS_HITBOX.radii if radii is None else radii,
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


# ---- the tunnelling limit -----------------------------------------------------------


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
