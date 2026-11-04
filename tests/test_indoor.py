"""Tests for the indoor geometry against distances computed by hand.

The boxes and cylinders here use sizes that give 3-4-5 triangles, so every expected
distance is an exact number. Two properties of a true distance are also checked
on random points (seeded): it never changes faster than the point moves
(|d(a) - d(b)| <= |a - b|), and outside a convex shape it grows at exactly 1 m per
metre moved straight away from the surface (gradient of length 1).
"""

import numpy as np
import pytest

from jineta.campo.indoor import Box, Cylinder, Room

# A box of 2 x 4 x 6 m centred at the origin: half sizes 1, 2, 3.
BOX = Box(center=[0.0, 0.0, 0.0], size=[2.0, 4.0, 6.0])


# ---- box --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0.0, 0.0, 0.0], -1.0),  # centre: the nearest face is 1 m away (x)
        ([1.0, 0.0, 0.0], 0.0),  # on a face
        ([3.5, 0.0, 0.0], 2.5),  # straight out past the x face
        ([0.0, -5.0, 0.0], 3.0),  # straight out past the y face
        ([4.0, 6.0, 0.0], 5.0),  # past an edge: overhangs 3 and 4 -> 5
        ([1.0 + 2.0, 2.0 + 3.0, 3.0 + 6.0], 7.0),  # past a corner: 2-3-6 -> 7
        ([0.5, 1.9, 0.0], -0.1),  # inside, 0.1 m from the y face
        ([0.0, 0.0, -2.75], -0.25),  # inside, near the top face (up is -z)
    ],
)
def test_box_distance_by_hand(point, expected):
    assert BOX.signed_distance(point) == pytest.approx(expected, abs=1e-12)


def test_box_moves_with_its_center():
    moved = Box(center=[10.0, -3.0, -1.5], size=[2.0, 4.0, 6.0])
    offset = np.array([10.0, -3.0, -1.5])
    for point in ([4.0, 6.0, 0.0], [0.5, 1.9, 0.0], [0.0, 0.0, 0.0]):
        assert moved.signed_distance(offset + point) == pytest.approx(
            BOX.signed_distance(point), abs=1e-12
        )


def test_box_yaw_turns_its_length_from_north_towards_east():
    """A wall 8 m long and 0.2 m thick: at yaw 0 it runs North-South, at yaw 90 deg
    East-West (positive yaw turns North towards East, as a heading)."""
    wall = dict(center=[0.0, 0.0, -1.0], size=[8.0, 0.2, 2.0])
    north_south = Box(**wall)
    east_west = Box(**wall, yaw=np.radians(90.0))
    assert north_south.signed_distance([3.0, 0.0, -1.0]) == pytest.approx(-0.1)
    assert north_south.signed_distance([0.0, 3.0, -1.0]) == pytest.approx(2.9)
    assert east_west.signed_distance([0.0, 3.0, -1.0]) == pytest.approx(-0.1)
    assert east_west.signed_distance([3.0, 0.0, -1.0]) == pytest.approx(2.9)


def test_box_yaw_sign_is_a_heading():
    """At 90 or 45 deg the sense of the turn does not show (the shape looks the
    same); at 30 deg it does. The wall turned to heading 30 deg contains the point
    3 m away on heading 30 deg, and the point 3 m away on heading -30 deg is 60 deg
    off its length: 3 sin 60 deg - 0.1 m to the side of it."""
    wall = Box(center=[0.0, 0.0, -1.0], size=[8.0, 0.2, 2.0], yaw=np.radians(30.0))
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    assert wall.signed_distance([3 * c, 3 * s, -1.0]) == pytest.approx(-0.1)
    assert wall.signed_distance([3 * c, -3 * s, -1.0]) == pytest.approx(
        3 * np.sin(np.radians(60.0)) - 0.1
    )


def test_box_yaw_45_deg_by_hand():
    """A 2 x 2 m square turned 45 deg is a diamond: one corner points North,
    sqrt(2) m from the centre, so a point 3 m North is 3 - sqrt(2) m away. One face
    now faces North-East, 1 m from the centre along that direction, so the point
    1 m away towards North-East, (sqrt(0.5), sqrt(0.5)), lies on it."""
    diamond = Box(center=[0.0, 0.0, 0.0], size=[2.0, 2.0, 2.0], yaw=np.radians(45.0))
    assert diamond.signed_distance([3.0, 0.0, 0.0]) == pytest.approx(3.0 - np.sqrt(2))
    on_face = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0])
    assert diamond.signed_distance(on_face) == pytest.approx(0.0, abs=1e-12)


def test_box_turning_shape_and_point_together_changes_nothing():
    rng = np.random.default_rng(1)
    for _ in range(50):
        yaw = rng.uniform(-np.pi, np.pi)
        turned = Box(center=[0.0, 0.0, 0.0], size=[2.0, 4.0, 6.0], yaw=yaw)
        p = rng.uniform(-6.0, 6.0, 3)
        c, s = np.cos(yaw), np.sin(yaw)
        p_turned = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
        assert turned.signed_distance(p_turned) == pytest.approx(
            BOX.signed_distance(p), abs=1e-12
        )


def test_box_rejects_invalid_input():
    with pytest.raises(ValueError, match="> 0"):
        Box(center=[0.0, 0.0, 0.0], size=[1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        Box(center=[0.0, 0.0], size=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        Box(center=[0.0, np.nan, 0.0], size=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="yaw"):
        Box(center=[0.0, 0.0, 0.0], size=[1.0, 1.0, 1.0], yaw=np.inf)
    with pytest.raises(ValueError, match="points"):
        BOX.signed_distance([1.0, 2.0])


def test_box_is_read_only():
    with pytest.raises(ValueError):
        BOX.center[0] = 5.0
    with pytest.raises(AttributeError):
        BOX.yaw = 1.0


# ---- cylinder ---------------------------------------------------------------------

# A column of radius 3 m and height 8 m standing on the floor at the origin: it
# spans -8 <= z <= 0, middle height z = -4.
COLUMN = Cylinder(base=[0.0, 0.0, 0.0], radius=3.0, height=8.0)


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0.0, 0.0, -4.0], -3.0),  # on the axis at mid height: 3 m to the side
        ([0.0, 0.0, -1.0], -1.0),  # on the axis near the floor: 1 m to the bottom
        ([5.0, 0.0, -4.0], 2.0),  # beside it: 5 - 3
        ([3.0, 4.0, -4.0], 2.0),  # beside it, diagonally: 5 - 3
        ([0.0, 1.0, -10.0], 2.0),  # above the top face
        ([6.0, 0.0, -12.0], 5.0),  # past the top edge: overhangs 3 and 4 -> 5
        ([2.9, 0.0, -4.0], -0.1),  # inside, 0.1 m from the side
    ],
)
def test_cylinder_distance_by_hand(point, expected):
    assert COLUMN.signed_distance(point) == pytest.approx(expected, abs=1e-12)


def test_cylinder_rejects_invalid_input():
    with pytest.raises(ValueError, match="radius"):
        Cylinder(base=[0.0, 0.0, 0.0], radius=0.0, height=1.0)
    with pytest.raises(ValueError, match="height"):
        Cylinder(base=[0.0, 0.0, 0.0], radius=1.0, height=-1.0)
    with pytest.raises(ValueError, match="shape"):
        Cylinder(base=[0.0, 0.0], radius=1.0, height=1.0)


# ---- room -------------------------------------------------------------------------

# A sports hall: 30 m North, 15 m East, 7 m high.
HALL = Room(size=[30.0, 15.0, 7.0])


@pytest.mark.parametrize(
    "point, expected",
    [
        ([15.0, 7.5, -3.5], 3.5),  # the middle: floor and ceiling 3.5 m away
        ([15.0, 7.5, -1.0], 1.0),  # 1 m above the floor
        ([15.0, 7.5, 0.0], 0.0),  # on the floor
        ([15.0, 7.5, -6.5], 0.5),  # 0.5 m below the ceiling
        ([2.0, 7.5, -3.5], 2.0),  # 2 m from the south wall (x = 0)
        ([15.0, 14.0, -3.5], 1.0),  # 1 m from the east wall (y = 15)
        ([15.0, 7.5, 0.2], -0.2),  # under the floor
        ([-3.0, -4.0, -3.5], -5.0),  # outside, past a vertical corner: 3-4-5
    ],
)
def test_room_clearance_by_hand(point, expected):
    assert HALL.clearance(point) == pytest.approx(expected, abs=1e-12)


def test_room_contains_its_inside_and_walls_only():
    assert HALL.contains([1.0, 1.0, -1.0])
    assert HALL.contains([0.0, 1.0, -1.0])  # on a wall
    assert not HALL.contains([31.0, 1.0, -1.0])
    assert HALL.contains([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]).tolist() == [True, False]


def test_room_rejects_invalid_size():
    with pytest.raises(ValueError, match="> 0"):
        Room(size=[30.0, 15.0, 0.0])


@pytest.mark.parametrize(
    "point, expected",
    [
        ([15.0, 7.5, -1.0], 6.0),  # 1 m above the floor: the ceiling is nearest
        ([15.0, 7.5, 0.0], 7.0),  # on the floor: still the ceiling
        ([15.0, 7.5, 5.0], 7.5),  # below the floor: the east and west walls
        ([15.0, 7.5, -6.5], 0.5),  # 0.5 m below the ceiling
        ([2.0, 7.5, -1.0], 2.0),  # 2 m from the south wall (x = 0)
        ([15.0, 14.0, -1.0], 1.0),  # 1 m from the east wall (y = 15)
        ([15.0, 7.5, -10.0], -3.0),  # 3 m above the ceiling
        ([-3.0, 7.5, -11.0], -5.0),  # past the south wall and above the ceiling: 3-4-5
        ([-3.0, -4.0, 20.0], -5.0),  # past a vertical corner, far below the floor
    ],
)
def test_room_wall_clearance_by_hand(point, expected):
    assert HALL.wall_clearance(point) == pytest.approx(expected, abs=1e-12)


def test_room_wall_clearance_is_the_clearance_without_the_floor():
    """Leaving a face out can only give more room, and nothing changes where the
    floor was not the nearest face (the upper half of the room)."""
    points = np.random.default_rng(5).uniform(-10.0, 30.0, (1000, 3))
    assert np.all(HALL.wall_clearance(points) >= HALL.clearance(points) - 1e-12)
    upper = HALL.center + np.random.default_rng(6).uniform(-0.5, 0.5, (500, 3)) * [
        30.0,
        15.0,
        7.0,
    ]
    upper[:, 2] = np.minimum(upper[:, 2], -3.5)
    np.testing.assert_allclose(
        HALL.wall_clearance(upper), HALL.clearance(upper), rtol=0, atol=1e-12
    )


# ---- properties of a distance -----------------------------------------------------

SHAPES = {
    "box": BOX.signed_distance,
    "turned box": Box(
        center=[1.0, -2.0, -3.0], size=[2.0, 4.0, 6.0], yaw=0.7
    ).signed_distance,
    "cylinder": COLUMN.signed_distance,
    "room": HALL.clearance,
    "room without floor": HALL.wall_clearance,
}


@pytest.mark.parametrize("name", SHAPES)
def test_many_points_at_once_give_the_same_as_one_by_one(name):
    distance = SHAPES[name]
    points = np.random.default_rng(2).uniform(-10.0, 30.0, (100, 3))
    together = distance(points)
    assert together.shape == (100,)
    one_by_one = [distance(p) for p in points]
    np.testing.assert_allclose(together, one_by_one, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", SHAPES)
def test_distance_never_changes_faster_than_the_point_moves(name):
    distance = SHAPES[name]
    rng = np.random.default_rng(3)
    a = rng.uniform(-10.0, 30.0, (1000, 3))
    b = a + rng.normal(0.0, 1.0, (1000, 3))
    change = np.abs(distance(a) - distance(b))
    assert np.all(change <= np.linalg.norm(a - b, axis=1) + 1e-12)


@pytest.mark.parametrize("name", ["box", "turned box", "cylinder"])
def test_outside_a_convex_shape_the_gradient_has_length_one(name):
    distance = SHAPES[name]
    rng = np.random.default_rng(4)
    h = 1e-6
    checked = 0
    for p in rng.uniform(-12.0, 12.0, (400, 3)):
        if distance(p) < 0.1:
            continue
        grad = [
            (distance(p + h * e) - distance(p - h * e)) / (2 * h) for e in np.eye(3)
        ]
        assert np.linalg.norm(grad) == pytest.approx(1.0, abs=1e-6)
        checked += 1
    assert checked > 100
