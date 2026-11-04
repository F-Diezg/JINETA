"""Tests for gates: distances to their frames by hand and against brute force, and
crossings through their openings (direction, fraction, edges).

The two test gates are centred 2 m up at the origin, crossed flying North (yaw 0),
so their own axes are n = North, u = East, v = z + 2. Their sizes give 3-4-5
triangles.
"""

from itertools import pairwise

import numpy as np
import pytest

from jineta.campo.collision import CollisionMonitor
from jineta.campo.gates import CircularGate, RectangularGate
from jineta.campo.indoor import Box, Room
from jineta.core.simulation import run

# Opening 2 m wide (half 1) and 1 m high (half 0.5), bars 0.2 m square.
RECT = RectangularGate(
    center=[0.0, 0.0, -2.0], yaw=0.0, width=2.0, height=1.0, thickness=0.2
)
# Opening of radius 0.4 m in a tube 0.2 m across: the tube's centre line is 0.5 m
# from the centre.
RING = CircularGate(center=[0.0, 0.0, -2.0], yaw=0.0, radius=0.4, thickness=0.2)
UP = np.array([0.0, 0.0, -2.0])  # the gates' centre


# ---- distance to the frame ----------------------------------------------------------


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0.0, 0.0, -2.0], 0.5),  # centre: top and bottom bars 0.5 m away
        ([0.0, 0.0, -1.4], -0.1),  # middle of the bottom bar (v = 0.6)
        ([0.4, 0.0, -1.4], 0.3),  # 0.4 m ahead of it: bar 0.1 m deep each side
        ([0.5, 0.0, -1.8], 0.5),  # in front of the opening: 0.3 to the edge, 0.4 ahead
        ([0.5, 1.5, -2.0], 0.5),  # beside the gate: 0.3 past the outer side, 0.4 ahead
        ([0.0, 1.5, -0.9], 0.5),  # past an outer corner: 0.3 and 0.4
        ([0.0, 0.7, -1.9], 0.3),  # in the opening near a corner: the side bar
    ],
)
def test_rectangular_frame_distance_by_hand(point, expected):
    assert RECT.signed_distance(point) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0.0, 0.0, -2.0], 0.4),  # centre: the radius of the opening
        ([1.2, 0.0, -2.0], 1.2),  # 1.2 m ahead on the axis: 5-12-13, minus 0.1
        ([0.0, 0.5, -2.0], -0.1),  # middle of the tube
        ([0.0, 0.0, -2.5], -0.1),  # middle of the tube, at the top
        ([0.4, 0.8, -2.0], 0.4),  # 0.3 out in the plane, 0.4 ahead: 0.5 - 0.1
    ],
)
def test_circular_frame_distance_by_hand(point, expected):
    assert RING.signed_distance(point) == pytest.approx(expected, abs=1e-12)


def four_bars(gate):
    """The rectangular frame as four boxes (only for a gate with yaw 0)."""
    w, h, t = gate.width, gate.height, gate.thickness
    c = gate.center
    return [
        Box(c + np.array([0, 0, -(h + t) / 2]), [t, w + 2 * t, t]),  # top
        Box(c + np.array([0, 0, (h + t) / 2]), [t, w + 2 * t, t]),  # bottom
        Box(c + np.array([0, -(w + t) / 2, 0]), [t, t, h]),  # left
        Box(c + np.array([0, (w + t) / 2, 0]), [t, t, h]),  # right
    ]


def test_rectangular_frame_is_its_four_bars():
    """Outside a union of shapes the distance is the smallest of their distances:
    away from the bars the frame's distance must equal that of the four boxes."""
    points = np.random.default_rng(10).uniform(-2.5, 2.5, (3000, 3)) + UP
    frame = RECT.signed_distance(points)
    bars = np.min([b.signed_distance(points) for b in four_bars(RECT)], axis=0)
    outside = bars > 0.0
    assert outside.sum() > 2500
    np.testing.assert_allclose(frame[outside], bars[outside], rtol=0, atol=1e-12)
    assert np.all(frame[~outside] <= 0.0)


def test_circular_frame_is_its_centre_circle_minus_the_tube():
    """Brute force: the distance to 36 000 points of the tube's centre line, minus
    the tube's radius (exact to ~1e-5 m with that many points)."""
    angles = np.linspace(0.0, 2.0 * np.pi, 36000, endpoint=False)
    circle = np.stack([np.zeros_like(angles), np.cos(angles), np.sin(angles)], 1)
    circle = RING.center + 0.5 * circle
    points = np.random.default_rng(11).uniform(-1.5, 1.5, (200, 3)) + UP
    brute = [np.min(np.linalg.norm(circle - p, axis=1)) - 0.1 for p in points]
    np.testing.assert_allclose(RING.signed_distance(points), brute, atol=2e-5)


def test_yaw_is_a_heading():
    """A gate with yaw 30 deg is crossed flying 30 deg East of North: a point 1.2 m
    along that heading is on its axis (1.2 m from the ring, as above). Turned the
    other way, the same point would be off the axis."""
    yaw = np.radians(30.0)
    gate = CircularGate(center=[0.0, 0.0, -2.0], yaw=yaw, radius=0.4, thickness=0.2)
    ahead = [1.2 * np.cos(yaw), 1.2 * np.sin(yaw), -2.0]
    assert gate.signed_distance(ahead) == pytest.approx(1.2, abs=1e-12)
    np.testing.assert_allclose(gate.to_local(ahead), [1.2, 0.0, 0.0], atol=1e-12)
    assert gate.crossing([0.0, 0.0, -2.0 - 0.0], ahead) is None  # starts on the plane
    back = [-0.1 * np.cos(yaw), -0.1 * np.sin(yaw), -2.0]
    assert gate.crossing(back, ahead).forward


SHAPES = {"rectangular": RECT, "circular": RING}


@pytest.mark.parametrize("name", SHAPES)
def test_many_points_at_once_give_the_same_as_one_by_one(name):
    gate = SHAPES[name]
    points = np.random.default_rng(12).uniform(-3.0, 3.0, (100, 3))
    together = gate.signed_distance(points)
    one_by_one = [gate.signed_distance(p) for p in points]
    np.testing.assert_allclose(together, one_by_one, rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", SHAPES)
def test_distance_never_changes_faster_than_the_point_moves(name):
    gate = SHAPES[name]
    rng = np.random.default_rng(13)
    a = rng.uniform(-3.0, 3.0, (1000, 3)) + UP
    b = a + rng.normal(0.0, 0.5, (1000, 3))
    change = np.abs(gate.signed_distance(a) - gate.signed_distance(b))
    assert np.all(change <= np.linalg.norm(a - b, axis=1) + 1e-12)


@pytest.mark.parametrize("name", SHAPES)
def test_outside_the_frame_the_gradient_has_length_one(name):
    """A true distance grows at 1 m per metre away from the nearest point, also in
    the opening (the frame is not convex, but the property holds outside it)."""
    gate = SHAPES[name]
    rng = np.random.default_rng(14)
    h = 1e-6
    checked = 0
    for p in rng.uniform(-2.0, 2.0, (400, 3)) + UP:
        if gate.signed_distance(p) < 0.05:
            continue
        grad = [
            (gate.signed_distance(p + h * e) - gate.signed_distance(p - h * e))
            / (2 * h)
            for e in np.eye(3)
        ]
        assert np.linalg.norm(grad) == pytest.approx(1.0, abs=1e-6)
        checked += 1
    assert checked > 200


# ---- crossings ----------------------------------------------------------------------


@pytest.mark.parametrize("gate", [RECT, RING], ids=["rectangular", "circular"])
def test_straight_through_the_middle(gate):
    hit = gate.crossing([-0.3, 0.0, -2.0], [0.1, 0.0, -2.0])
    assert hit.forward
    assert hit.fraction == pytest.approx(0.75, abs=1e-15)
    np.testing.assert_allclose(hit.point, [0.0, 0.0, -2.0], atol=1e-15)
    back = gate.crossing([0.1, 0.0, -2.0], [-0.3, 0.0, -2.0])
    assert not back.forward
    assert back.fraction == pytest.approx(0.25, abs=1e-15)


def test_crossing_point_is_interpolated_on_a_slanted_move():
    """From 0.2 m before to 0.6 m after, climbing 0.4 m and drifting 0.8 m East:
    the plane is cut at a quarter of the move."""
    hit = RECT.crossing([-0.2, 0.0, -1.9], [0.6, 0.8, -2.3])
    assert hit.fraction == pytest.approx(0.25)
    np.testing.assert_allclose(hit.point, [0.0, 0.2, -2.0], atol=1e-15)


@pytest.mark.parametrize(
    "start, end",
    [
        ([-0.5, 0.0, -2.0], [-0.1, 0.0, -2.0]),  # does not reach the gate
        ([0.0, 0.0, -2.0], [0.2, 0.0, -2.0]),  # starts on the plane: already after
        ([-0.3, 1.5, -2.0], [0.3, 1.5, -2.0]),  # beside the gate
        ([-0.3, 1.1, -2.0], [0.3, 1.1, -2.0]),  # through the side bar, not the opening
        ([-0.3, 0.0, -1.3], [0.3, 0.0, -1.3]),  # under the gate
        ([-0.3, 0.0, -2.0], [-0.3, 0.5, -2.0]),  # along the plane, before it
    ],
)
def test_moves_that_do_not_go_through_the_opening(start, end):
    assert RECT.crossing(start, end) is None


def test_opening_edges_count_as_inside():
    assert RECT.crossing([-0.1, 1.0, -2.5], [0.1, 1.0, -2.5]) is not None  # corner
    assert RECT.crossing([-0.2, 0.0, -2.0], [0.0, 0.0, -2.0]).fraction == 1.0
    assert RING.crossing([-0.1, 0.25, -2.3], [0.1, 0.25, -2.3]) is not None
    assert RING.crossing([-0.1, 0.3, -2.3], [0.1, 0.3, -2.3]) is None  # 0.42 m out


def test_crossing_found_once_along_a_sampled_flight():
    """Flying North at 3 m/s, sampled at 100 Hz: exactly one pair of samples
    crosses the gate, at the exact moment (the move is straight)."""

    def straight_line(t, y):
        return np.concatenate([y[3:6], np.zeros(3)])

    _, states = run(straight_line, np.array([-1.0, 0.1, -2.1, 3.0, 0, 0]), 0.01, 1.0)
    hits = [
        (k, RING.crossing(a[:3], b[:3])) for k, (a, b) in enumerate(pairwise(states))
    ]
    hits = [(k, h) for k, h in hits if h is not None]
    assert len(hits) == 1
    k, hit = hits[0]
    assert (k + hit.fraction) * 0.01 == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_hitting_a_gate_frame_is_a_gate_collision():
    """A sphere of 0.3 m flying North level with the top bar (2.5 to 2.7 m up)
    touches the bar's front face (x = 4.9) when its centre is 0.3 m short of it.
    The frame is reported as a gate, not as an obstacle."""
    gate = RectangularGate([5.0, 4.0, -2.0], 0.0, 2.0, 1.0, 0.2)
    monitor = CollisionMonitor(
        Room([10.0, 8.0, 4.0]),
        [Box([1.0, 1.0, -1.0], [0.5, 0.5, 2.0])],
        [[0.0, 0.0, 0.0]],
        [0.3],
        [[0.0, 0.0, 0.1]],
        gates=[gate],
    )
    y0 = np.zeros(13)
    y0[0:3] = [3.0, 4.0, -2.6]  # level with the top bar (2.5 to 2.7 m up)
    y0[3] = 2.0
    y0[6] = 1.0

    def straight_line(t, y):
        dy = np.zeros(13)
        dy[0:3] = y[3:6]
        return dy

    run(straight_line, y0, 0.01, 3.0, stop=monitor)
    hit = monitor.collision
    assert (hit.kind, hit.index) == ("gate", 0)
    assert hit.time == pytest.approx((5.0 - 0.1 - 0.3 - 3.0) / 2.0, abs=1e-9)


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: RectangularGate([0, 0], 0.0, 1.0, 1.0, 0.1), "center"),
        (lambda: RectangularGate([0, 0, 0], np.nan, 1.0, 1.0, 0.1), "yaw"),
        (lambda: RectangularGate([0, 0, 0], 0.0, 0.0, 1.0, 0.1), "width"),
        (lambda: RectangularGate([0, 0, 0], 0.0, 1.0, 1.0, 0.0), "thickness"),
        (lambda: CircularGate([0, 0, 0], 0.0, -1.0, 0.1), "radius"),
        (lambda: CircularGate([0, 0, 0], 0.0, 1.0, -0.1), "thickness"),
    ],
)
def test_gates_reject_invalid_input(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_gates_are_read_only():
    with pytest.raises(ValueError):
        RECT.center[0] = 1.0
    hit = RECT.crossing([-0.3, 0.0, -2.0], [0.1, 0.0, -2.0])
    with pytest.raises(ValueError):
        hit.point[0] = 1.0


def test_the_opening_is_checked_where_the_plane_is_cut():
    """A diagonal move is judged at the point where it cuts the plane (a quarter of
    the way here), not where it ends: in through the opening and out to the side
    (u = 0.9 at the plane, 1.5 at the end) is a crossing; in from the side and on
    into the opening (u = 1.3 at the plane, 0.7 at the end) is not."""
    hit = RECT.crossing([-0.1, 0.7, -2.0], [0.3, 1.5, -2.0])
    assert hit is not None
    np.testing.assert_allclose(hit.point, [0.0, 0.9, -2.0], atol=1e-15)
    assert RECT.crossing([-0.1, 1.5, -2.0], [0.3, 0.7, -2.0]) is None
