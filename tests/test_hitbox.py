"""Tests for the hitbox: spheres fixed to a vehicle, and the Iris's shape.

The covering is checked on points sampled over the collision geometry it must hold
(rotor cylinders and the body box, surfaces and inside); the sizes that matter for
flying (how far it reaches up, down and forward) are checked against hand-computed
values.
"""

import numpy as np
import pytest

from jineta.jinete.hitbox import (
    Hitbox,
    bounding_sphere,
    box_corners,
    box_spheres,
    disc_reach,
    rotor_spheres,
    union,
)
from jineta.jinete.profiles import (
    IRIS,
    IRIS_BOUNDING_SPHERE,
    IRIS_COLLISION_BOX,
    IRIS_HITBOX,
    IRIS_PROPELLER_RADIUS,
    IRIS_PROPELLER_THICKNESS,
    MULTIROTOR_PROFILES,
)


def cylinder_points(center, radius, thickness, n=720):
    """Points on the edges of a rotor cylinder (axis along body z): its top and
    bottom rims, every 0.5 deg, and its centre."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], axis=1) * radius
    top = rim + np.array([0.0, 0.0, -thickness / 2.0])
    bottom = rim + np.array([0.0, 0.0, thickness / 2.0])
    return np.asarray(center) + np.vstack([top, bottom, [[0.0, 0.0, 0.0]]])


def box_points(size, per_side=25):
    """A grid over the whole box (inside and every face, corners included)."""
    axes = [np.linspace(-s / 2.0, s / 2.0, per_side) for s in size]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def iris_rotor_cylinders():
    return np.vstack(
        [
            cylinder_points(r.position, IRIS_PROPELLER_RADIUS, IRIS_PROPELLER_THICKNESS)
            for r in IRIS.rotors
        ]
    )


# ---- building blocks ----------------------------------------------------------------


def test_disc_reach_by_hand():
    # |c_xy| = 0.5 (3-4-5), + radius 0.5 = 1.0; with c_z = 0.75: sqrt(1 + 0.5625).
    assert disc_reach([0.3, 0.4, 0.0], 0.5) == pytest.approx(1.0, abs=1e-15)
    assert disc_reach([0.3, 0.4, 0.75], 0.5) == pytest.approx(1.25, abs=1e-15)
    # A thickness of 0.5 adds 0.25 to the height: (|c_z| + t/2) = 1.0 -> sqrt(2).
    assert disc_reach([0.3, 0.4, -0.75], 0.5, 0.5) == pytest.approx(np.sqrt(2.0))
    # The triangle inequality would say |c| + r = 1.401...: more than the truth.
    assert np.linalg.norm([0.3, 0.4, 0.75]) + 0.5 > 1.4


@pytest.mark.parametrize("center", [[0.13, 0.22, -0.023], [-0.2, 0.05, 0.3], [0, 0, 1]])
@pytest.mark.parametrize("thickness", [0.0, 0.005, 0.2])
def test_disc_reach_is_the_farthest_point_of_the_cylinder(center, thickness):
    distances = np.linalg.norm(cylinder_points(center, 0.128, thickness), axis=1)
    reach = disc_reach(center, 0.128, thickness)
    assert distances.max() <= reach + 1e-12
    assert distances.max() == pytest.approx(reach, abs=1e-6)


def test_box_corners():
    corners = box_corners([2.0, 4.0, 6.0])
    assert corners.shape == (8, 3)
    assert {tuple(c) for c in corners} == {
        (sx, sy, sz) for sx in (1, -1) for sy in (2, -2) for sz in (3, -3)
    }


def test_bounding_sphere_takes_the_farthest_of_rotors_and_body():
    by_rotor = bounding_sphere([[0.3, 0.4, 0.0]], 0.5, [[0.1, 0.0, 0.0]])
    assert by_rotor.radii[0] == pytest.approx(1.0)
    by_body = bounding_sphere([[0.3, 0.4, 0.0]], 0.5, [[0.0, 2.0, 0.0]])
    assert by_body.radii[0] == pytest.approx(2.0)
    np.testing.assert_array_equal(by_body.centers, [[0.0, 0.0, 0.0]])


def test_rotor_spheres_hold_their_cylinders_exactly():
    spheres = rotor_spheres([[0.3, 0.4, 0.0], [-0.3, 0.4, 0.1]], 0.4, 0.6)
    np.testing.assert_allclose(spheres.radii, [0.5, 0.5])  # 3-4-5: sqrt(0.4² + 0.3²)
    for center in spheres.centers:
        assert np.all(spheres.contains(cylinder_points(center, 0.4, 0.6)))


@pytest.mark.parametrize("divisions", [(1, 1, 1), (2, 3, 1), (3, 3, 1), (4, 2, 3)])
def test_box_spheres_cover_the_box(divisions):
    size = [0.6, 0.8, 0.24]
    spheres = box_spheres(size, divisions)
    assert len(spheres.radii) == np.prod(divisions)
    cell = np.array(size) / divisions
    np.testing.assert_allclose(spheres.radii, np.linalg.norm(cell) / 2.0)
    assert np.all(spheres.contains(box_points(size)))
    # The spheres are centred in the box: as far out on one side as on the other.
    np.testing.assert_allclose(spheres.centers.mean(axis=0), 0.0, atol=1e-15)


def test_box_spheres_reach_by_hand():
    """A 0.6 x 0.8 x 0.24 box in 2 x 2 x 1 cells of 0.3 x 0.4 x 0.24: spheres of
    radius sqrt(0.09 + 0.16 + 0.0576) / 2 = 0.2773 at (+-0.15, +-0.2, 0)."""
    spheres = box_spheres([0.6, 0.8, 0.24], (2, 2, 1))
    r = np.sqrt(0.3**2 + 0.4**2 + 0.24**2) / 2.0
    assert spheres.extent([1, 0, 0]) == pytest.approx(0.15 + r)
    assert spheres.extent([0, 0, -1]) == pytest.approx(r)


def test_box_spheres_reject_invalid_input():
    with pytest.raises(ValueError, match="divisions"):
        box_spheres([1.0, 1.0, 1.0], (0, 1, 1))
    with pytest.raises(ValueError, match="size"):
        box_spheres([1.0, -1.0, 1.0], (1, 1, 1))


def test_union_and_extent():
    a = Hitbox([[1.0, 0.0, 0.0]], [0.5])
    b = Hitbox([[0.0, 0.0, -2.0], [0.0, 3.0, 0.0]], [0.25, 1.0])
    both = union(a, b)
    assert len(both.radii) == 3
    assert both.extent([1, 0, 0]) == pytest.approx(1.5)
    assert both.extent([0, 0, -1]) == pytest.approx(2.25)
    assert both.extent([0, 2, 0]) == pytest.approx(4.0)  # the direction is normalised
    assert both.contains([[1.4, 0.0, 0.0], [0.0, 0.0, -2.3]]).tolist() == [True, False]


# ---- the Iris -----------------------------------------------------------------------


def test_iris_hitbox_holds_its_rotor_cylinders_and_its_box():
    assert np.all(IRIS_HITBOX.contains(iris_rotor_cylinders()))
    assert np.all(IRIS_HITBOX.contains(box_points(IRIS_COLLISION_BOX)))


def test_iris_hitbox_is_four_rotors_and_nine_body_spheres():
    assert len(IRIS_HITBOX.radii) == 13
    rotors = IRIS_HITBOX.radii[:4]
    np.testing.assert_allclose(rotors, np.hypot(0.128, 0.0025), rtol=0, atol=1e-15)
    body = IRIS_HITBOX.radii[4:]
    np.testing.assert_allclose(body, np.linalg.norm([0.47 / 3, 0.47 / 3, 0.11]) / 2)
    assert body[0] == pytest.approx(0.1237, abs=1e-4)


def test_iris_hitbox_reach_by_hand():
    """How far the 13 spheres reach from the centre of mass, level:
    up: a rotor hub 0.023 m up + 0.128 m; down: a body sphere, 0.124 m;
    forward: the front body spheres, 0.47/3 + 0.124 m (the front hubs are only
    0.13 m forward); sideways: a rotor 0.22 m out + 0.128 m."""
    rotor_r = np.hypot(0.128, 0.0025)
    body_r = np.linalg.norm([0.47 / 3, 0.47 / 3, 0.11]) / 2
    assert IRIS_HITBOX.extent([0, 0, -1]) == pytest.approx(0.023 + rotor_r)
    assert IRIS_HITBOX.extent([0, 0, 1]) == pytest.approx(body_r)
    assert IRIS_HITBOX.extent([1, 0, 0]) == pytest.approx(0.47 / 3 + body_r)
    assert IRIS_HITBOX.extent([0, 1, 0]) == pytest.approx(0.22 + rotor_r)
    height = IRIS_HITBOX.extent([0, 0, -1]) + IRIS_HITBOX.extent([0, 0, 1])
    assert height == pytest.approx(0.275, abs=1e-3)


def test_iris_bounding_sphere_holds_everything():
    radius = IRIS_BOUNDING_SPHERE.radii[0]
    expected = np.hypot(np.hypot(0.13, 0.22) + 0.128, 0.023 + 0.0025)
    assert radius == pytest.approx(expected, abs=1e-12)
    assert radius == pytest.approx(0.3844, abs=1e-4)
    assert np.all(IRIS_BOUNDING_SPHERE.contains(iris_rotor_cylinders()))
    assert np.all(IRIS_BOUNDING_SPHERE.contains(box_points(IRIS_COLLISION_BOX)))


def test_iris_hitbox_is_smaller_where_it_matters_not_everywhere():
    """In any direction the 13 spheres reach at most max(|c| + r), along the line
    through a front rotor hub: 0.2566 + 0.1280 = 0.3846 m, 0.2 mm more than the
    bounding sphere (a sphere holding a disc bulges above and below its plane). The
    room given back is up (0.233 m), forward (0.104 m) and sideways (0.036 m)."""
    farthest = np.max(np.linalg.norm(IRIS_HITBOX.centers, axis=1) + IRIS_HITBOX.radii)
    rng = np.random.default_rng(7)
    reach = [IRIS_HITBOX.extent(u) for u in rng.normal(size=(2000, 3))]
    assert max(reach) <= farthest + 1e-12
    assert farthest - IRIS_BOUNDING_SPHERE.radii[0] == pytest.approx(2e-4, abs=1e-4)
    sphere = IRIS_BOUNDING_SPHERE.radii[0]
    assert sphere - IRIS_HITBOX.extent([0, 0, -1]) == pytest.approx(0.233, abs=1e-3)
    assert sphere - IRIS_HITBOX.extent([1, 0, 0]) == pytest.approx(0.104, abs=1e-3)
    assert sphere - IRIS_HITBOX.extent([0, 1, 0]) == pytest.approx(0.036, abs=1e-3)


def test_iris_profile_carries_its_hitbox():
    assert MULTIROTOR_PROFILES["iris"].hitbox is IRIS_HITBOX


@pytest.mark.parametrize(
    "centers, radii, match",
    [
        (np.zeros((0, 3)), [], "shape"),
        ([[0.0, 0.0]], [0.1], "shape"),
        ([[0.0, 0.0, 0.0]], [0.1, 0.2], "one value per centre"),
        ([[0.0, 0.0, np.nan]], [0.1], "finite"),
        ([[0.0, 0.0, 0.0]], [0.0], "> 0"),
        ([[0.0, 0.0, 0.0]], [np.inf], "> 0"),
    ],
)
def test_hitbox_rejects_invalid_input(centers, radii, match):
    with pytest.raises(ValueError, match=match):
        Hitbox(centers, radii)


def test_hitbox_is_read_only():
    with pytest.raises(ValueError):
        IRIS_HITBOX.radii[0] = 1.0
    with pytest.raises(ValueError):
        IRIS_HITBOX.centers[0, 0] = 1.0
