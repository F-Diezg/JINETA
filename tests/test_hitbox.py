"""Tests for the hitbox: spheres fixed to a vehicle, and the Iris's bounding sphere."""

import numpy as np
import pytest

from jineta.jinete.hitbox import Hitbox, bounding_sphere, box_corners, disc_reach
from jineta.jinete.profiles import (
    IRIS,
    IRIS_HITBOX,
    IRIS_PROPELLER_RADIUS,
    MULTIROTOR_PROFILES,
)


def rim_points(center, radius, n=3600):
    """Points on the rim of a rotor disc (plane of body x and y), every 0.1 deg."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    rim = np.stack([np.cos(angles), np.sin(angles), np.zeros(n)], axis=1)
    return np.asarray(center) + radius * rim


def test_disc_reach_by_hand():
    # |c_xy| = 0.5 (3-4-5), + radius 0.5 = 1.0; with c_z = 0.75: sqrt(1 + 0.5625).
    assert disc_reach([0.3, 0.4, 0.0], 0.5) == pytest.approx(1.0, abs=1e-15)
    assert disc_reach([0.3, 0.4, 0.75], 0.5) == pytest.approx(1.25, abs=1e-15)
    # The triangle inequality would say |c| + r = 1.401...: more than the truth.
    assert np.linalg.norm([0.3, 0.4, 0.75]) + 0.5 > 1.4


@pytest.mark.parametrize("center", [[0.13, 0.22, -0.023], [-0.2, 0.05, 0.3], [0, 0, 1]])
def test_disc_reach_is_the_farthest_point_of_the_rim(center):
    distances = np.linalg.norm(rim_points(center, 0.128), axis=1)
    reach = disc_reach(center, 0.128)
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


def test_iris_hitbox_holds_its_rotor_discs_and_its_box():
    assert len(IRIS_HITBOX.radii) == 1
    np.testing.assert_array_equal(IRIS_HITBOX.centers, [[0.0, 0.0, 0.0]])
    radius = IRIS_HITBOX.radii[0]
    # The front rotors (|hub_xy| = 0.2555 m, 0.023 m above the CoM) reach farthest.
    expected = np.hypot(np.hypot(0.13, 0.22) + 0.128, 0.023)
    assert radius == pytest.approx(expected, abs=1e-12)
    assert radius == pytest.approx(0.3842, abs=1e-4)
    for rotor in IRIS.rotors:
        rim = rim_points(rotor.position, IRIS_PROPELLER_RADIUS)
        assert np.linalg.norm(rim, axis=1).max() <= radius + 1e-12
    assert np.linalg.norm(box_corners([0.47, 0.47, 0.11]), axis=1).max() < radius


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
