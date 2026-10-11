"""Tests for the multirotor parameters, the mixer and the Iris profile."""

import dataclasses

import numpy as np
import pytest

from jineta.core.environment import GRAVITY
from jineta.jinete.multirotor import Rotor
from jineta.jinete.profiles import IRIS, flu_to_frd

# --- Mixer -------------------------------------------------------------------


def test_equal_thrusts_give_pure_lift():
    """Iris is not square (arms 0.22 m front, 0.20 m back) but still balanced."""
    assert np.allclose(IRIS.wrench([3.0, 3.0, 3.0, 3.0]), [12.0, 0.0, 0.0, 0.0])


def test_single_rotor_wrench():
    """Rotor 2 (front left, cw) at x = 0.13, y = -0.22 pushing 1 N."""
    t_total, mx, my, mz = IRIS.wrench([0.0, 0.0, 1.0, 0.0])
    assert np.isclose(t_total, 1.0)
    assert np.isclose(mx, 0.22)  # left side pushes up -> rolls right
    assert np.isclose(my, 0.13)  # ahead of the CoM -> nose up
    assert np.isclose(mz, -0.06)  # cw rotor -> body yaws left


def test_mixer_round_trip():
    rng = np.random.default_rng(seed=0)
    for _ in range(20):
        thrusts = rng.uniform(0.0, IRIS.max_rotor_thrust, size=4)
        assert np.allclose(IRIS.rotor_thrusts(IRIS.wrench(thrusts)), thrusts)


def test_roll_command_pushes_harder_on_the_left():
    thrusts = IRIS.rotor_thrusts([15.0, 0.5, 0.0, 0.0])
    y = np.array([rotor.position[1] for rotor in IRIS.rotors])
    assert thrusts[y < 0].min() > thrusts[y > 0].max()
    assert np.allclose(IRIS.wrench(thrusts), [15.0, 0.5, 0.0, 0.0])


def test_yaw_command_speeds_up_the_ccw_rotors():
    """To yaw right, the rotors whose reaction torque yaws right push harder."""
    thrusts = IRIS.rotor_thrusts([15.0, 0.0, 0.0, 0.1])
    spin = np.array([rotor.spin for rotor in IRIS.rotors])
    assert thrusts[spin == 1].min() > thrusts[spin == -1].max()
    assert np.allclose(IRIS.wrench(thrusts), [15.0, 0.0, 0.0, 0.1])


def test_rotor_thrusts_are_clipped():
    assert np.allclose(IRIS.rotor_thrusts([1000.0, 0, 0, 0]), IRIS.max_rotor_thrust)
    assert np.allclose(IRIS.rotor_thrusts([-10.0, 0, 0, 0]), 0.0)


# --- Validation ----------------------------------------------------------------


def test_rejects_rotors_all_spinning_the_same_way():
    """Without opposite spins the reaction torques cannot be balanced: no yaw."""
    same_spin = tuple(Rotor(rotor.position, spin=1) for rotor in IRIS.rotors)
    with pytest.raises(ValueError, match="cannot control"):
        dataclasses.replace(IRIS, rotors=same_spin)


@pytest.mark.parametrize(
    "position, spin",
    [
        ([0.1, 0.1], 1),  # position with 2 components
        ([0.1, 0.1, 0.0], 0),  # spin must be +1 or -1
    ],
)
def test_rejects_invalid_rotor(position, spin):
    with pytest.raises(ValueError):
        Rotor(position, spin)


@pytest.mark.parametrize(
    "name", ["mass", "thrust_coefficient", "moment_constant", "max_rotor_speed"]
)
def test_rejects_non_positive_constants(name):
    with pytest.raises(ValueError, match=name):
        dataclasses.replace(IRIS, **{name: 0.0})


def test_params_cannot_be_changed():
    """Profiles are shared by every vehicle that uses them: no silent edits."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        IRIS.mass = 2.0
    with pytest.raises(ValueError):
        IRIS.rotors[0].position[0] = 1.0


# --- Iris profile ----------------------------------------------------------------


def test_flu_to_frd_flips_y_and_z():
    assert np.array_equal(flu_to_frd([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0])


def test_iris_front_right_rotor_is_front_right():
    """Gazebo calls rotor 0 'front_right': in FRD that is x > 0, y > 0."""
    x, y, z = IRIS.rotors[0].position
    assert x > 0 and y > 0
    assert z < 0  # rotors sit above the CoM: negative z in FRD


def test_iris_thrust_to_weight():
    """4 x 5.84e-6 x 1100^2 = 28.27 N against 1.52 kg x g = 14.91 N."""
    thrust_to_weight = 4 * IRIS.max_rotor_thrust / (IRIS.mass * GRAVITY)
    assert np.isclose(thrust_to_weight, 1.896, atol=1e-3)
