"""Tests for the fixed-wing aerodynamic model, with the generic small UAV profile.

Each test checks one physical statement of the model: lift perpendicular to the air
and drag along it, the scaling with the dynamic pressure, the lift-curve slope and
the stall, the induced-drag polar, static stability (pitch and weathercock), the
damping derivatives, the sign of each control surface, and the mirror symmetry of
the aircraft about its plane of symmetry.
"""

import dataclasses

import numpy as np
import pytest

from jineta.jinete.aero import airspeed_from_air_data
from jineta.jinete.fixed_wing import (
    ControlSurfaces,
    aero_force_moment,
    drag_coefficient,
    lift_coefficient,
    stall_blend,
)
from jineta.jinete.profiles import SMALL_FIXED_WING

P = SMALL_FIXED_WING
RHO = 1.225  # [kg/m^3]
V = 25.0  # [m/s]


def loads(alpha_deg=0.0, beta_deg=0.0, speed=V, rates=(0, 0, 0), density=RHO, **surf):
    """Force and moment for an airflow given as V, alpha, beta (degrees)."""
    airspeed = airspeed_from_air_data(
        speed, np.radians(alpha_deg), np.radians(beta_deg)
    )
    return aero_force_moment(P, airspeed, rates, ControlSurfaces(**surf), density)


def qbar_s(speed=V, density=RHO):
    return 0.5 * density * speed**2 * P.wing_area


# --- Forces ---------------------------------------------------------------------------


def test_no_air_no_load():
    force, moment = aero_force_moment(
        P, [0.0, 0.0, 0.0], [0.3, 0.1, 0.0], ControlSurfaces(), RHO
    )
    assert np.array_equal(force, np.zeros(3))
    assert np.array_equal(moment, np.zeros(3))


def test_reference_condition_alpha_zero():
    """alpha = beta = 0, no rates, no controls: drag straight back, lift straight up
    (-z), no side force, and only the zero-alpha pitching moment."""
    force, moment = loads()
    expected_force = qbar_s() * np.array([-drag_coefficient(P, 0.0), 0.0, -P.c_lift_0])
    expected_moment = [0.0, qbar_s() * P.chord * P.c_m_0, 0.0]
    assert np.allclose(force, expected_force)
    assert np.allclose(moment, expected_moment)


@pytest.mark.parametrize("alpha_deg", [-5.0, 4.0, 12.0])
def test_lift_is_perpendicular_to_the_air_and_drag_along_it(alpha_deg):
    """Projecting the body force on the airflow direction gives -D, and on the
    direction perpendicular to it in the plane of symmetry gives L."""
    alpha = np.radians(alpha_deg)
    force, _ = loads(alpha_deg)
    along_air = np.array([np.cos(alpha), 0.0, np.sin(alpha)])  # unit, where air goes
    lift_dir = np.array([np.sin(alpha), 0.0, -np.cos(alpha)])  # unit, perpendicular
    assert np.isclose(-force @ along_air, qbar_s() * drag_coefficient(P, alpha))
    assert np.isclose(force @ lift_dir, qbar_s() * lift_coefficient(P, alpha))


def test_loads_scale_with_the_dynamic_pressure():
    """Twice the speed: four times the loads. Half the density: half the loads."""
    force, moment = loads(3.0, 2.0)
    force_2v, moment_2v = loads(3.0, 2.0, speed=2 * V)
    force_half, moment_half = loads(3.0, 2.0, density=RHO / 2)
    assert np.allclose(force_2v, 4 * force)
    assert np.allclose(moment_2v, 4 * moment)
    assert np.allclose(force_half, force / 2)
    assert np.allclose(moment_half, moment / 2)


# --- Lift curve, stall and drag polar -------------------------------------------------


def test_lift_curve_slope_before_the_stall():
    h = np.radians(1.0)
    slope = (lift_coefficient(P, h) - lift_coefficient(P, -h)) / (2 * h)
    assert np.isclose(slope, P.c_lift_alpha, rtol=1e-6)


def test_stall_blend():
    """0 with attached flow, exactly 1/2 at the stall angle, 1 well past it, and the
    same for negative angles."""
    m, a0 = P.stall_sharpness, P.stall_angle
    assert stall_blend(0.0, m, a0) < 1e-9
    assert np.isclose(stall_blend(a0, m, a0), 0.5)
    assert stall_blend(np.radians(45.0), m, a0) > 1 - 1e-6  # 1 - 1.4e-7 here
    for alpha in (0.1, 0.4, 0.5, 1.0):
        assert np.isclose(stall_blend(-alpha, m, a0), stall_blend(alpha, m, a0))


def test_lift_falls_past_the_stall():
    """Past the stall the wing behaves as a flat plate: lift drops, and a plate
    broadside to the air (90 deg) makes no lift at all."""
    assert lift_coefficient(P, np.radians(35.0)) < lift_coefficient(P, np.radians(20.0))
    assert np.isclose(lift_coefficient(P, np.radians(90.0)), 0.0)


def test_drag_polar():
    """C_D - C_D_p is the induced drag C_L_lin^2 / (pi e AR), smallest (zero) where the
    linear lift vanishes, alpha = -C_L_0 / C_L_alpha."""
    k = 1.0 / (np.pi * P.oswald * P.aspect_ratio)
    for alpha in np.radians([-4.0, 0.0, 6.0]):
        linear = P.c_lift_0 + P.c_lift_alpha * alpha
        assert np.isclose(drag_coefficient(P, alpha) - P.c_drag_p, k * linear**2)
    zero_lift = -P.c_lift_0 / P.c_lift_alpha
    assert np.isclose(drag_coefficient(P, zero_lift), P.c_drag_p)


# --- Stability and damping ------------------------------------------------------------


def test_static_pitch_stability():
    """C_m_alpha < 0: more angle of attack gives a nose-down moment, which brings the
    nose back. Exactly qbar S c C_m_alpha per radian."""
    _, low = loads(2.0)
    _, high = loads(4.0)
    slope = (high[1] - low[1]) / np.radians(2.0)
    assert slope < 0.0
    assert np.isclose(slope, qbar_s() * P.chord * P.c_m_alpha)


def test_weathercock_stability_and_dihedral_effect():
    """Air from the right (beta > 0): the side force pushes left (C_Y_beta < 0), the
    nose yaws right into the air like a weathercock (C_n_beta > 0), and the aircraft
    rolls away from the slip (C_ell_beta < 0)."""
    force, moment = loads(0.0, 5.0)
    assert force[1] < 0.0
    assert moment[2] > 0.0
    assert moment[0] < 0.0


@pytest.mark.parametrize(
    "rates, axis, derivative, length",
    [
        ((0.5, 0.0, 0.0), 0, "c_ell_p", "span"),  # roll damping
        ((0.0, 0.5, 0.0), 1, "c_m_q", "chord"),  # pitch damping
        ((0.0, 0.0, 0.5), 2, "c_n_r", "span"),  # yaw damping
    ],
)
def test_rotation_is_damped(rates, axis, derivative, length):
    """A positive rate about an axis gives a moment against it, of exactly
    qbar S l C_x * (rate l / 2V) (l = b for roll and yaw, c for pitch)."""
    _, still = loads()
    _, turning = loads(rates=rates)
    ref = getattr(P, length)
    expected = qbar_s() * ref * getattr(P, derivative) * rates[axis] * ref / (2 * V)
    assert turning[axis] - still[axis] < 0.0
    assert np.isclose(turning[axis] - still[axis], expected)


# --- Controls -------------------------------------------------------------------------


def test_control_surface_signs():
    """Elevator > 0 pitches the nose down, aileron > 0 rolls right, rudder > 0 yaws
    the nose left; each by exactly qbar S l C_x_delta * delta."""
    delta = np.radians(5.0)
    _, base = loads()
    _, elevator = loads(elevator=delta)
    _, aileron = loads(aileron=delta)
    _, rudder = loads(rudder=delta)
    q_sb, q_sc = qbar_s() * P.span, qbar_s() * P.chord
    assert np.isclose(elevator[1] - base[1], q_sc * P.c_m_elevator * delta)
    assert elevator[1] - base[1] < 0.0
    assert np.isclose(aileron[0] - base[0], q_sb * P.c_ell_aileron * delta)
    assert aileron[0] - base[0] > 0.0
    assert np.isclose(rudder[2] - base[2], q_sb * P.c_n_rudder * delta)
    assert rudder[2] - base[2] < 0.0


def test_mirror_symmetry():
    """The aircraft is symmetric about its x-z plane: sideslip to the other side flips
    the side force, the roll and the yaw, and leaves lift, drag and pitch alone."""
    force_right, moment_right = loads(3.0, 6.0)
    force_left, moment_left = loads(3.0, -6.0)
    assert np.allclose(force_left * [1, -1, 1], force_right)
    assert np.allclose(moment_left * [-1, 1, -1], moment_right)


# --- Profile and validation -----------------------------------------------------------


def test_small_fixed_wing_profile():
    assert np.isclose(P.aspect_ratio, 2.8956**2 / 0.55)  # ~15.2, a glider-like wing
    assert P.inertia[0, 2] < 0.0  # -J_xz off the diagonal
    with pytest.raises(dataclasses.FrozenInstanceError):
        P.mass = 5.0


@pytest.mark.parametrize(
    "change",
    [
        {"mass": 0.0},
        {"wing_area": -0.5},
        {"chord": 0.0},
        {"oswald": 1.2},
        {"stall_angle": 0.0},
    ],
)
def test_rejects_invalid_parameters(change):
    with pytest.raises(ValueError):
        dataclasses.replace(P, **change)
