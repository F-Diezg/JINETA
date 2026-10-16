"""Tests for the linearisation and the dynamic modes of the small fixed-wing UAV.

The modes of the full 6-DOF model (eigenvalues of the numerical Jacobian at the
trim) are compared with the classic approximations of flight mechanics, each built
from a few stability derivatives (Etkin & Reid, Dynamics of Flight, ch. 6;
Beard & McLain ch. 5):

- short period, the (alpha, q) pair at constant speed:
  wn^2 = Z_a M_q / V - M_a,  2 zeta wn = -(M_q + Z_a / V);
- phugoid, Lanchester: wn = sqrt(2) g / V (speed and height only, constant alpha).
  It overestimates wn by ~10 % here; letting the short period settle quasi-
  statically gives the refined wn = sqrt(2) g / V * sqrt(-M_a / wn_sp^2).
  Damping zeta = -X_u / (2 wn), X_u = (dT/dV - 2 D / V) / m;
- roll subsidence: lambda = L_p (roll damping alone);
- Dutch roll, yaw and sideslip only: wn^2 = N_b + Y_b N_r / V,
  2 zeta wn = -(Y_b / V + N_r) (rough: it ignores the roll coupling);
- spiral: stable only if C_ell_beta C_n_r > C_ell_r C_n_beta.

Z, M, X, N, L, Y are the forces and moments per unit mass or inertia per unit of
alpha, beta, rate or speed. Constant sea-level density, so altitude is neutral.
"""

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.core.environment import GRAVITY, sea_level_density
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.fixed_wing import FixedWing, drag_coefficient
from jineta.jinete.profiles import SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION
from jineta.jinete.propulsion import propeller_thrust_torque
from jineta.jinete.stability import (
    Mode,
    classify_modes,
    dynamic_modes,
    jacobian,
)
from jineta.jinete.trim import trim_level_flight

P = SMALL_FIXED_WING
PROP = SMALL_FIXED_WING_PROPULSION
RHO = sea_level_density(0.0)
SPEEDS = [18.0, 25.0, 30.0]


def trimmed(speed):
    plane = FixedWing(P, PROP)
    trim = trim_level_flight(plane, speed)
    return plane, trim


def modes_at(speed):
    plane, trim = trimmed(speed)
    matrix = jacobian(lambda s: plane.derivative(0.0, s), trim.state)
    return classify_modes(dynamic_modes(matrix))


class Derivatives:
    """Dimensional stability derivatives at a trim (per unit mass or inertia)."""

    def __init__(self, speed):
        _, trim = trimmed(speed)
        self.speed = speed
        qbar_s = 0.5 * RHO * speed**2 * P.wing_area
        c, b = P.chord, P.span
        ix, iy, iz = np.diag(P.inertia)
        c_drag = drag_coefficient(P, trim.alpha) + P.c_drag_elevator * (
            trim.surfaces.elevator
        )
        self.z_alpha = -qbar_s * (P.c_lift_alpha + c_drag) / P.mass
        self.m_alpha = qbar_s * c * P.c_m_alpha / iy
        self.m_q = qbar_s * c * P.c_m_q * c / (2 * speed) / iy
        axial = speed * np.cos(trim.alpha)
        h = 1e-4
        thrust_slope = (
            propeller_thrust_torque(PROP, axial + h, trim.throttle, RHO).thrust
            - propeller_thrust_torque(PROP, axial - h, trim.throttle, RHO).thrust
        ) / (2 * h)
        self.drag_slope = 2 * qbar_s * c_drag / speed  # dD/dV at constant alpha
        self.x_u = (thrust_slope - self.drag_slope) / P.mass
        self.l_p = qbar_s * b * P.c_ell_p * b / (2 * speed) / ix
        self.n_beta = qbar_s * b * P.c_n_beta / iz
        self.n_r = qbar_s * b * P.c_n_r * b / (2 * speed) / iz
        self.y_beta = qbar_s * P.c_y_beta / P.mass

    @property
    def short_period(self):
        wn = np.sqrt(self.z_alpha * self.m_q / self.speed - self.m_alpha)
        zeta = -(self.m_q + self.z_alpha / self.speed) / (2 * wn)
        return wn, zeta

    @property
    def lanchester(self):
        return np.sqrt(2.0) * GRAVITY / self.speed

    @property
    def phugoid(self):
        wn_sp, _ = self.short_period
        wn = self.lanchester * np.sqrt(-self.m_alpha / wn_sp**2)
        return wn, -self.x_u / (2 * wn)

    @property
    def dutch_roll(self):
        wn = np.sqrt(self.n_beta + self.y_beta * self.n_r / self.speed)
        zeta = -(self.y_beta / self.speed + self.n_r) / (2 * wn)
        return wn, zeta


# ---- the numerical tools ---------------------------------------------------------


def test_jacobian_of_a_linear_map_is_its_matrix():
    rng = np.random.default_rng(7)
    matrix = rng.normal(size=(4, 4))
    x = rng.normal(size=4) * 100.0
    assert np.allclose(jacobian(lambda v: matrix @ v, x), matrix, atol=1e-8)


def test_jacobian_of_a_nonlinear_function():
    def f(v):
        return np.array([np.sin(v[0]) * v[1], v[0] ** 2 + np.exp(v[1])])

    x = np.array([0.7, -0.3])
    exact = np.array([[np.cos(0.7) * -0.3, np.sin(0.7)], [1.4, np.exp(-0.3)]])
    assert np.allclose(jacobian(f, x), exact, atol=1e-9)


def test_mass_spring_damper_mode():
    # x'' + 2 zeta wn x' + wn^2 x = 0 as a first-order system.
    wn, zeta = 3.0, 0.2
    matrix = np.array([[0.0, 1.0], [-(wn**2), -2 * zeta * wn]])
    (mode,) = dynamic_modes(matrix)
    assert np.isclose(mode.natural_frequency, wn, rtol=1e-12)
    assert np.isclose(mode.damping_ratio, zeta, rtol=1e-12)
    assert np.isclose(mode.period, 2 * np.pi / (wn * np.sqrt(1 - zeta**2)))


def test_real_modes_do_not_oscillate():
    decaying, growing = Mode(-2.0 + 0j, False), Mode(0.5 + 0j, True)
    assert not decaying.oscillates and decaying.period == np.inf
    assert decaying.damping_ratio == 1.0
    assert growing.damping_ratio == -1.0


def test_neutral_directions_are_left_out():
    assert dynamic_modes(np.zeros((3, 3))) == []


# ---- the aircraft ----------------------------------------------------------------


@pytest.mark.parametrize("speed", SPEEDS)
def test_short_period_matches_the_classic_approximation(speed):
    mode = modes_at(speed).short_period
    wn, zeta = Derivatives(speed).short_period
    assert np.isclose(mode.natural_frequency, wn, rtol=0.02)
    assert np.isclose(mode.damping_ratio, zeta, rtol=0.02)


@pytest.mark.parametrize("speed", SPEEDS)
def test_phugoid_frequency_lanchester_and_refined(speed):
    mode = modes_at(speed).phugoid
    derivatives = Derivatives(speed)
    wn, _ = derivatives.phugoid
    assert np.isclose(mode.natural_frequency, wn, rtol=0.01)
    # Lanchester alone is ~10 % high: it ignores the pitch stiffness.
    ratio = derivatives.lanchester / mode.natural_frequency
    assert 1.05 < ratio < 1.15


@pytest.mark.parametrize("speed", SPEEDS)
def test_phugoid_is_damped_mostly_by_the_propeller(speed):
    mode = modes_at(speed).phugoid
    derivatives = Derivatives(speed)
    _, zeta = derivatives.phugoid
    assert np.isclose(mode.damping_ratio, zeta, rtol=0.2)
    # With constant thrust only the drag would damp it: 3 to 4 times less.
    wn, _ = derivatives.phugoid
    zeta_constant_thrust = derivatives.drag_slope / P.mass / (2 * wn)
    assert mode.damping_ratio > 2.5 * zeta_constant_thrust


@pytest.mark.parametrize("speed", SPEEDS)
def test_roll_subsidence_is_the_roll_damping(speed):
    mode = modes_at(speed).roll
    assert np.isclose(mode.eigenvalue.real, Derivatives(speed).l_p, rtol=0.03)


@pytest.mark.parametrize("speed", SPEEDS)
def test_dutch_roll_roughly_matches_its_approximation(speed):
    # The two-degree-of-freedom approximation ignores the roll coupling: within
    # ~12 % in frequency and ~15 % in damping here.
    mode = modes_at(speed).dutch_roll
    wn, zeta = Derivatives(speed).dutch_roll
    assert np.isclose(mode.natural_frequency, wn, rtol=0.15)
    assert np.isclose(mode.damping_ratio, zeta, rtol=0.2)
    assert mode.damping_ratio > 0.0


@pytest.mark.parametrize("speed", SPEEDS)
def test_spiral_is_unstable_as_the_criterion_predicts(speed):
    criterion = P.c_ell_beta * P.c_n_r - P.c_ell_r * P.c_n_beta
    assert criterion < 0.0  # -0.0059: the profile is spirally unstable
    spiral = modes_at(speed).spiral
    assert spiral.eigenvalue.real > 0.0
    # Every other mode is stable.
    others = modes_at(speed)
    for mode in (others.short_period, others.phugoid, others.roll, others.dutch_roll):
        assert mode.eigenvalue.real < 0.0


def test_simulated_phugoid_matches_the_eigenvalue():
    # 0.5 m/s more speed at the 25 m/s trim: measure period and decay on the speed.
    plane, trim = trimmed(25.0)
    mode = modes_at(25.0).phugoid
    start = np.array(trim.state)
    start[3] += 0.5
    dt = 0.01
    times, states = run(plane.derivative, start, dt, 40.0, post_step=normalize_attitude)
    excess = np.linalg.norm(states[:, 3:6], axis=1) - 25.0
    # Period: the first zero crossings (interpolated) are half a period apart.
    k = np.flatnonzero(np.sign(excess[:-1]) != np.sign(excess[1:]))[:4]
    crossings = times[k] - excess[k] * dt / (excess[k + 1] - excess[k])
    period = 2.0 * np.mean(np.diff(crossings))
    assert np.isclose(period, mode.period, rtol=0.01)
    # Damping: two troughs one period apart shrink by e^(zeta wn T).
    troughs = np.flatnonzero((excess[1:-1] < excess[:-2]) & (excess[1:-1] < excess[2:]))
    first, second = excess[troughs[0] + 1], excess[troughs[1] + 1]
    decrement = np.log(first / second)
    zeta = decrement / np.sqrt(4 * np.pi**2 + decrement**2)
    assert np.isclose(zeta, mode.damping_ratio, rtol=0.03)


def test_simulated_spiral_diverges_at_the_eigenvalue_rate():
    # 0.5 deg of bank at the 25 m/s trim grows to ~7 deg in 30 s. Once the fast
    # modes have died out (after ~10 s) the bank grows like e^(lambda t).
    plane, trim = trimmed(25.0)
    rate = modes_at(25.0).spiral.eigenvalue.real
    start = np.array(trim.state)
    bank = quaternion.from_axis_angle([1.0, 0.0, 0.0], np.radians(0.5))
    start[6:10] = quaternion.multiply(start[6:10], bank)
    dt = 0.01
    _, states = run(plane.derivative, start, dt, 20.0, post_step=normalize_attitude)

    def roll_angle(q):
        w, x, y, z = q
        return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    early, late = (roll_angle(states[round(t / dt), 6:10]) for t in (10.0, 20.0))
    assert late > early > np.radians(0.5)
    assert np.isclose(np.log(late / early) / 10.0, rate, rtol=0.05)
