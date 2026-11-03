"""Tests for the floor contact against closed-form results.

- At rest on its four corners the Iris sinks exactly m g / (4 k) per corner: the
  state is an equilibrium (derivative zero to rounding).
- A point dropped on the floor follows, while it touches, the linear damped
  oscillator m x'' = m g - k x - c x' (x = depth, down positive), whose solution is
  closed form; it leaves when the spring and damper stop pushing (k x + c x' = 0).
  The rebound height then follows from the exit depth and speed.
- Sliding, the friction gives dv/dt = -mu g v / sqrt(v^2 + v_s^2), whose solution
  is closed form too: t = (F(v0) - F(v)) / (mu g), with
  F(v) = sqrt(v^2 + v_s^2) - v_s ln((v_s + sqrt(v^2 + v_s^2)) / v).
- RK4 at 100 Hz is stable on the ground: every mode of the resting Iris has an
  amplification |R(lambda dt)| <= 1, R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.core.rigid_body import RigidBody, normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.contact import (
    GroundContact,
    LandingGear,
    box_bottom_corners,
    contact_forces,
)
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR, MULTIROTOR_PROFILES
from jineta.jinete.stability import jacobian

DT = 0.01
HALF_HEIGHT = 0.055  # the Iris rests 0.055 m below its centre of mass


def state(position, velocity=(0, 0, 0), q=(1, 0, 0, 0), omega=(0, 0, 0)):
    return np.concatenate([position, velocity, q, omega]).astype(float)


def iris_on_the_floor(gear=IRIS_LANDING_GEAR):
    """An Iris with its motors off, standing on the floor z = 0."""
    return Multirotor(IRIS, contact=GroundContact(gear))


def resting_state():
    sink = IRIS.mass * GRAVITY / (4 * IRIS_LANDING_GEAR.stiffness)
    return state([0.0, 0.0, -HALF_HEIGHT + sink])


def rk4_amplification(z):
    return abs(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)


# ---- the gear and the forces --------------------------------------------------------


def test_box_bottom_corners_are_below_the_center_of_mass():
    corners = box_bottom_corners([0.47, 0.47, 0.11])
    assert corners.shape == (4, 3)
    np.testing.assert_allclose(np.abs(corners), [[0.235, 0.235, 0.055]] * 4)
    assert np.all(corners[:, 2] > 0)  # z down: below the centre of mass
    assert {tuple(np.sign(c[:2])) for c in corners} == {
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    }


def test_gear_from_damping_ratio():
    """n k sink = m g, and the vertical bounce has the damping ratio asked for."""
    gear = LandingGear.from_damping_ratio(
        box_bottom_corners([0.47, 0.47, 0.11]), 1.52, 0.001, 0.5, 1.0, 0.2
    )
    assert 4 * gear.stiffness * 0.001 == pytest.approx(1.52 * GRAVITY)
    zeta = 4 * gear.damping / (2 * np.sqrt(4 * gear.stiffness * 1.52))
    assert zeta == pytest.approx(0.5)
    assert gear.stiffness == pytest.approx(3727, abs=1)  # the profile's comment


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(points=[[0, 0, 0.1]], stiffness=0.0), "stiffness"),
        (dict(points=[[0, 0, 0.1]], damping=-1.0), "damping"),
        (dict(points=[[0, 0, 0.1]], friction=-0.1), "friction"),
        (dict(points=[[0, 0, 0.1]], slip_speed=0.0), "slip_speed"),
        (dict(points=[[0, 0]]), "shape"),
        (dict(points=np.zeros((0, 3))), "shape"),
        (dict(points=[[0, np.nan, 0]]), "finite"),
    ],
)
def test_gear_rejects_invalid_input(kwargs, match):
    values = dict(stiffness=1000.0, damping=10.0, friction=1.0, slip_speed=0.1)
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        LandingGear(**values)


def test_gear_is_read_only():
    with pytest.raises(ValueError):
        IRIS_LANDING_GEAR.points[0, 0] = 1.0


ONE_POINT = LandingGear(
    points=[[0.0, 0.0, 0.0]],
    stiffness=1000.0,
    damping=10.0,
    friction=0.5,
    slip_speed=0.1,
)


def test_no_force_above_the_floor_or_on_it():
    for z in (-0.5, 0.0):
        force, moment = contact_forces(ONE_POINT, state([0, 0, z], velocity=(1, 0, 2)))
        assert not force.any() and not moment.any()


def test_normal_force_is_spring_plus_damper_and_never_pulls():
    sinking = contact_forces(ONE_POINT, state([0, 0, 0.002], velocity=(0, 0, 0.1)))[0]
    np.testing.assert_allclose(sinking, [0, 0, -(1000 * 0.002 + 10 * 0.1)])
    still = contact_forces(ONE_POINT, state([0, 0, 0.002]))[0]
    np.testing.assert_allclose(still, [0, 0, -2.0])
    leaving_fast = contact_forces(ONE_POINT, state([0, 0, 0.002], velocity=(0, 0, -1)))
    assert not leaving_fast[0].any()  # k x + c v = 2 - 10 < 0: no pull


@pytest.mark.parametrize("speed", [0.05, 0.1, 1.0, 10.0])
def test_friction_opposes_the_slide_with_at_most_mu_n(speed):
    direction = np.array([0.6, -0.8])
    force = contact_forces(
        ONE_POINT, state([0, 0, 0.002], velocity=(*(speed * direction), 0.0))
    )[0]
    normal = 2.0
    expected = 0.5 * normal * speed / np.sqrt(speed**2 + 0.1**2)
    np.testing.assert_allclose(force[:2], -expected * direction, atol=1e-12)
    assert np.linalg.norm(force[:2]) < 0.5 * normal


def test_forces_come_back_in_body_axes_with_their_moment():
    """The Iris banked 10 deg right (right wing down) with only its two right
    corners touching: the floor pushes up (along -z NED), which in body axes has a
    component along -y as well, and the moment about the centre of mass rolls it
    back left (negative roll)."""
    q = quaternion.from_euler(np.radians(10.0), 0.0, 0.0)
    right_corner = IRIS_LANDING_GEAR.points[0]  # front right, (+, +, +)
    rotation = quaternion.to_matrix(q)
    height = (rotation @ right_corner)[2]
    s = state([0.0, 0.0, 0.001 - height], q=q)  # the right corners 1 mm deep
    force, moment = contact_forces(IRIS_LANDING_GEAR, s)
    np.testing.assert_allclose(
        rotation @ force, [0, 0, -2 * IRIS_LANDING_GEAR.stiffness * 0.001], atol=1e-9
    )
    assert moment[0] < 0  # rolls back left
    # By hand: in body axes each push is N (0, -sin phi, -cos phi), applied at
    # (+-0.235, 0.235, 0.055); r x f summed over the front and back corner gives
    # 2 N (-0.235 cos phi + 0.055 sin phi) in roll and nothing in pitch or yaw.
    n = IRIS_LANDING_GEAR.stiffness * 0.001
    c, s_ = np.cos(np.radians(10.0)), np.sin(np.radians(10.0))
    np.testing.assert_allclose(
        moment, [2 * n * (-0.235 * c + 0.055 * s_), 0.0, 0.0], atol=1e-12
    )


def test_a_spinning_iris_is_braked_by_the_friction_of_its_four_feet():
    """Resting on the floor with a heading of 30 deg and spinning at 1 rad/s about
    the vertical: each foot slides at s = omega r_c along its circle, so the floor
    pushes nothing sideways overall and brakes the spin with
    M_z = -4 r_c mu N s / sqrt(s^2 + v_s^2), N = m g / 4 per foot."""
    q = quaternion.from_euler(0.0, 0.0, np.radians(30.0))
    s = resting_state()
    s[6:10] = q
    s[12] = 1.0
    force, moment = contact_forces(IRIS_LANDING_GEAR, s)
    r_c = 0.235 * np.sqrt(2)
    slide = 1.0 * r_c
    n = IRIS.mass * GRAVITY / 4
    expected = -4 * r_c * 1.0 * n * slide / np.sqrt(slide**2 + 0.2**2)
    np.testing.assert_allclose(force, [0.0, 0.0, -4 * n], atol=1e-9)
    np.testing.assert_allclose(moment, [0.0, 0.0, expected], atol=1e-9)


# ---- resting, stability ----------------------------------------------------------


def test_iris_at_rest_sinks_mg_over_4k_and_stays():
    vehicle = iris_on_the_floor()
    derivative = vehicle.derivative(0.0, resting_state())
    assert np.abs(derivative).max() < 1e-12
    sink = IRIS.mass * GRAVITY / (4 * IRIS_LANDING_GEAR.stiffness)
    assert sink == pytest.approx(0.001)


def test_iris_put_down_touching_settles_to_the_resting_depth_at_100_hz():
    vehicle = iris_on_the_floor()
    _, states = run(
        vehicle.derivative,
        state([0.0, 0.0, -HALF_HEIGHT]),
        DT,
        2.0,
        post_step=normalize_attitude,
    )
    np.testing.assert_allclose(states[-1], resting_state(), atol=1e-6)


def test_every_mode_at_rest_is_stable_with_rk4_at_100_hz():
    vehicle = iris_on_the_floor()
    modes = np.linalg.eigvals(
        jacobian(lambda y: vehicle.derivative(0.0, y), resting_state())
    )
    assert max(rk4_amplification(lam * DT) for lam in modes) <= 1.0 + 1e-9


@pytest.mark.parametrize("change", [dict(damping_ratio=0.7), dict(slip_speed=0.1)])
def test_a_little_more_damping_or_less_slip_speed_is_unstable_at_100_hz(change):
    """The gear sits inside the stability limit, not far from it: this is why
    its damping ratio and slip speed are what they are."""
    values = dict(
        points=box_bottom_corners([0.47, 0.47, 0.11]),
        mass=1.52,
        sink=0.001,
        damping_ratio=0.5,
        friction=1.0,
        slip_speed=0.2,
    )
    values.update(change)
    vehicle = iris_on_the_floor(LandingGear.from_damping_ratio(**values))
    modes = np.linalg.eigvals(
        jacobian(lambda y: vehicle.derivative(0.0, y), resting_state())
    )
    assert max(rk4_amplification(lam * DT) for lam in modes) > 1.0


def test_the_iris_profile_carries_its_gear():
    assert MULTIROTOR_PROFILES["iris"].gear is IRIS_LANDING_GEAR


# ---- a bounce against its closed form -----------------------------------------------


def exact_bounce(mass, k, c, impact_speed):
    """Exit time, depth and speed (down positive) of a point mass that hits the
    floor at `impact_speed`, from the closed-form damped oscillator around the
    resting depth x_s = m g / k, ending when k x + c x' = 0."""
    omega = np.sqrt(k / mass)
    zeta = c / (2 * np.sqrt(k * mass))
    omega_d = omega * np.sqrt(1 - zeta**2)
    x_s = mass * GRAVITY / k
    a = -x_s
    b = (impact_speed + zeta * omega * a) / omega_d

    def depth(t):
        return x_s + np.exp(-zeta * omega * t) * (
            a * np.cos(omega_d * t) + b * np.sin(omega_d * t)
        )

    def speed(t):
        e = np.exp(-zeta * omega * t)
        return e * (
            (-zeta * omega * a + omega_d * b) * np.cos(omega_d * t)
            + (-zeta * omega * b - omega_d * a) * np.sin(omega_d * t)
        )

    def push(t):
        return k * depth(t) + c * speed(t)

    # The push starts at 0 (depth 0, speed down) and must fall back to 0 within
    # half a period: find where it changes sign.
    t_grid = np.linspace(1e-6, np.pi / omega_d, 2000)
    i = np.argmax(np.array([push(t) for t in t_grid]) < 0)
    t_exit = brentq(push, t_grid[i - 1], t_grid[i])
    return t_exit, depth(t_exit), speed(t_exit), zeta


def test_a_drop_bounces_as_the_closed_form_says():
    """A point mass (contact at its centre of mass, no rotation) dropped from
    0.2 m: the rebound height and the contact time match the exact solution, and
    the restitution is above the textbook exp(-zeta pi / sqrt(1 - zeta^2)), which
    assumes a damper that also pulls on the way out (ours never pulls, so the body
    leaves earlier and faster)."""
    mass, k, c = 1.0, 4000.0, 20.0
    gear = LandingGear(
        [[0.0, 0.0, 0.0]], stiffness=k, damping=c, friction=0.0, slip_speed=0.1
    )
    body = RigidBody(mass, np.eye(3) * 0.01, GroundContact(gear))
    drop = 0.2
    impact = np.sqrt(2 * GRAVITY * drop)
    t_exit, x_exit, v_exit, zeta = exact_bounce(mass, k, c, impact)
    rebound = -x_exit + v_exit**2 / (2 * GRAVITY)

    # Below the floor (z > 0) from the impact until it rises back through z = 0:
    # the exact exit (no more push) comes a little before, still x_exit deep, and
    # from there it flies freely, x_exit + v_exit s + g s^2 / 2 = 0.
    s_up = (-v_exit - np.sqrt(v_exit**2 - 2 * GRAVITY * x_exit)) / GRAVITY
    below_floor = t_exit + s_up

    times, states = run(body.derivative, state([0, 0, -drop]), 1e-4, 0.5)
    z = states[:, 2]
    first_in = np.argmax(z > 0)
    first_out = first_in + np.argmax(z[first_in:] <= 0)
    next_in = first_out + np.argmax(z[first_out:] > 0)
    simulated_rebound = -z[first_out:next_in].min()
    assert simulated_rebound == pytest.approx(rebound, rel=2e-3)
    assert times[first_out] - times[first_in] == pytest.approx(below_floor, abs=3e-4)

    restitution = -v_exit / impact
    textbook = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    assert restitution > textbook


# ---- sliding ----------------------------------------------------------------------


def slide_time(v0, v, mu, v_s):
    def f(u):
        root = np.sqrt(u**2 + v_s**2)
        return root - v_s * np.log((v_s + root) / u)

    return (f(v0) - f(v)) / (mu * GRAVITY)


def test_a_slide_brakes_as_the_closed_form_says():
    """The Iris pushed along the floor at 3 m/s, motors off: it slows to 1 m/s in
    the closed-form time of dv/dt = -mu g v / sqrt(v^2 + v_s^2), which for
    v >> v_s is the classic constant deceleration mu g."""
    vehicle = iris_on_the_floor()
    start = resting_state()
    start[3] = 3.0
    times, states = run(
        vehicle.derivative, start, DT, 0.5, post_step=normalize_attitude
    )
    speed = states[:, 3]
    t1 = np.interp(-1.0, -speed, times)  # first time at 1 m/s (speed decreasing)
    expected = slide_time(3.0, 1.0, 1.0, 0.2)
    assert t1 == pytest.approx(expected, rel=0.01)
    assert expected == pytest.approx(2.0 / GRAVITY, rel=0.02)  # nearly mu g


# ---- a rough landing at 100 Hz ------------------------------------------------------


def test_a_rough_landing_at_100_hz_ends_at_rest_and_level():
    """Dropped from 0.5 m, banked 15 deg, pitched 10 deg, moving at 2 m/s and
    spinning at 2 rad/s: it hits the floor, slides, rights itself and stops
    standing on its four corners, without ever sinking more than a few cm."""
    vehicle = iris_on_the_floor()
    q = quaternion.from_euler(np.radians(15.0), np.radians(10.0), 0.0)
    start = state([0.0, 0.0, -0.5], velocity=(2.0, 0.0, 0.0), q=q, omega=(0, 0, 2.0))
    _, states = run(vehicle.derivative, start, DT, 4.0, post_step=normalize_attitude)
    assert np.all(np.isfinite(states))
    assert states[:, 2].max() < -HALF_HEIGHT + 0.03
    end = states[-1]
    roll, pitch, _ = quaternion.to_euler(end[6:10])
    assert abs(roll) < np.radians(0.1) and abs(pitch) < np.radians(0.1)
    assert end[2] == pytest.approx(resting_state()[2], abs=1e-5)
    assert np.linalg.norm(end[3:6]) < 1e-3
    assert np.linalg.norm(end[10:13]) < 1e-3
