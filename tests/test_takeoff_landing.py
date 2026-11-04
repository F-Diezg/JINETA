"""Tests for take-off and landing: the autopilot's landed state and land detection.

What a correct landed state guarantees, checked exactly where it can be:
- waiting on the ground leaves nothing behind: a take-off after 1 s or after 5 s
  on the floor is the same flight (to the last bit), however hard the vehicle was
  told to push down meanwhile;
- the take-off is the same climb as the same command given in the air, once the
  feet have left the floor;
- a landing is detected from what a real autopilot knows (its velocity and its
  own thrust), never in the air.
"""

import numpy as np
import pytest

from jineta.core import quaternion
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.contact import GroundContact
from jineta.jinete.control import Autopilot, LandDetection
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_LANDING_GEAR

DT = 0.01
CLIMB = [0.0, 0.0, -1.0]  # 1 m/s up (NED: up is -z)
DESCEND = [0.0, 0.0, 0.5]  # 0.5 m/s down


def resting_state():
    y = np.zeros(13)
    y[6] = 1.0
    y[2] = -0.055 + 0.001  # corners 1 mm into the floor
    return y


def airborne_state(altitude):
    y = np.zeros(13)
    y[6] = 1.0
    y[2] = -altitude
    return y


def fly(schedule, t_end, start=None, landed=True, contact=True):
    """Fly an Iris with its autopilot; `schedule(t)` gives the velocity command.
    Returns times, states and the autopilot's `landed` flag at every step."""
    vehicle = Multirotor(
        IRIS, contact=GroundContact(IRIS_LANDING_GEAR) if contact else None
    )
    autopilot = Autopilot(vehicle, IRIS_GAINS, DT, landed=landed)
    flags = []

    def pre_step(t, y):
        autopilot.velocity_command = np.array(schedule(t), dtype=float)
        autopilot(t, y)
        flags.append(autopilot.landed)

    start = resting_state() if start is None else start
    times, states = run(
        vehicle.derivative,
        start,
        DT,
        t_end,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    return times, states, np.array(flags), autopilot


# ---- land detection settings --------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(takeoff_speed=0.0), "takeoff_speed"),
        (dict(max_speed=-0.1), "max_speed"),
        (dict(thrust_fraction=1.0), "thrust_fraction"),
        (dict(hold_time=0.0), "hold_time"),
    ],
)
def test_land_detection_rejects_invalid_settings(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LandDetection(**kwargs)


def test_every_gain_set_has_a_land_detection():
    assert IRIS_GAINS.land == LandDetection()


# ---- on the ground ------------------------------------------------------------------


@pytest.mark.parametrize("command", [[0.0, 0.0, 0.0], DESCEND, [0.0, 0.0, -0.1]])
def test_on_the_ground_the_motors_are_stopped_and_nothing_moves(command):
    """Hold still, push down, or climb slower than the take-off speed: the vehicle
    stays landed, motors stopped, integrators at zero, exactly at rest."""
    _, states, flags, autopilot = fly(lambda t: command, 3.0)
    assert flags.all()
    assert not autopilot.vehicle.rotor_speed_command.any()
    assert not autopilot.velocity_loop.integral.any()
    assert not autopilot.rate_loop.integral.any()
    assert np.abs(states - resting_state()).max() < 1e-12


def test_waiting_on_the_ground_leaves_nothing_behind():
    """Told to push down for 1 s or for 5 s, then to climb: the two flights after
    the climb command are identical. Without the landed state, 3 s pressed against
    the floor charged the vertical integrator with 3 m/s^2 and the climb was still
    15 % slow 3 s later (measured while designing this step)."""

    def after(wait):
        times, states, _, _ = fly(
            lambda t: DESCEND if t < wait - 1e-9 else CLIMB, wait + 3
        )
        return states[np.searchsorted(times, wait - 1e-9) :]

    short, long = after(1.0), after(5.0)
    assert np.array_equal(short, long[: len(short)])


def test_the_take_off_is_the_same_climb_as_in_the_air():
    """From the floor or from a hover in the air, the same 1 m/s climb command
    gives the same vertical speed once the feet are off the floor; the take-off is
    straight up (no tilt, no drift) and never dips below the resting height."""
    times, ground, flags, _ = fly(lambda t: CLIMB, 4.0)
    _, air, _, _ = fly(
        lambda t: CLIMB, 4.0, airborne_state(5.0), landed=False, contact=False
    )
    difference = np.abs(ground[:, 5] - air[:, 5])
    assert difference[times > 0.3].max() < 0.02
    assert difference[times > 1.0].max() < 0.005
    assert not flags[1:].any()
    assert ground[:, 2].max() <= resting_state()[2] + 1e-12
    assert np.abs(ground[:, 0:2]).max() < 1e-9
    tilt = [np.abs(quaternion.to_euler(q)[:2]).max() for q in ground[:, 6:10]]
    assert max(tilt) < 1e-9


# ---- landing -----------------------------------------------------------------------


def test_a_landing_is_detected_and_the_motors_stop():
    """Descending at 0.5 m/s from 1 m: it touches the floor at 2 s, does not
    bounce, the landing is detected in less than 2 s, and it ends exactly at rest
    with its motors stopped."""
    times, states, flags, autopilot = fly(
        lambda t: DESCEND, 6.0, airborne_state(1.0), landed=False
    )
    touch = times[np.argmax(-states[:, 2] < 0.055)]
    assert touch == pytest.approx(2.0, abs=0.05)
    landed_at = times[np.argmax(flags)]
    assert 0.0 < landed_at - touch < 2.0
    assert flags[np.searchsorted(times, landed_at) :].all()
    assert (-states[times > touch, 2]).max() < 0.055 + 0.005  # no bounce
    assert not autopilot.vehicle.rotor_speed_command.any()
    assert np.abs(states[-1] - resting_state()).max() < 1e-9


def test_a_take_off_after_a_landing_is_a_clean_take_off():
    """One autopilot, one flight: descend from 1 m until landed (pressing on the
    floor charges the vertical integrator before the landing is detected), wait,
    then climb. The landed state must clear what was charged, so the climb is the
    same as a take-off from a fresh start."""
    times, states, flags, _ = fly(
        lambda t: DESCEND if t < 5.0 - 1e-9 else CLIMB,
        8.0,
        airborne_state(1.0),
        landed=False,
    )
    start = np.searchsorted(times, 5.0 - 1e-9)
    assert flags[start - 1]
    climb = states[start:]
    fresh = fly(lambda t: CLIMB, 3.0)[1]
    np.testing.assert_allclose(climb[:, 3:], fresh[:, 3:], atol=1e-6)


def test_an_autopilot_told_it_is_flying_finds_out_it_is_on_the_floor():
    """Started on the floor as if airborne and asked to descend, it presses on the
    floor and detects the landing within 2 s. Asked to hold still instead, it
    really does hover (a few millimetres up), so no landing is the right answer."""
    times, _, flags, _ = fly(lambda t: DESCEND, 3.0, landed=False)
    assert flags[-1]
    assert times[np.argmax(flags)] < 2.0
    times, states, flags, _ = fly(lambda t: [0.0, 0.0, 0.0], 3.0, landed=False)
    assert not flags.any()
    assert -states[-1, 2] > 0.055  # the corners are off the floor


@pytest.mark.parametrize(
    "schedule",
    [
        lambda t: [0.0, 0.0, 0.0],  # hover
        lambda t: [5.0, 0.0, 0.0] if t < 3 else [0.0, 0.0, 0.0],  # cruise and brake
        lambda t: [0.0, 0.0, 2.0] if t < 1 else [0.0, 0.0, 0.0],  # drop and stop
        lambda t: [0.0, 0.0, 5.0] if t < 2 else [0.0, 0.0, 0.0],  # fast descent
    ],
)
def test_no_landing_is_ever_detected_in_the_air(schedule):
    """Including a fast descent: asked to go down at 5 m/s, the thrust stays below
    70 % of the weight for most of a second while it speeds up, so it is the speed
    condition that keeps this from looking like a landing."""
    _, _, flags, _ = fly(schedule, 10.0, airborne_state(20.0), landed=False)
    assert not flags.any()


def test_take_off_land_and_take_off_again():
    """Climb 2 s, hover 2 s, descend until landed, wait, climb again."""

    def schedule(t):
        if t < 1:
            return [0.0, 0.0, 0.0]
        if t < 3:
            return CLIMB
        if t < 5:
            return [0.0, 0.0, 0.0]
        if t < 11:
            return DESCEND
        return CLIMB

    times, states, flags, _ = fly(schedule, 13.0)

    def at(t):
        return flags[np.searchsorted(times, t)]

    assert at(0.5) and not at(4.0) and at(10.5) and not at(12.5)
    assert -states[np.searchsorted(times, 4.0), 2] > 1.5
    assert -states[-1, 2] > 1.0
