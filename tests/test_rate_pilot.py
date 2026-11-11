"""Tests for the rate pilot (collective thrust + body rates, the entry point of a
policy) and its landed state, in the reference and in the batch engine.

On the floor the rate loop must not integrate: the floor stops the vehicle from
turning, so a loop asked to turn would charge its integral against it and the
take-off would start with a kick. In the air the pilot is the rate loop itself.
"""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.core.environment import GRAVITY
from jineta.core.integrators import rk4_step
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.control import BatchRatePilot
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import batch_referee
from jineta.escuadron.rigid_body import normalize_attitudes
from jineta.escuadron.simulation import BatchFlight
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import RateController, RatePilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import (
    IRIS,
    IRIS_HITBOX,
    IRIS_LANDING_GEAR,
    IRIS_RATE_GAINS,
)

DT = 0.01  # 100 Hz
WEIGHT = IRIS.mass * GRAVITY
REST = resting_height(IRIS_LANDING_GEAR, IRIS.mass)
# Turning on the floor: the floor does not let it, so the error never goes away.
TURN = np.array([0.5, -0.5, 0.3])


def standing() -> np.ndarray:
    """The Iris standing on its gear, level, heading North."""
    y = np.zeros(13)
    y[2] = -REST
    y[6] = 1.0
    return y


def vehicle() -> Multirotor:
    return Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))


def fly(pilot, start, schedule, t_end):
    """Fly `pilot` (a RatePilot) from `start`; `schedule(t)` gives (thrust, rates).
    Returns the times, the states and `landed` after each step's call."""
    landed = []

    def pre_step(t, y):
        pilot.thrust_command, rates = schedule(t)
        pilot.rate_command = np.array(rates, dtype=float)
        pilot(t, y)
        landed.append(pilot.landed)

    times, states = run(
        pilot.vehicle.derivative,
        start,
        DT,
        t_end,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    return times, states, np.array(landed)


# ---- reference -----------------------------------------------------------------------


def test_on_the_floor_turning_charges_nothing():
    pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    integrals = []

    def schedule(t):
        integrals.append(pilot.rate_loop.integral.copy())
        return 0.5 * WEIGHT, TURN

    _, states, landed = fly(pilot, standing(), schedule, 2.0)
    assert np.all(landed)
    assert np.all(np.array(integrals) == 0.0)
    assert np.all(pilot.rate_loop.integral == 0.0)
    # Still standing: half its weight in thrust and small moments do not move it.
    assert np.max(np.abs(states[:, 2] + REST)) < 2e-3
    assert np.max(np.abs(states[:, 10:13])) < 0.05


def test_the_plain_rate_loop_on_the_floor_charges_its_integral():
    """Why the landed state is needed: the same 2 s with the loop alone."""
    model = vehicle()
    loop = RateController(IRIS, IRIS_RATE_GAINS, DT)

    def pre_step(t, y):
        model.rotor_speed_command = loop.update(TURN, 0.5 * WEIGHT, y)

    run(model.derivative, standing(), DT, 2.0, pre_step=pre_step)
    # Roll and pitch at their limits (10 rad/s^2) after 2 s; yaw turns a little.
    assert np.allclose(np.abs(loop.integral[0:2]), IRIS_RATE_GAINS.integral_limit[0:2])


def test_it_takes_off_when_the_thrust_exceeds_the_weight():
    ramp = 0.5  # thrust from 0.9 to 1.1 weights in 2 s, crossing the weight at 1 s

    def schedule(t):
        return (0.9 + 0.1 * t / ramp / 2.0) * WEIGHT, [0.0, 0.0, 0.0]

    pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    times, states, landed = fly(pilot, standing(), schedule, 2.0)
    first_flying = int(np.argmin(landed))
    thrusts = np.array([schedule(t)[0] for t in times[:-1]])
    assert thrusts[first_flying] > WEIGHT >= thrusts[first_flying - 1]
    assert not np.any(landed[first_flying:])
    assert -states[-1, 2] > REST + 0.05  # it really left the floor


def test_waiting_on_the_floor_leaves_nothing_behind():
    """1 s on the floor asked to turn, then a take-off; the take-off is the one a
    fresh pilot flies from the same state, bit for bit."""

    def schedule(t):
        if t < 1.0 - 1e-9:
            return 0.8 * WEIGHT, TURN
        return 1.3 * WEIGHT, [0.2, 0.1, 0.0]

    pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    times, states, landed = fly(pilot, standing(), schedule, 2.0)
    k = int(np.argmin(landed))  # the step of the take-off
    assert times[k] == pytest.approx(1.0)

    fresh = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    _, again, _ = fly(fresh, states[k], lambda t: schedule(t + times[k]), 1.0)
    np.testing.assert_array_equal(again, states[k:])


def test_in_the_air_it_is_the_rate_loop():
    def schedule(t):
        step = int(t / 0.25 + 1e-9)
        rng = np.random.default_rng(step)
        return rng.uniform(0.6, 1.4) * WEIGHT, rng.normal(size=3)

    start = standing()
    start[2] = -5.0
    pilot = RatePilot(Multirotor(IRIS), IRIS_RATE_GAINS, DT)
    _, states, landed = fly(pilot, start, schedule, 1.5)
    assert not np.any(landed)

    model = Multirotor(IRIS)
    loop = RateController(IRIS, IRIS_RATE_GAINS, DT)

    def pre_step(t, y):
        thrust, rates = schedule(t)
        model.rotor_speed_command = loop.update(rates, thrust, y)

    _, expected = run(
        model.derivative,
        start,
        DT,
        1.5,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    np.testing.assert_array_equal(states, expected)


def test_it_never_lands_by_itself():
    def schedule(t):
        return (1.2 if t < 0.5 else 0.0) * WEIGHT, [0.0, 0.0, 0.0]

    pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    _, states, landed = fly(pilot, standing(), schedule, 3.0)
    assert not np.any(landed)  # 1.2 weights from the first step: off at once
    assert -states[-1, 2] < REST + 0.01  # it came back down onto the floor


# ---- batch ---------------------------------------------------------------------------


def scenario(i: int):
    """Drone i's commands: thrusts and rates changing every 0.5 s, some below the
    weight (on the floor), some above (taking off at different moments)."""

    def schedule(t):
        step = int(t / 0.5 + 1e-9)
        rng = np.random.default_rng(100 * i + step)
        lift = 0.7 + 0.15 * (step + i % 3) + rng.uniform(-0.05, 0.05)
        return lift * WEIGHT, rng.normal(size=3) * 0.4

    return schedule


def batch_flight(schedules, starts, landed, t_end):
    """The schedules flown by one batch; returns the states (steps + 1, n, 13) and
    `landed` (steps, n)."""
    n = len(schedules)
    vehicles = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    pilot = BatchRatePilot(vehicles, IRIS_RATE_GAINS, DT, landed=landed)
    states = np.array(starts, dtype=float)
    history, landed_history = [states.copy()], []
    t = 0.0
    for _ in range(round(t_end / DT)):
        commands = [s(t) for s in schedules]
        pilot.thrust_command = np.array([c[0] for c in commands])
        pilot.rate_command = np.array([c[1] for c in commands])
        pilot(t, states)
        landed_history.append(pilot.landed.copy())
        states = normalize_attitudes(rk4_step(vehicles.derivative, t, states, DT))
        history.append(states.copy())
        t += DT
    return np.array(history), np.array(landed_history), pilot


def starts(n: int) -> np.ndarray:
    """Most drones standing on the floor, every fourth one 3 m up."""
    result = np.tile(standing(), (n, 1))
    result[::4, 2] = -3.0
    return result


def test_batch_matches_the_reference_drone_by_drone():
    n = 8
    schedules = [scenario(i) for i in range(n)]
    landed = np.ones(n, dtype=bool)
    landed[::4] = False  # the ones in the air
    history, landed_history, _ = batch_flight(schedules, starts(n), landed, 3.0)
    took_off = 0
    for i in range(n):
        pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=bool(landed[i]))
        _, expected, expected_landed = fly(pilot, starts(n)[i], schedules[i], 3.0)
        assert np.max(np.abs(history[:, i] - expected)) < 1e-9
        np.testing.assert_array_equal(landed_history[:, i], expected_landed)
        took_off += bool(expected_landed[0]) and not expected_landed[-1]
    assert took_off >= 4  # take-offs at different moments are tested


def test_a_drone_gives_the_same_numbers_alone_or_in_a_batch():
    n = 8
    schedules = [scenario(i) for i in range(n)]
    landed = np.ones(n, dtype=bool)
    landed[::4] = False
    history, landed_history, _ = batch_flight(schedules, starts(n), landed, 2.0)
    for i in (1, 4):
        alone, alone_landed, _ = batch_flight(
            [schedules[i]], starts(n)[i : i + 1], landed[i : i + 1], 2.0
        )
        np.testing.assert_array_equal(alone[:, 0], history[:, i])
        np.testing.assert_array_equal(alone_landed[:, 0], landed_history[:, i])


def test_batch_reset_forgets_only_the_chosen_drones_and_keeps_landed():
    vehicles = BatchMultirotor(IRIS, 4)
    pilot = BatchRatePilot(vehicles, IRIS_RATE_GAINS, DT, landed=[True, False] * 2)
    pilot.rate_loop.integral[:] = 1.0
    pilot.reset([1, 2])
    assert np.all(pilot.rate_loop.integral[[0, 3]] == 1.0)
    assert np.all(pilot.rate_loop.integral[[1, 2]] == 0.0)
    np.testing.assert_array_equal(pilot.landed, [True, False, True, False])


def test_batch_pilot_flies_in_the_batch_loop():
    """The pilot is a batch controller: a drone asked for more than its weight
    takes off, one asked for 90 % of it stays on its gear, and its unloading gear
    (a hop of a fraction of a millimetre) is not a take-off and then a crash."""
    course = load_course(COURSE_FILE)
    vehicles = BatchMultirotor(IRIS, 2, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    referee = batch_referee(
        course, IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points, 2
    )
    start = course.start_state(REST)
    flight = BatchFlight(vehicles, referee, DT, np.tile(start, (2, 1)))
    pilot = BatchRatePilot(vehicles, IRIS_RATE_GAINS, DT, landed=True)
    pilot.thrust_command = np.array([1.2, 0.9]) * WEIGHT
    for _ in range(50):
        flight.step(pilot)
    np.testing.assert_array_equal(pilot.landed, [False, True])
    assert -flight.states[0, 2] > 0.3
    assert -flight.states[1, 2] == pytest.approx(REST, abs=2e-3)
    assert not np.any(referee.ended)


def test_landed_drones_hold_their_integrals_at_zero_after_every_call():
    vehicles = BatchMultirotor(IRIS, 3, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    pilot = BatchRatePilot(vehicles, IRIS_RATE_GAINS, DT, landed=[True, True, False])
    states = np.tile(standing(), (3, 1))
    states[2, 2] = -3.0
    pilot.thrust_command = np.array([0.5, 0.9, 1.0]) * WEIGHT
    pilot.rate_command = np.tile(TURN, (3, 1))
    pilot(0.0, states)
    assert np.all(pilot.rate_loop.integral[0:2] == 0.0)
    assert np.all(pilot.rate_loop.integral[2] != 0.0)


def test_a_pilot_declared_landed_starts_from_a_clean_loop():
    """Charged integrals, then declared landed and asked to take off: the same
    rotor commands as a fresh pilot (reference and batch)."""
    state = standing()
    thrust = 1.2 * WEIGHT
    fresh = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    fresh.thrust_command, fresh.rate_command = thrust, TURN
    fresh(0.0, state)

    charged = RatePilot(vehicle(), IRIS_RATE_GAINS, DT)
    charged.rate_loop.integral = np.array([5.0, -5.0, 1.0])
    charged.landed = True
    charged.thrust_command, charged.rate_command = thrust, TURN
    charged(0.0, state)
    np.testing.assert_array_equal(
        charged.vehicle.rotor_speed_command, fresh.vehicle.rotor_speed_command
    )

    vehicles = BatchMultirotor(IRIS, 1)
    batch = BatchRatePilot(vehicles, IRIS_RATE_GAINS, DT)
    batch.rate_loop.integral[:] = [5.0, -5.0, 1.0]
    batch.landed[:] = True
    batch.thrust_command[:] = thrust
    batch.rate_command[:] = TURN
    batch(0.0, state[None, :])
    assert np.allclose(
        vehicles.rotor_speed_command[0], fresh.vehicle.rotor_speed_command, atol=1e-9
    )


def test_exactly_the_weight_does_not_take_off():
    pilot = RatePilot(vehicle(), IRIS_RATE_GAINS, DT, landed=True)
    pilot.thrust_command = WEIGHT
    pilot(0.0, standing())
    batch = BatchRatePilot(BatchMultirotor(IRIS, 1), IRIS_RATE_GAINS, DT, landed=True)
    batch.thrust_command[:] = WEIGHT
    batch(0.0, standing()[None, :])
    assert pilot.landed and batch.landed[0]
    pilot.thrust_command = np.nextafter(WEIGHT, np.inf)
    pilot(0.0, standing())
    batch.thrust_command[:] = np.nextafter(WEIGHT, np.inf)
    batch(0.0, standing()[None, :])
    assert not pilot.landed and not batch.landed[0]
