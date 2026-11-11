"""Tests for the batch loop: drones of one batch fly the first course at different
speeds as the reference engine flies it (examples/iris_course.py), laps end on
their own, a drone's next lap after a reset is its first lap bit for bit, and the
other drones are not disturbed."""

import numpy as np
import pytest

import examples.iris_course as iris_course
from examples.batch_course import course_flight, fly_laps
from jineta.campo.course import load_course
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import CRASHED, FINISHED, FLYING, STATUS_NAMES
from jineta.escuadron.simulation import BatchFlight
from jineta.jinete.profiles import IRIS, IRIS_LANDING_GEAR

COURSE = load_course(iris_course.COURSE_FILE)
SPEEDS = np.array([2.0, 2.5, 3.0, 4.0])


def test_laps_at_different_speeds_match_the_reference_engine():
    """Four Iris in one batch, from 2 to 4 m/s: each flies its lap as the
    reference engine does at its speed (states within 1e-9 at every step, the same
    end, moment and passes); the 4 m/s one clips the frame of gate 2, as measured in
    phase 8."""
    flight, guidance = course_flight(COURSE, SPEEDS)
    history = [flight.states.copy()]
    while not np.all(flight.referee.ended):
        flight.step(guidance)
        history.append(flight.states.copy())
    history = np.array(history)
    for i, speed in enumerate(SPEEDS):
        _, states, referee = iris_course.fly_course(COURSE, speed)
        steps = len(states)
        assert np.allclose(history[:steps, i], states, rtol=0.0, atol=1e-9)
        assert STATUS_NAMES[flight.referee.status[i]] == referee.status
        assert flight.referee.end_time[i] == pytest.approx(referee.end_time, abs=1e-9)
        got = flight.referee.passes(i)
        assert [p.gate for p in got] == [p.gate for p in referee.passes]
        for a, b in zip(got, referee.passes, strict=True):
            assert a.time == pytest.approx(b.time, abs=1e-9)
        # after its end the drone stays where it ended
        assert np.all(history[steps:, i] == history[steps - 1, i])
    assert np.array_equal(
        flight.referee.status, [FINISHED, FINISHED, FINISHED, CRASHED]
    )
    assert flight.referee.lap_time[2] == pytest.approx(8.8145, abs=1e-4)
    collision = flight.referee.monitor.collision(3)
    assert (collision.kind, collision.index) == ("gate", 1)


def test_a_reset_drone_repeats_its_lap_and_disturbs_nobody():
    """Drone 0 (4 m/s) crashes into gate 2 and starts a second lap at once, while
    drone 1 (3 m/s) is still on its first: drone 0's second lap is its first bit
    for bit (states, end), and drone 1 flies exactly as in a batch where nobody is
    reset."""
    speeds = np.array([4.0, 3.0])
    flight, guidance = course_flight(COURSE, speeds)
    quiet, quiet_guidance = course_flight(COURSE, speeds)
    start = flight.states[0].copy()
    first, second = [flight.states[0].copy()], []
    laps = []
    for _ in range(2000):  # two laps of 3.24 s take 648 steps
        ended = flight.step(guidance)
        quiet.step(quiet_guidance)
        (first if not laps else second).append(flight.states[0].copy())
        assert np.array_equal(flight.states[1], quiet.states[1])
        if ended[0]:
            laps.append((flight.referee.status[0], flight.referee.end_time[0]))
            if len(laps) == 2:
                break
            assert flight.referee.status[1] == FLYING  # drone 1 flies on
            flight.reset([0], start)
            guidance.reset(np.array([0]))
            second.append(flight.states[0].copy())
            assert flight.clock[0] == 0.0
    assert laps[0] == laps[1]
    assert laps[0][0] == CRASHED
    assert np.array_equal(np.array(second), np.array(first))


def test_step_reports_only_the_laps_that_end_in_it():
    flight, guidance = course_flight(COURSE, np.array([4.0, 3.0]))
    reports = []
    while not np.all(flight.referee.ended):
        reports.append(flight.step(guidance))
        assert len(reports) < 1500
    reports = np.array(reports)
    assert np.array_equal(reports.sum(axis=0), [1, 1])  # each reported once
    crash_step = np.argmax(reports[:, 0])
    assert flight.clock[0] == pytest.approx((crash_step + 1) * 0.01)
    # the crashed drone's clock stopped there while the other flew on
    assert flight.clock[1] > flight.clock[0]


def test_fly_laps_restarts_each_drone_on_its_own():
    results, history, steps = fly_laps(COURSE, np.array([4.0, 3.5]), laps=3)
    for laps in results:
        assert [status for status, _ in laps] == ["crashed"] * 3
        assert len({end for _, end in laps}) == 1  # bit for bit the same lap
    assert results[0][0][1] < results[1][0][1]  # each on its own clock
    assert steps * 0.01 == pytest.approx(3 * results[1][0][1], abs=0.02)
    assert len(history) > 1


def test_rejects_inconsistent_input():
    flight, _ = course_flight(COURSE, np.array([3.0, 3.0]))
    referee = flight.referee
    vehicles = BatchMultirotor(IRIS, 2, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    with pytest.raises(ValueError, match="dt"):
        BatchFlight(vehicles, referee, 0.0, flight.states)
    with pytest.raises(ValueError, match="same n"):
        BatchFlight(BatchMultirotor(IRIS, 3), referee, 0.01, np.zeros((3, 13)))
    with pytest.raises(ValueError, match="start"):
        BatchFlight(vehicles, referee, 0.01, np.zeros((2, 12)))
