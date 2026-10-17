"""Tests for the engagement loop: the Iris under pure pursuit against scripted targets.

The stop condition is tested alone first (cheap, no dynamics). The closed-loop tests
fly the full cascade (6-DOF Iris + autopilot) and compare it with the closed-form
pure pursuit of an ideal point (see test_guidance.py): the real vehicle can only be
LATER than the ideal one, because its velocity lags the command.
"""

import numpy as np
import pytest

from jineta.engagement import EndOfPursuit, engage
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS
from jineta.jinete.targets import (
    RecordedTrack,
    StraightLineTarget,
    TurningTarget,
    track_positions,
)

DT = 0.01  # control and simulation step [s]
STILL_TARGET = StraightLineTarget(start=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])


def at_distance(x):
    """A 13-element state whose position is (x, 0, 0): the target is at the origin,
    so the separation is |x|. Only the position matters to the stop condition."""
    state = np.zeros(13)
    state[0] = x
    return state


def hover_at(position):
    """Iris state hovering (level, at rest) at a NED position."""
    return np.array([*position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)


def fly(target, start, speed, t_end, **kwargs):
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    return engage(autopilot, target, hover_at(start), speed, t_end, **kwargs)


def separation(result):
    return np.linalg.norm(result.target_positions - result.states[:, :3], axis=1)


# --- Stop condition alone -------------------------------------------------------------


def test_stops_inside_the_capture_radius():
    stop = EndOfPursuit(
        STILL_TARGET, capture_radius=0.5, margin=10.0, initial_distance=50
    )
    assert not stop(0.1, at_distance(5.0))
    assert not stop(0.2, at_distance(0.6))
    assert stop(0.3, at_distance(0.4))


def test_stops_only_after_growing_by_the_margin_over_the_smallest():
    stop = EndOfPursuit(
        STILL_TARGET, capture_radius=0.5, margin=10.0, initial_distance=50
    )
    for x in (40.0, 30.0, 20.0, 12.0):  # closing in: smallest is 12
        assert not stop(0.0, at_distance(x))
    assert not stop(0.0, at_distance(21.9))  # opened up by 9.9 m: not yet
    assert stop(0.0, at_distance(22.1))  # 10.1 m above the smallest: it passed


def test_growing_separation_is_not_a_pass_until_it_has_closed_in():
    """Taking off from rest behind a fleeing target the separation grows at first;
    that is not a pass, however far it grows."""
    stop = EndOfPursuit(
        STILL_TARGET, capture_radius=0.5, margin=10.0, initial_distance=50
    )
    for x in (52.0, 60.0, 75.0, 100.0):
        assert not stop(0.0, at_distance(x))
    assert not stop(0.0, at_distance(49.0))  # now it closes in...
    assert stop(0.0, at_distance(60.0))  # ...and then opens: a pass


@pytest.mark.parametrize("radius, margin", [(0.0, 10.0), (-1.0, 10.0), (0.5, 0.0)])
def test_stop_rejects_invalid_parameters(radius, margin):
    with pytest.raises(ValueError):
        EndOfPursuit(STILL_TARGET, radius, margin, initial_distance=50.0)


# --- Closed loop with the Iris --------------------------------------------------------


def test_catches_a_slower_target_in_three_dimensions():
    """Target 40 m East and 20 m above, flying East at 4 m/s; interceptor at 10 m/s.
    The ideal point pursuer of the closed form (T = r0 (k + cos phi0) / (Vt (k^2 - 1)))
    catches it at T. The Iris starts from rest and follows the velocity command with
    a lag, so it can only take longer: between T and T + 3 s (the +3 s is a margin
    over the ~1.8 s seen when this test was written, not a derived bound)."""
    target = StraightLineTarget(start=[0.0, 40.0, -120.0], velocity=[0.0, 4.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=10.0, t_end=40.0)

    r0 = np.array([0.0, 40.0, -20.0])
    k = 10.0 / 4.0
    ideal = np.linalg.norm(r0) * (k + r0[1] / np.linalg.norm(r0)) / (4.0 * (k**2 - 1))

    assert result.pursuit_feasible
    assert result.captured
    assert result.closest.distance < 0.5
    assert ideal - 0.2 < result.closest.time < ideal + 3.0
    assert result.times[-1] < 40.0  # it stopped at the capture, not at t_end


def test_catches_a_fleeing_target_it_starts_behind():
    """Regression: from rest, behind a target that runs away at 10 m/s, the
    separation grows for the first seconds. With a margin of 5 m it grows by more
    than that, and it must not end the run as a miss: it has not closed in yet."""
    target = StraightLineTarget(start=[40.0, 0.0, -100.0], velocity=[10.0, 0.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=16.0, t_end=40.0, margin=5.0)
    distance = separation(result)
    assert distance.max() > distance[0] + 5.0  # it opened up by more than the margin
    assert result.captured


def test_a_faster_target_is_reported_infeasible_and_never_approached():
    target = StraightLineTarget(start=[50.0, 0.0, -100.0], velocity=[25.0, 0.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=20.0, t_end=10.0)
    assert not result.pursuit_feasible
    assert not result.captured
    assert result.closest.time == 0.0  # the closest it ever was is the start
    assert np.isclose(result.closest.distance, 50.0)
    assert separation(result)[-1] > 100.0


def test_a_pass_without_capture_ends_the_run_after_the_closest_approach():
    """A target crossing at 15 m/s against 10 m/s: the interceptor gets closer, goes
    past and falls behind; the run ends once the separation has grown by the margin
    over its smallest value."""
    target = StraightLineTarget(start=[100.0, -60.0, -100.0], velocity=[0.0, 15.0, 0.0])
    result = fly(target, start=[0.0, 0.0, -100.0], speed=10.0, t_end=30.0, margin=10.0)
    distance = separation(result)

    assert not result.pursuit_feasible
    assert not result.captured
    assert result.times[-1] < 30.0  # stopped early
    assert 0.0 < result.closest.time < result.times[-1]
    assert result.closest.distance < distance[0] - 10.0  # it did close in
    assert distance[-1] > distance.min() + 10.0  # and then opened by the margin


def test_the_same_autopilot_can_be_reused():
    """The integrators of the loops are reset at the start of each engagement."""
    target = StraightLineTarget(start=[0.0, 25.0, -100.0], velocity=[0.0, 3.0, 0.0])
    autopilot = Autopilot(Multirotor(IRIS), IRIS_GAINS, DT)
    first = engage(autopilot, target, hover_at([0.0, 0.0, -100.0]), 10.0, 20.0)
    second = engage(autopilot, target, hover_at([0.0, 0.0, -100.0]), 10.0, 20.0)
    assert first.captured
    assert np.array_equal(first.times, second.times)
    assert np.array_equal(first.states, second.states)


def test_negative_speed_is_rejected():
    target = StraightLineTarget(start=[10.0, 0.0, -100.0], velocity=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        fly(target, start=[0.0, 0.0, -100.0], speed=-1.0, t_end=1.0)


def test_a_recorded_target_gives_the_same_engagement():
    # A simulated threat (e.g. the fixed wing) is flown first, recorded at the step
    # rate and replayed through `RecordedTrack`. Replaying a recorded turn must give
    # the engagement of the turn itself: the samples fall on the step grid.
    turn = TurningTarget(
        start=[30.0, 30.0, -110.0], speed=5.0, heading=1.0, turn_rate=0.1
    )
    times = DT * np.arange(round(40.0 / DT) + 1)
    velocities = np.array([turn(t)[1] for t in times])
    recorded = RecordedTrack(times, track_positions(turn, times), velocities)
    start = [0.0, 0.0, -100.0]
    original = fly(turn, start=start, speed=10.0, t_end=30.0)
    replayed = fly(recorded, start=start, speed=10.0, t_end=30.0)
    assert original.captured and replayed.captured
    assert np.allclose(replayed.states, original.states, atol=1e-9)
    assert np.isclose(replayed.closest.time, original.closest.time, atol=1e-9)
