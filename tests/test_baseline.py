"""Tests for the classical baseline: the path through the gates, the speed plan,
the guidance, the lap it flies (in the batch, in the reference engine and through
the training environment's actions) and the sweep that picks its settings."""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.escuadron.control import BatchAttitudeController, BatchVelocityController
from jineta.escuadron.referee import FINISHED
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_HITBOX, IRIS_LANDING_GEAR
from jineta.picadero.baseline import (
    BaselineSettings,
    GatePath,
    PathGuidance,
    baseline_flight,
    fly_baseline,
    sweep,
)
from jineta.picadero.env import CourseEnv

COURSE = load_course(COURSE_FILE)
DT = 0.01
# The settings the sweep of examples/baseline_course.py picks for the first course.
BEST = BaselineSettings(5.0, 8.0, 5.0, 3.0, approach=2.0)
BEST_LAP = 6.773  # [s]


# ---- the path ----------------------------------------------------------------------


def test_the_path_through_the_first_course_by_hand():
    path = GatePath.through(COURSE, approach=2.0, takeoff_height=0.5)
    assert len(path.points) == 1 + 3 * 4
    np.testing.assert_allclose(path.points[0], [1.5, 5.0, -0.5])
    # Gate 0 (5, 5, -1.5), flown North: entry 2 m South, exit 2 m North.
    np.testing.assert_allclose(
        path.points[1:4], [[3, 5, -1.5], [5, 5, -1.5], [7, 5, -1.5]]
    )
    # Gate 2 (14, 7, -1.5), yaw 30°: heading (cos 30°, sin 30°, 0).
    heading = np.array([np.sqrt(3) / 2, 0.5, 0.0])
    np.testing.assert_allclose(path.points[7], COURSE.gates[2].center - 2 * heading)
    np.testing.assert_allclose(path.points[9], COURSE.gates[2].center + 2 * heading)
    # Straight through every gate: no turn at the centres (points 2, 5, 8, 11).
    np.testing.assert_allclose(path.turns[[1, 4, 7, 10]], 0.0, atol=1e-7)
    assert path.length == pytest.approx(np.sum(path.lengths))
    np.testing.assert_allclose(np.linalg.norm(path.directions, axis=1), 1.0)


def test_points_along_the_path_and_past_its_ends():
    path = GatePath(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    assert path.length == 7.0
    np.testing.assert_allclose(path.turns, [np.pi / 2])
    np.testing.assert_allclose(
        path.point_at(np.array([0.0, 1.5, 3.0, 5.0, 7.0, 9.0, -1.0])),
        [
            [0, 0, 0],
            [1.5, 0, 0],
            [3, 0, 0],
            [3, 2, 0],
            [3, 4, 0],
            [3, 6, 0],
            [-1, 0, 0],
        ],
    )


def test_the_closest_point_is_searched_only_a_few_segments_ahead():
    # A zigzag that comes back past the start: segment 5 passes next to x = 0.5.
    corners = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [2, 1, 0],
        [2, 3, 0],
        [0.5, 3, 0],
        [0.5, 0.1, 0],
    ]
    path = GatePath(np.array(corners, dtype=float))
    point = np.array([[0.5, 0.05, 0.0]])
    s, segment = path.closest(point, np.array([0]))
    assert segment[0] == 0 and s[0] == pytest.approx(0.5)  # not segment 5
    s, segment = path.closest(point, np.array([3]))  # from segment 3 on, it is 5
    assert segment[0] == 5


def test_rejects_bad_paths_and_settings():
    with pytest.raises(ValueError):
        GatePath(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="coincide"):
        GatePath(np.array([[0.0, 0, 0], [0.0, 0, 0], [1.0, 0, 0]]))
    with pytest.raises(ValueError, match="cruise"):
        BaselineSettings(cruise=0.0)
    with pytest.raises(ValueError, match="lookahead"):
        BaselineSettings(lookahead=[1.0, -1.0])
    with pytest.raises(ValueError, match="approach"):
        BaselineSettings(approach=0.0)


# ---- the speed plan ----------------------------------------------------------------


def test_the_speed_plan_by_hand():
    """One right-angle corner 10 m along: radius L / tan(45°) = L, so the corner
    speed is sqrt(a L); before it, braking at b from 2 L before the corner."""
    path = GatePath(np.array([[0.0, 0.0, -1.0], [10.0, 0.0, -1.0], [10.0, 20.0, -1.0]]))
    settings = BaselineSettings(
        cruise=6.0, lateral_accel=2.0, braking_accel=1.0, lookahead=2.0
    )
    guidance = PathGuidance(COURSE, settings, 1, path=path)
    corner = np.sqrt(2.0 * 2.0)  # 2 m/s
    s = np.array([0.0, 4.0, 7.0, 8.0, 10.0, 12.0, 12.5, 20.0])
    expected = [
        min(6.0, np.sqrt(corner**2 + 2.0 * 8.0)),  # 8 m before the arc starts
        np.sqrt(corner**2 + 2.0 * 4.0),
        np.sqrt(corner**2 + 2.0 * 1.0),
        corner,  # the arc: from 2 m before the corner ...
        corner,
        corner,  # ... to 2 m after it
        6.0,  # past the corner: cruise again
        6.0,
    ]
    for value, want in zip(s, expected, strict=True):
        got = guidance.speed_limit(np.array([value]))[0]
        assert got == pytest.approx(want, abs=1e-12)


def test_straight_through_a_gate_the_corner_does_not_slow_down():
    settings = BaselineSettings(
        cruise=5.0, lateral_accel=4.0, lookahead=1.0, approach=2.0
    )
    guidance = PathGuidance(COURSE, settings, 1)
    # The corners at gate centres have no turn: their allowed speed is the cruise.
    np.testing.assert_allclose(guidance._corner_speed[0, [1, 4, 7, 10]], 5.0)


# ---- the guidance -------------------------------------------------------------------


def test_it_climbs_straight_up_first_and_never_goes_back_along_the_path():
    flight, pilot = baseline_flight(COURSE, BEST, 1)
    progress, commands = [], []
    for _ in range(300):
        flight.step(pilot)
        progress.append(pilot.guidance.progress[0])
        commands.append(pilot.autopilot.velocity_command[0].copy())
    commands = np.array(commands)
    first = int(np.argmax(np.any(commands != [0.0, 0.0, -1.0], axis=1)))
    assert first > 10  # straight up until 0.5 m above the floor
    np.testing.assert_array_equal(commands[:first], [[0.0, 0.0, -1.0]] * first)
    assert np.all(np.diff(progress) >= 0.0)
    assert progress[-1] > 8.0  # well on its way after 3 s


def test_the_best_settings_finish_the_first_course():
    statuses, laps, clearance, _ = fly_baseline(COURSE, BEST)
    assert statuses[0] == FINISHED
    assert laps[0] == pytest.approx(BEST_LAP, abs=0.02)
    assert clearance[0] > 0.05  # never within 5 cm of a wall, obstacle or frame
    assert laps[0] < 8.81 - 2.0  # 2 s faster than straight at each centre (phase 8)


def test_a_lap_in_the_batch_is_the_reference_engine_lap():
    """The same guidance driving the reference autopilot: every state within 1e-9,
    the same lap time."""
    flight, pilot = baseline_flight(COURSE, BEST, 1)
    history = [flight.states[0].copy()]
    while not flight.referee.ended[0]:
        flight.step(pilot)
        history.append(flight.states[0].copy())

    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    autopilot = Autopilot(vehicle, IRIS_GAINS, DT, landed=True)
    referee = COURSE.referee(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    guidance = PathGuidance(COURSE, BEST, 1)

    def pre_step(t, y):
        velocity, yaw = guidance.commands(y[None, :])
        autopilot.velocity_command = velocity[0]
        autopilot.yaw_command = float(yaw[0])
        autopilot(t, y)

    start = COURSE.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    _, states = run(
        vehicle.derivative,
        start,
        DT,
        COURSE.time_limit + 1.0,
        stop=referee,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    assert referee.status == "finished"
    assert len(states) == len(history)
    assert np.max(np.abs(states - np.array(history))) < 1e-9
    assert referee.lap_time == pytest.approx(flight.referee.lap_time[0], abs=1e-9)


def test_drones_with_different_settings_fly_as_they_would_alone():
    cruise = np.array([3.0, 5.0, 8.0])
    lateral = np.array([4.0, 8.0, 12.0])
    settings = BaselineSettings(cruise, lateral, 5.0, 3.0, approach=2.0)
    statuses, laps, gaps, _ = fly_baseline(COURSE, settings, 3)
    for i in range(3):
        alone = BaselineSettings(cruise[i], lateral[i], 5.0, 3.0, approach=2.0)
        s, lap, gap, _ = fly_baseline(COURSE, alone)
        assert s[0] == statuses[i]
        np.testing.assert_array_equal(lap, laps[i : i + 1])
        np.testing.assert_array_equal(gap, gaps[i : i + 1])


# ---- through the training environment ----------------------------------------------


def fly_through_the_environment(action_repeat: int):
    """The baseline as a policy: its guidance, velocity loop and attitude loop at
    the environment's rate turn the state into thrust and body rates, given to the
    environment as actions. Returns the info of the step that ended the lap, and
    the samples (positions, target gates) of every step, the last one where the
    lap ended."""
    env = CourseEnv(COURSE, action_repeat=action_repeat)
    env.reset()
    ended = []
    restart = env._restart

    def remember_where_it_ended(drones):
        ended.append(env.flight.states[0, 0:3].copy())
        restart(drones)

    env._restart = remember_where_it_ended
    guidance = PathGuidance(COURSE, BEST, 1)
    velocity_loop = BatchVelocityController(IRIS, IRIS_GAINS.velocity, env.step_time, 1)
    attitude_loop = BatchAttitudeController(IRIS_GAINS.attitude)
    positions = [env.flight.states[0, 0:3].copy()]
    targets = []
    for _ in range(2000):
        states = env.flight.states
        velocity, yaw = guidance.commands(states)
        attitude, thrust = velocity_loop.update(velocity, yaw, states)
        rates = attitude_loop.update(attitude, states)
        targets.append(env.flight.referee.next_gate[0])
        _, _, terminated, truncated, info = env.step(env.limits.actions(thrust, rates))
        positions.append(ended[0] if ended else env.flight.states[0, 0:3].copy())
        if terminated[0] or truncated[0]:
            break
    return info, np.array(positions), np.array(targets)


def test_the_baseline_flown_through_the_actions_finishes():
    """At 50 Hz, inside the action envelope: the envelope allows a winning lap."""
    info, _, _ = fly_through_the_environment(action_repeat=2)
    assert info["status"][0] == FINISHED
    assert info["lap_time"][0] == pytest.approx(BEST_LAP, abs=0.2)


def test_the_reward_of_a_whole_lap_through_the_environment():
    """At 100 Hz (every physics sample seen): the return is the progress gate by
    gate, measured against the gate targeted at the start of each step, plus one
    bonus per gate. (This also checks that the environment reads the target before
    the step: read after it, the steps that cross a gate would measure against the
    next one.)"""
    info, positions, targets = fly_through_the_environment(action_repeat=1)
    assert info["status"][0] == FINISHED
    expected = 0.0
    for gate in range(4):
        steps = np.flatnonzero(targets == gate)
        center = COURSE.gates[gate].center
        first, last = positions[steps[0]], positions[steps[-1] + 1]
        expected += np.linalg.norm(first - center) - np.linalg.norm(last - center)
    assert info["episode_return"][0] == pytest.approx(expected + 4.0, abs=1e-9)


# ---- the sweep ---------------------------------------------------------------------


def test_the_sweep_keeps_the_fastest_robust_lap():
    result = sweep(
        COURSE,
        approach=(2.0,),
        cruise=(5.0,),
        lateral_accel=(8.0, 20.0),
        braking_accel=(5.0,),
        lookahead=(3.0,),
    )
    # 20 m/s² in the corners is too much for the Iris: not robust.
    np.testing.assert_array_equal(result.robust, [True, False])
    assert result.lap_time == pytest.approx(BEST_LAP, abs=0.02)
    assert float(result.best.lateral_accel) == 8.0 and result.best.approach == 2.0
    np.testing.assert_allclose(result.tried[0], [2.0, 5.0, 8.0, 5.0, 3.0])


def test_the_sweep_fails_loudly_when_nothing_finishes():
    with pytest.raises(RuntimeError, match="robustly"):
        sweep(
            COURSE,
            approach=(2.0,),
            cruise=(8.0,),
            lateral_accel=(30.0,),
            braking_accel=(10.0,),
            lookahead=(0.5,),
        )


def test_progress_holds_when_the_drone_falls_back():
    """A drone pushed back along the path (a gust, an overshoot that turns round)
    keeps its progress and the segment it reached: it does not chase a point
    behind it."""
    guidance = PathGuidance(COURSE, BEST, 1)
    path = guidance.path
    states = np.zeros((1, 13))
    states[0, 6] = 1.0
    for s in (6.0, 3.0):
        states[0, 0:3] = path.point_at(np.array([s]))[0]
        guidance.commands(states)
    assert guidance.progress[0] == pytest.approx(6.0)
    assert guidance.segment[0] == np.searchsorted(path.distance, 6.0) - 1
