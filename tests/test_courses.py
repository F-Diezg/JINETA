"""Tests for the surroundings in the observation, the random courses and the
environment that flies many courses at once."""

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.campo.gates import CircularGate
from jineta.campo.indoor import Box, Cylinder, Room
from jineta.core import quaternion
from jineta.escuadron import quaternion as batch_quaternion
from jineta.escuadron.referee import FINISHED, FLYING
from jineta.picadero.baseline import BaselineSettings, fly_baseline
from jineta.picadero.courses import (
    CourseRanges,
    CourseSetEnv,
    _path_samples,
    random_course,
)
from jineta.picadero.env import CourseEnv
from jineta.picadero.spaces import Surroundings, closest_points

COURSE = load_course(COURSE_FILE)


# ---- the surroundings ----------------------------------------------------------------


def test_closest_points_by_hand():
    box = Box(center=[5.0, 5.0, -1.0], size=[2.0, 2.0, 2.0])
    points = np.array(
        [
            [5.0, 9.0, -1.0],  # in front of the East face: its foot
            [9.0, 10.0, -1.0],  # off a vertical edge: the edge (3-4-5)
            [9.0, 10.0, -4.0],  # off a corner
        ]
    )
    closest, distance = closest_points(box, points)
    np.testing.assert_allclose(
        closest,
        [[5.0, 6.0, -1.0], [6.0, 6.0, -1.0], [6.0, 6.0, -2.0]],
        atol=1e-8,
    )
    np.testing.assert_allclose(distance, [3.0, 5.0, np.sqrt(9 + 16 + 4)], atol=1e-12)
    column = Cylinder(base=[0.0, 0.0, 0.0], radius=1.0, height=3.0)
    closest, distance = closest_points(column, np.array([[3.0, 4.0, -1.0]]))
    np.testing.assert_allclose(closest, [[0.6, 0.8, -1.0]], atol=1e-8)
    assert distance[0] == pytest.approx(4.0)
    ring = CircularGate(center=[0.0, 0.0, -2.0], yaw=0.0, radius=1.0, thickness=0.2)
    # In the gate's plane, 2 m to the right of the centre: the opening's radius
    # is 1 m, the bar 0.2 m thick around it (a tube of radius 0.1 centred 1.1 m
    # out), so the closest point is 1.2 m out.
    closest, distance = closest_points(ring, np.array([[0.0, 2.0, -2.0]]))
    np.testing.assert_allclose(closest, [[0.0, 1.2, -2.0]], atol=1e-8)
    assert distance[0] == pytest.approx(0.8)


def test_the_room_faces_and_the_nearest_first():
    room = Room([10.0, 6.0, 3.0])
    surroundings = Surroundings(room, nearest=6)
    position = np.array([[2.0, 1.0, -0.5]])  # 0.5 m up, 1 m from the West wall
    points, distances = surroundings.points(position)
    np.testing.assert_allclose(distances[0], [0.5, 2.5, 2.0, 8.0, 1.0, 5.0])
    np.testing.assert_allclose(points[0, 0], [2.0, 1.0, 0.0])  # floor
    np.testing.assert_allclose(points[0, 4], [2.0, 0.0, -0.5])  # wall y = 0
    level = np.repeat(np.eye(3)[None], 1, axis=0)  # nose North: body = NED
    seen = surroundings(position, level).reshape(6, 3)
    np.testing.assert_allclose(
        seen,
        [
            [0.0, 0.0, 0.5],  # the floor, 0.5 m below
            [0.0, -1.0, 0.0],  # the wall y = 0, 1 m to the left
            [-2.0, 0.0, 0.0],  # the wall x = 0, 2 m behind
            [0.0, 0.0, -2.5],  # the ceiling
            [0.0, 5.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
    )
    # Nose East: the wall y = 0 is now behind, the wall x = 0 to the right.
    east = quaternion.to_matrix(quaternion.from_euler(0.0, 0.0, np.pi / 2))
    seen = Surroundings(room, nearest=3)(position, east.T[None]).reshape(3, 3)
    np.testing.assert_allclose(
        seen, [[0.0, 0.0, 0.5], [-1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], atol=1e-12
    )


def test_obstacles_and_frames_among_the_surfaces():
    pillar = Cylinder(base=[5.0, 3.0, 0.0], radius=0.5, height=3.0)
    room = Room([10.0, 6.0, 4.0])
    surroundings = Surroundings(room, [pillar], nearest=2)
    position = np.array([[3.0, 3.0, -1.6]])  # 1.5 m from the pillar's side
    seen = surroundings(position, np.eye(3)[None]).reshape(2, 3)
    np.testing.assert_allclose(seen[0], [1.5, 0.0, 0.0], atol=1e-8)  # the pillar
    np.testing.assert_allclose(seen[1], [0.0, 0.0, 1.6])  # then the floor


def test_the_observation_with_surroundings():
    plain = CourseEnv(COURSE, n=3)
    seeing = CourseEnv(COURSE, n=3, nearest=4)
    assert seeing.observation_space.shape == (plain.observer.size + 12,)
    a, _ = plain.reset()
    b, _ = seeing.reset()
    np.testing.assert_array_equal(b[:, : a.shape[1]], a)  # the old part unchanged
    # Standing at the start (1.5, 5) of the 18 x 10 x 4 m hall, 5.4 cm up: the
    # floor first, then the wall x = 0 (1.5 m behind), then the next.
    np.testing.assert_allclose(b[0, -12:-9], [0.0, 0.0, 0.054], atol=1e-3)
    np.testing.assert_allclose(b[0, -9:-6], [-1.5, 0.0, 0.0], atol=1e-6)
    # Many drones give what one gives, bit for bit.
    rng = np.random.default_rng(0)
    states = np.tile(seeing.start, (6, 1))
    states[:, 0:3] += rng.uniform(-1.0, 1.0, (6, 3)) * [3.0, 3.0, 0.5]
    states[:, 6:10] = batch_quaternion.normalize(rng.standard_normal((6, 4)))
    observer = seeing.observer
    together = observer(states, np.zeros(6, dtype=int), np.zeros((6, 4)))
    for i in range(6):
        alone = observer(states[i : i + 1], np.zeros(1, dtype=int), np.zeros((1, 4)))
        np.testing.assert_array_equal(alone[0], together[i])


def test_rejects_bad_surroundings():
    with pytest.raises(ValueError):
        Surroundings(Room([5.0, 5.0, 3.0]), nearest=7)
    with pytest.raises(ValueError):
        Surroundings(Room([5.0, 5.0, 3.0]), nearest=0)


# ---- random courses ------------------------------------------------------------------


def test_random_courses_keep_their_promises():
    """200 courses: within their ranges, and the path through the gates keeps
    its distance from every obstacle and from the walls."""
    rng = np.random.default_rng(0)
    r = CourseRanges()
    for _ in range(200):
        course = random_course(rng)
        length, width, _ = course.room.size
        assert r.room[0][0] <= length <= r.room[0][1]
        assert r.gates[0] <= len(course.gates) <= r.gates[1]
        assert len(course.obstacles) <= r.obstacles[1]
        for gate in course.gates:
            assert r.wall_margin <= gate.center[0] <= length - r.wall_margin
            assert r.wall_margin <= gate.center[1] <= width - r.wall_margin
        centers = np.array([g.center for g in course.gates])
        gaps = np.linalg.norm(centers[:, None, :2] - centers[None, :, :2], axis=2)
        assert np.all(gaps[np.triu_indices(len(centers), 1)] >= r.gate_spacing)
        samples = _path_samples(course, r.approach)
        for obstacle in course.obstacles:
            assert np.min(obstacle.signed_distance(samples)) >= r.path_clearance
        assert np.min(course.room.clearance(samples[5:])) >= 0.6


def test_random_courses_repeat_with_their_seed_and_vary():
    a = random_course(np.random.default_rng(5))
    b = random_course(np.random.default_rng(5))
    c = random_course(np.random.default_rng(6))
    np.testing.assert_array_equal(a.room.size, b.room.size)
    for g, h in zip(a.gates, b.gates, strict=True):
        np.testing.assert_array_equal(g.center, h.center)
    assert not np.array_equal(a.room.size, c.room.size)


def test_the_baseline_flies_random_courses():
    """A cautious baseline (2.5 m/s) finishes the first six courses of seed 0:
    they can be flown along their path."""
    rng = np.random.default_rng(0)
    settings = BaselineSettings(2.5, 4.0, 3.0, 1.5, approach=1.5)
    for _ in range(6):
        status, lap, _, _ = fly_baseline(random_course(rng), settings, 1)
        assert status[0] == FINISHED and np.isfinite(lap[0])


# ---- many courses at once ------------------------------------------------------


def fixed_courses(seed: int):
    """make_course for CourseSetEnv: random courses from its own generator."""
    rng = np.random.default_rng(seed)
    return lambda _: random_course(rng)


def test_many_courses_fly_as_their_own_environments():
    """Three courses of four drones: until a course is renewed, its drones fly
    exactly as a CourseEnv of that course; at the renewal the drones still
    flying are truncated with their last observation, and fly the new course."""
    env = CourseSetEnv(fixed_courses(1), k=3, n=4, renew_every=30, nearest=4)
    observations, _ = env.reset(seed=0)
    assert observations.shape == (12, 43 + 12) and env.n == 12
    alone = [CourseEnv(e.course, n=4, nearest=4) for e in env.envs]
    for a in alone:
        a.reset()
    rng = np.random.default_rng(2)
    # Course i is renewed at the steps where (step + 10 i) % 30 == 0: course 2 at
    # step 10, course 1 at 20, course 0 at 30.
    for step in range(1, 11):
        actions = rng.uniform(-0.3, 0.6, (12, 4))
        observations, _, terminated, truncated, info = env.step(actions)
        for i, a in enumerate(alone):
            rows = slice(4 * i, 4 * i + 4)
            mine, _, ended, cut, _ = a.step(actions[rows])
            if i == 2 and step == 10:  # renewed
                going_on = ~(ended | cut)
                assert np.all(truncated[rows][going_on])
                np.testing.assert_array_equal(
                    info["final_observation"][rows][going_on], mine[going_on]
                )
                assert np.all(info["status"][rows][going_on] == FLYING)
                assert env.envs[2].course is not a.course
                np.testing.assert_array_equal(
                    observations[rows], env.envs[2].reset()[0]
                )
            else:
                np.testing.assert_array_equal(observations[rows], mine)
                np.testing.assert_array_equal(truncated[rows], cut)
                np.testing.assert_array_equal(terminated[rows], ended)
    assert env.renewals == 1


def test_many_courses_repeat_with_their_seed():
    def run():
        env = CourseSetEnv(fixed_courses(3), k=2, n=2, renew_every=5)
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        out = [env.step(rng.uniform(-0.2, 0.5, (4, 4)))[0] for _ in range(12)]
        return np.array(out), env.renewals

    (a, ra), (b, rb) = run(), run()
    np.testing.assert_array_equal(a, b)
    assert ra == rb == 4  # course 0 at steps 5 and 10, course 1 at 3 and 8


def test_rejects_bad_course_sets():
    with pytest.raises(ValueError):
        CourseSetEnv(fixed_courses(0), k=0, n=2)
    with pytest.raises(ValueError):
        CourseSetEnv(fixed_courses(0), k=2, n=2, renew_every=0)
