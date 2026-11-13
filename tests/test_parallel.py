"""Tests for several environments at once (`picadero.parallel`): workers in
their own processes give the same numbers bit for bit as the same parts in one
process, and as one environment with all the drones; errors in a worker reach
the main process; a training gives the same result with any number of
workers."""

import csv
import dataclasses
from functools import partial

import numpy as np
import pytest

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.ppo import PPOSettings
from jineta.picadero.courses import CourseRanges, random_course
from jineta.picadero.env import CourseEnv
from jineta.picadero.parallel import (
    CourseEnvPart,
    CourseSetEnvPart,
    ParallelEnv,
    SerialEnv,
    WorkerError,
    make_parallel,
    worker_seed,
)
from jineta.picadero.training import train

COURSE = load_course(COURSE_FILE)
SHORT = dataclasses.replace(COURSE, time_limit=1.0)


def scripted_actions(steps: int, n: int, seed: int = 0) -> np.ndarray:
    """Random actions with the thrust mostly high: drones take off, tumble,
    crash or reach the time limit, so episodes end and restart."""
    rng = np.random.default_rng(seed)
    actions = rng.uniform(-1.0, 1.0, size=(steps, n, 4))
    actions[:, :, 0] = rng.uniform(-0.2, 1.0, size=(steps, n))
    return actions


def assert_same_step(a, b) -> None:
    """Two step results equal bit for bit (nan as nan in the infos)."""
    for x, y in zip(a[:4], b[:4], strict=True):
        np.testing.assert_array_equal(x, y)
    assert set(a[4]) == set(b[4])
    for key in a[4]:
        np.testing.assert_array_equal(a[4][key], b[4][key])


def test_worker_seeds():
    assert worker_seed(None, 3) is None
    assert worker_seed(5, 0) == worker_seed(5, 0)
    seeds = {worker_seed(5, i) for i in range(8)} | {worker_seed(6, 0)}
    assert len(seeds) == 9  # every part and every seed its own


def test_workers_fly_as_one_environment_with_all_the_drones():
    """Two workers of three drones and one environment of six: the same
    observations, rewards, ends and infos at every step, restarts included."""
    whole = CourseEnv(SHORT, n=6)
    actions = scripted_actions(80, 6)
    with ParallelEnv(CourseEnvPart(SHORT, 3), 2) as parallel:
        assert parallel.n == 6 and parallel.sizes == [3, 3]
        for mine, theirs in (
            (parallel.observation_space, whole.observation_space),
            (parallel.action_space, whole.action_space),
        ):
            np.testing.assert_array_equal(mine.low, theirs.low)
            np.testing.assert_array_equal(mine.high, theirs.high)
        first, _ = parallel.reset(seed=4)
        np.testing.assert_array_equal(first, whole.reset(seed=4)[0])
        ends = 0
        for action in actions:
            a = parallel.step(action)
            b = whole.step(action)
            assert_same_step(a, b)
            ends += int(np.sum(a[2] | a[3]))
    assert ends >= 6  # episodes ended (crashes and the 1 s limit) and restarted


def test_parallel_and_serial_parts_agree_on_random_courses():
    """Random courses drawn from each part's own seed, renewed on the way: the
    workers give what the same parts give in one process, bit for bit."""
    part = CourseSetEnvPart(
        partial(random_course, ranges=CourseRanges(time_limit=1.0)),
        k=2, n=2, renew_every=7, nearest=4,
    )  # fmt: skip
    actions = scripted_actions(30, 8, seed=1)
    serial = SerialEnv(part, 2)
    with ParallelEnv(part, 2) as parallel:
        np.testing.assert_array_equal(
            parallel.reset(seed=3)[0], serial.reset(seed=3)[0]
        )
        for action in actions:
            assert_same_step(parallel.step(action), serial.step(action))
    # The two parts flew different courses (their own seeds).
    rooms = [env.envs[0].course.room.size for env in serial.envs]
    assert not np.array_equal(rooms[0], rooms[1])


class FailingPart:
    """A part that fails when it is built (index 1) or at its third step."""

    def __init__(self, fail_on_build: bool) -> None:
        self.fail_on_build = fail_on_build

    def __call__(self, index: int):
        if self.fail_on_build and index == 1:
            raise RuntimeError("cannot build part 1")
        env = CourseEnv(SHORT, n=1)
        original = env.step
        count = {"steps": 0}

        def step(action):
            count["steps"] += 1
            if count["steps"] == 3:
                raise RuntimeError("step 3 failed")
            return original(action)

        env.step = step
        return env


def test_an_error_in_a_worker_reaches_the_main_process():
    with pytest.raises(WorkerError, match="cannot build part 1"):
        ParallelEnv(FailingPart(fail_on_build=True), 2)
    env = ParallelEnv(FailingPart(fail_on_build=False), 2)
    env.reset()
    env.step(np.zeros((2, 4)))
    env.step(np.zeros((2, 4)))
    with pytest.raises(WorkerError, match="step 3 failed"):
        env.step(np.zeros((2, 4)))
    env.close()
    env.close()  # closing twice is harmless
    assert not any(p.is_alive() for p in env.processes)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        SerialEnv(CourseEnvPart(SHORT, 1), 0)
    with pytest.raises(ValueError):
        make_parallel(CourseEnvPart(SHORT, 1), 0)
    serial = SerialEnv(CourseEnvPart(SHORT, 2), 2)
    serial.reset()
    with pytest.raises(ValueError):
        serial.step(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        train(SHORT, 64, "unused", n=6, workers=4)
    # One worker: the bare part, no process.
    assert isinstance(make_parallel(CourseEnvPart(SHORT, 2), 1), CourseEnv)


def test_a_training_is_the_same_with_any_number_of_workers(tmp_path):
    """The same short training with one worker and with two: the same log, bit
    for bit (the drones fly the same, and the network sees the same batch)."""
    settings = PPOSettings(rollout_steps=16, minibatch_size=64, epochs=2)

    def log(workers: int) -> list[dict]:
        out = tmp_path / str(workers)
        train(
            SHORT, 2 * 16 * 8, out, n=8, seed=3, settings=settings,
            evaluate_every=1, margin=0.0, workers=workers,
        )  # fmt: skip
        with open(out / "log.csv", encoding="ascii") as file:
            return [
                {k: v for k, v in row.items() if k != "seconds"}
                for row in csv.DictReader(file)
            ]

    one, two = log(1), log(2)
    assert len(one) == 2 and one == two
