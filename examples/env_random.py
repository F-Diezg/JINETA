"""The training environment flown with random actions, and its speed.

A policy that knows nothing acts at random: it crashes within a second or two,
and every crash restarts that drone at once. This is what the first minutes of
training look like, and how fast the environment runs.

    python -m examples.env_random

prints how the random episodes ended and the speed of the environment against n:
environment steps per second (each is two physics steps, 50 Hz), and the same in
drone-seconds of flight per second (times real time).
"""

import time

import numpy as np

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.escuadron.referee import STATUS_NAMES
from jineta.picadero.env import CourseEnv


def random_episodes(env: CourseEnv, steps: int, seed: int = 0) -> list[tuple]:
    """Fly `steps` environment steps with uniform random actions; returns the
    episodes that ended as (status, steps, return)."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    ended = []
    for _ in range(steps):
        action = env.action_space.sample(rng, env.n)
        _, _, terminated, truncated, info = env.step(action)
        for i in np.flatnonzero(terminated | truncated):
            ended.append(
                (
                    STATUS_NAMES[info["status"][i]],
                    int(info["episode_steps"][i]),
                    float(info["episode_return"][i]),
                )
            )
    return ended


def speed(env: CourseEnv, steps: int) -> float:
    """Environment steps per second (drone-steps: one per drone)."""
    rng = np.random.default_rng(1)
    env.reset()
    actions = env.action_space.sample(rng, env.n * steps).reshape(steps, env.n, -1)
    begin = time.perf_counter()
    for action in actions:
        env.step(action)
    return steps * env.n / (time.perf_counter() - begin)


def main() -> None:
    course = load_course(COURSE_FILE)
    env = CourseEnv(course, n=64)
    episodes = random_episodes(env, 500)
    print(f"{course.name}: 64 drones, 500 random actions each (10 s of flight)")
    for status in ("crashed", "timeout", "finished"):
        mine = [e for e in episodes if e[0] == status]
        if mine:
            length = np.mean([e[1] for e in mine]) * env.step_time
            reward = np.mean([e[2] for e in mine])
            print(
                f"  {status:8s}: {len(mine):4d} episodes, "
                f"{length:5.2f} s long, return {reward:6.2f} on average"
            )

    print("\nSpeed of the environment (random actions, restarts included):")
    for n in (1, 16, 64, 256, 1024):
        rate = speed(CourseEnv(course, n=n), 200 if n < 1024 else 100)
        real = rate * 2 * env.dt
        print(f"  n = {n:5d}: {rate:8.0f} drone-steps/s, {real:6.1f} x real time")


if __name__ == "__main__":
    main()
