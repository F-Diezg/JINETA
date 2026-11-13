"""The training environment split over several processes, and its speed.

    python -m examples.parallel_speed
    python -m examples.parallel_speed --drones 1024 --workers 1 2 4 8

The same drones (random actions on the first course, restarts included) flown
by one process and by several (`picadero.parallel.ParallelEnv`, each worker with
drones / workers of them). Prints environment steps per second for every
number of workers, and the speed-up over one. Run it with nothing else busy:
every worker takes a core. (On Windows set `$env:OPENBLAS_NUM_THREADS = 1`
first, as for a training.)
"""

import argparse
import time

import numpy as np

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.picadero.parallel import CourseEnvPart, make_parallel


def speed(env, steps: int, seed: int = 1) -> float:
    """Environment steps per second (one per drone) with random actions."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    actions = env.action_space.sample(rng, env.n * steps).reshape(steps, env.n, -1)
    env.step(actions[0])  # the first step pays for warming up
    begin = time.perf_counter()
    for action in actions[1:]:
        env.step(action)
    return (steps - 1) * env.n / (time.perf_counter() - begin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drones", type=int, default=1024)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    course = load_course(COURSE_FILE)
    print(f"{course.name}: {args.drones} drones, random actions")
    print("  workers   steps/s   speed-up")
    first = None
    for workers in args.workers:
        if args.drones % workers != 0:
            print(f"  {workers:7d}   (does not divide {args.drones})")
            continue
        env = make_parallel(CourseEnvPart(course, args.drones // workers), workers)
        try:
            rate = speed(env, args.steps)
        finally:
            env.close()
        first = first or rate
        print(f"  {workers:7d}  {rate:8.0f}   {rate / first:6.2f}x")


if __name__ == "__main__":
    main()
