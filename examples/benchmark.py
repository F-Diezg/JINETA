"""How fast does the reference engine fly? A lap of the first course, timed.

The lap is the one of examples/iris_course.py: the Iris with its velocity
autopilot, the ground contact and the referee (13 collision spheres, 3 obstacles,
4 gates), at 100 Hz. It is flown several times and the fastest run is kept: the
slower ones were slowed down by something else running on the computer, never by
the simulator itself.

    python -m examples.benchmark              steps per second, real-time factor
    python -m examples.benchmark --profile    and where the time goes

The profile (Python's cProfile) lists the functions by "tottime", the time spent
inside each function itself, not counting the functions it calls ("cumtime" counts
them). Profiling slows every function call down, so its total is longer than the
timed lap; what matters is the share of each function.
"""

import argparse
import cProfile
import platform
import pstats
import time

import numpy as np

from examples.iris_course import COURSE_FILE, DT, fly_course
from jineta.campo.course import Course, load_course


def time_lap(course: Course, repeats: int) -> tuple[int, float]:
    """Fly the lap `repeats` times; return the number of steps and the fastest
    wall-clock time [s]."""
    best = np.inf
    steps = 0
    for _ in range(repeats):
        start = time.perf_counter()
        times, _, _ = fly_course(course)
        best = min(best, time.perf_counter() - start)
        steps = len(times) - 1
    return steps, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=5, help="laps to time")
    parser.add_argument(
        "--profile", action="store_true", help="also show where the time goes"
    )
    args = parser.parse_args()

    course = load_course(COURSE_FILE)
    steps, seconds = time_lap(course, args.repeats)
    rate = steps / seconds
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{platform.processor() or platform.machine()}"
    )
    print(f"{course.name}: {steps} steps in {seconds:.3f} s (best of {args.repeats})")
    print(f"  {rate:.0f} steps/s, {rate * DT:.1f} x real time")

    if args.profile:
        with cProfile.Profile() as profiler:
            fly_course(course)
        print()
        pstats.Stats(profiler).sort_stats("tottime").print_stats(15)


if __name__ == "__main__":
    main()
