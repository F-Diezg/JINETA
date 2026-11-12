"""Train the Iris on random courses (gates and obstacles), and measure it on
courses it never saw against the classical baseline swept on each.

    python -m examples.train_random                     # train, then compare
    python -m examples.train_random --steps 100e6 --courses 8 --drones 128
    python -m examples.train_random --compare policies/random_courses.npz

Training: k courses at once, n drones each, every course replaced by a new one
every 500 steps (10 s of flight); the drones see the 4 nearest surfaces. Every
10 updates the mean action flies 8 validation courses; the agent that finishes
the most (then the fastest) is kept (`best.npz`).

Comparison: on 20 test courses (never trained on, nor used to pick the agent),
the policy's mean action flies one lap, and the baseline is swept on each course
(a smaller grid than the first course's, ~10 s per course) for its fastest
robust lap.
"""

import argparse
import time

import numpy as np

from jineta.doma.policy import Agent
from jineta.picadero.baseline import sweep
from jineta.picadero.training import (
    TEST_SEED,
    evaluate_courses,
    random_courses,
    train_random,
)

# The grid the baseline is swept over on each random course.
GRID = {
    "approach": (1.5,),
    "cruise": (2.5, 3.5, 5.0, 6.5),
    "lateral_accel": (3.0, 5.0, 8.0),
    "braking_accel": (2.0, 3.5, 5.0),
    "lookahead": (1.5, 2.5, 3.0),
}


def compare(agent: Agent, count: int = 20) -> None:
    """The policy against the baseline on `count` test courses."""
    courses = random_courses(count, TEST_SEED)
    nearest = int(agent.info.get("nearest", 4))
    policy = evaluate_courses(agent, courses, nearest=nearest)
    print(f"\n{count} test courses (seed {TEST_SEED}), policy {agent.info}:")
    print("  course  gates obst | policy            | baseline")
    wins = both = 0
    baseline_times = []
    for i, course in enumerate(courses):
        try:
            best = sweep(course, **GRID).lap_time
        except RuntimeError:
            best = float("nan")
        baseline_times.append(best)
        mine = policy.lap_times[i]
        if np.isfinite(mine):
            text = f"{mine:6.2f} s          "
        else:
            text = f"no ({policy.gates[i]} gates)     "
        print(
            f"  {i:6d}  {len(course.gates):5d} {len(course.obstacles):4d} | {text}| "
            + (f"{best:6.2f} s" if np.isfinite(best) else "no robust lap")
        )
        if np.isfinite(mine) and np.isfinite(best):
            both += 1
            wins += mine < best
    baseline_times = np.array(baseline_times)
    print(
        f"  policy finished {policy.finished}/{count}, baseline "
        f"{int(np.sum(np.isfinite(baseline_times)))}/{count}; where both finished "
        f"({both}), the policy was faster on {wins}"
    )
    done = np.isfinite(policy.lap_times) & np.isfinite(baseline_times)
    if np.any(done):
        ratio = np.mean(policy.lap_times[done] / baseline_times[done])
        print(f"  policy lap / baseline lap on those: {ratio:.3f} on average")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=float, default=100e6)
    parser.add_argument("--courses", type=int, default=8, help="courses at once")
    parser.add_argument("--drones", type=int, default=128, help="drones per course")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/random_courses")
    parser.add_argument("--compare", help="only compare this agent (.npz)")
    args = parser.parse_args()

    if args.compare:
        compare(Agent.load(args.compare))
        return
    begin = time.perf_counter()

    def report(row: dict) -> None:
        line = (
            f"{row['steps'] / 1e6:7.2f} M steps {row['seconds']:6.0f} s | "
            f"finished {row['finished']:4d} crashed {row['crashed']:4d} "
            f"timeout {row['timeout']:3d} | return {row['mean_return']:6.2f} | "
            f"spread {row['std']:.3f}"
        )
        if row.get("eval_status") is not None:
            line += f" | validation: {row['eval_status']}"
            if np.isfinite(row["eval_lap_time"]):
                line += f", {row['eval_lap_time']:.2f} s"
        print(line, flush=True)

    agent, _ = train_random(
        int(args.steps),
        args.out,
        k=args.courses,
        n=args.drones,
        seed=args.seed,
        report=report,
    )
    print(f"\nTrained in {(time.perf_counter() - begin) / 60:.1f} min.")
    compare(agent)


if __name__ == "__main__":
    main()
