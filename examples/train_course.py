"""Train the Iris to fly the first course with reinforcement learning (PPO).

    python -m examples.train_course
    python -m examples.train_course --steps 20e6 --refine 10e6 --drones 256 --seed 1
    python -m examples.train_course --workers 8      # the drones split over 8 cores

Two stages: first the policy learns to fly the course (`--steps`), then it keeps
learning with every sphere of its hitbox 5 cm larger (`--refine`), so that it
keeps the 5 cm from everything that the baseline's robust laps keep. (With the
margin from the start, crashes are so frequent that the policy learns to wait
until the time limit.) Every update prints a line (how the episodes ended, the
lap times, the spread of the actions); every 10 updates the deterministic policy
(the mean action) flies a lap. Each stage writes its log (`log.csv`) and agents
(`best.npz`, the fastest lap so far, preferring laps that keep the margin;
`last.npz`) to `<out>/fly` and `<out>/margin`. At the end the best agent is
measured against the classical baseline (6.77 s); `python -m
examples.policy_course <out>/margin/best.npz` draws its lap.

Training is repeatable: the same seed on the same computer gives the same
numbers. On another computer the numbers differ from the last digits on (the
matrix products), and a training is a long chain of them, so its result
differs too: a seed is not a recipe for the same policy everywhere.
"""

import argparse
import time

import numpy as np

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.ppo import PPOSettings
from jineta.picadero.training import (
    TRAINING_MARGIN,
    TRAINING_WEIGHTS,
    evaluate,
    train,
)

BASELINE_LAP = 6.773  # [s], examples.baseline_course


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=float, default=20e6, help="first stage")
    parser.add_argument("--refine", type=float, default=10e6, help="with the margin")
    parser.add_argument("--drones", type=int, default=256, help="drones at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--workers", type=int, default=1, help="processes (must divide --drones)"
    )
    parser.add_argument("--out", default="runs/first_course", help="output folder")
    args = parser.parse_args()

    course = load_course(COURSE_FILE)
    settings = PPOSettings()
    begin = time.perf_counter()
    print(
        f"{course.name}: {args.drones} drones ({args.workers} processes), "
        f"{args.steps:.3g} steps, seed {args.seed}, reward {TRAINING_WEIGHTS}"
    )

    def report(row: dict) -> None:
        lap = row["mean_lap_time"]
        line = (
            f"{row['steps'] / 1e6:7.2f} M steps {row['seconds']:6.0f} s | "
            f"episodes {row['episodes']:4d}: finished {row['finished']:4d} "
            f"crashed {row['crashed']:4d} timeout {row['timeout']:3d} | "
            f"return {row['mean_return']:6.2f} | lap {lap:5.2f} s | "
            f"spread {row['std']:.3f}"
        )
        if row.get("eval_status") is not None:
            line += f" | mean action: {row['eval_status']}"
            if row["eval_status"] == "finished":
                line += f" {row['eval_lap_time']:.2f} s"
            else:
                line += f" ({row['eval_gates']} gates)"
        print(line, flush=True)

    print("Stage 1: learning to fly the course")
    agent, _ = train(
        course,
        int(args.steps),
        f"{args.out}/fly",
        n=args.drones,
        seed=args.seed,
        settings=settings,
        margin=0.0,
        report=report,
        workers=args.workers,
    )
    print(f"Stage 2: keeping {100 * TRAINING_MARGIN:.0f} cm from everything")
    agent, _ = train(
        course,
        int(args.refine),
        f"{args.out}/margin",
        n=args.drones,
        seed=args.seed + 1,
        settings=PPOSettings(learning_rate=1e-4),
        margin=TRAINING_MARGIN,
        agent=agent,
        report=report,
        workers=args.workers,
    )
    minutes = (time.perf_counter() - begin) / 60.0
    result = evaluate(agent, course, n=256, seed=args.seed)
    print(f"\nTrained in {minutes:.1f} min. Best agent ({args.out}/margin/best.npz):")
    print(
        f"  mean action: {result.status}"
        + (f" in {result.lap_time:.2f} s" if result.finished else "")
        + f", {result.gates} gates, closest approach {100 * result.clearance:.1f} cm"
    )
    print(
        f"  with its noise: {100 * result.success_rate:.0f} % of 256 laps finish"
        + (
            f", in {result.mean_lap_time:.2f} s on average"
            if np.isfinite(result.mean_lap_time)
            else ""
        )
    )
    print(f"  classical baseline: {BASELINE_LAP:.2f} s")


if __name__ == "__main__":
    main()
