"""Train the Iris to fly the first course for real: every effect of a real lab,
a different drone every episode, and a stop after the finish (phase 12).

    python -m examples.train_realism --workers 8
    python -m examples.train_realism --start policies/first_course.npz --workers 8
    python -m examples.train_realism --steps 20e6 --refine 10e6 --drones 1024

The drones fly in `RANDOMISED_LAB` (motor lag, ground effect, motion-capture
sensors, latency 0-40 ms, mass, inertia, thrust, drag and motor lag drawn every
episode) and must stop after the last gate (`StopAfterFinish`).

From scratch (the default) it trains in two stages as phase 11 did: first
without the 5 cm margin (`--steps`), then with it (`--refine`), from the first
stage's best agent. With `--start` it continues a trained agent (phase 11's
policy, widened by one input for the stop's flag) with the margin from the
start, at a smaller learning rate, for `--refine` steps.

Every 10 updates the mean action flies 64 drawn drones; the agent that flies
the most of them (then the fastest) is kept (`best.npz`). At the end the best
agent is measured on 256 drones it never flew (another seed), next to the
phase 11 policy and the classical baseline in the same conditions (the
baseline through the same interface, reading the same estimate; it was not
designed to stop, and on this course it cannot: it is measured without the
stop).
"""

import argparse
import dataclasses
import time

import numpy as np

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.policy import Agent
from jineta.doma.ppo import PPOSettings
from jineta.picadero.baseline import BaselineAgent, BaselineSettings
from jineta.picadero.env import CourseEnv
from jineta.picadero.finish import StopAfterFinish
from jineta.picadero.realism import RANDOMISED_LAB
from jineta.picadero.training import (
    TRAINING_MARGIN,
    TRAINING_WEIGHTS,
    evaluate_drawn,
    fly,
    train,
)

STOP = StopAfterFinish()
# The baseline's best settings on the first course (examples.baseline_course).
BASELINE = BaselineSettings(5.0, 8.0, 5.0, 3.0, approach=2.0)
MEASURE_SEED = 12345  # drones never flown in training (which judges with seed 0)


def measure(agent: Agent, course, label: str, n: int = 256) -> None:
    """The mean action on n drawn drones (the stop included when the agent has
    the flag)."""
    stop = STOP if agent.info.get("stop") else None
    result = evaluate_drawn(
        agent, course, n, MEASURE_SEED, realism=RANDOMISED_LAB, stop=stop,
        nearest=int(agent.info.get("nearest", 0)),
    )  # fmt: skip
    done = np.isfinite(result.lap_times)
    closest = np.min(result.clearances[done]) if np.any(done) else float("nan")
    print(
        f"  {label}: {100 * result.success_rate:.1f} % of {n} drawn drones finish"
        + (" and stop" if stop else "")
        + (
            f", in {result.mean_lap_time:.2f} s on average, closest "
            f"{100 * closest:.1f} cm"
            if np.any(done)
            else ""
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=float, default=20e6, help="first stage")
    parser.add_argument("--refine", type=float, default=10e6, help="with the margin")
    parser.add_argument("--start", help="continue this agent (.npz)")
    parser.add_argument("--drones", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/realism")
    parser.add_argument(
        "--no-stop", action="store_true", help="laps end at the last gate"
    )
    parser.add_argument(
        "--overrun", type=float, default=1.0, help="penalty per metre after the finish"
    )
    args = parser.parse_args()

    course = load_course(COURSE_FILE)
    begin = time.perf_counter()

    def report(row: dict) -> None:
        line = (
            f"{row['steps'] / 1e6:7.2f} M steps {row['seconds']:6.0f} s | "
            f"finished {row['finished']:4d} crashed {row['crashed']:4d} "
            f"timeout {row['timeout']:3d} | return {row['mean_return']:6.2f} | "
            f"lap {row['mean_lap_time']:5.2f} s | spread {row['std']:.3f}"
        )
        if row.get("eval_status") is not None:
            line += f" | drawn drones: {row['eval_status']}"
            if np.isfinite(row["eval_lap_time"]):
                line += f" {row['eval_lap_time']:.2f} s"
        print(line, flush=True)

    stop = None if args.no_stop else STOP
    weights = dataclasses.replace(TRAINING_WEIGHTS, overrun=args.overrun)
    common = {
        "n": args.drones, "realism": RANDOMISED_LAB, "stop": stop,
        "report": report, "workers": args.workers, "weights": weights,
    }  # fmt: skip
    if args.start:
        agent = Agent.load(args.start)
        if stop is not None and not agent.info.get("stop"):
            agent = agent.widened(1)  # the stop's flag
        print(f"Continuing {args.start} with every effect and the margin")
        agent, _ = train(
            course, int(args.refine), f"{args.out}/margin", seed=args.seed,
            settings=PPOSettings(learning_rate=1e-4), margin=TRAINING_MARGIN,
            agent=agent, **common,
        )  # fmt: skip
    else:
        print("Stage 1: every effect, no margin")
        agent, _ = train(
            course, int(args.steps), f"{args.out}/fly", seed=args.seed,
            margin=0.0, **common,
        )  # fmt: skip
        print(f"Stage 2: keeping {100 * TRAINING_MARGIN:.0f} cm from everything")
        agent, _ = train(
            course, int(args.refine), f"{args.out}/margin", seed=args.seed + 1,
            settings=PPOSettings(learning_rate=1e-4), margin=TRAINING_MARGIN,
            agent=agent, **common,
        )  # fmt: skip
    minutes = (time.perf_counter() - begin) / 60.0
    print(f"\nTrained in {minutes:.1f} min. In the randomised lab:")
    measure(agent, course, f"best agent ({args.out}/margin/best.npz)")
    measure(Agent.load("policies/first_course.npz"), course, "phase 11 policy")
    env = CourseEnv(course, n=256, realism=RANDOMISED_LAB)
    lap = fly(BaselineAgent(env, BASELINE), env, seed=MEASURE_SEED)
    done = lap["status"] == 1
    print(
        f"  classical baseline (no stop): {100 * np.mean(done):.1f} % finish"
        + (f", in {np.mean(lap['lap_time'][done]):.2f} s" if np.any(done) else "")
    )


if __name__ == "__main__":
    main()
