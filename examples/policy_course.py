"""Fly a trained policy on the first course and draw its lap next to the
classical baseline's.

    python -m examples.policy_course                      # policies/first_course.npz
    python -m examples.policy_course runs/first_course/best.npz

Prints the policy's lap (mean action), its success rate with its own noise, and
the baseline's lap; the figure shows both paths from above and from the side,
the policy's coloured by speed.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.iris_course import COURSE_FILE, draw_course
from jineta.campo.course import load_course
from jineta.doma.policy import Agent
from jineta.picadero.baseline import BaselineSettings, baseline_flight
from jineta.picadero.env import CourseEnv
from jineta.picadero.training import evaluate, fly

POLICY_FILE = Path(__file__).resolve().parent.parent / "policies" / "first_course.npz"
# The baseline's best settings on the first course (examples.baseline_course).
BASELINE = BaselineSettings(5.0, 8.0, 5.0, 3.0, approach=2.0)


def baseline_states(course) -> tuple[np.ndarray, float]:
    """The baseline's lap of a course: its states and lap time."""
    flight, pilot = baseline_flight(course, BASELINE, 1)
    states = [flight.states[0].copy()]
    while not flight.referee.ended[0]:
        flight.step(pilot)
        states.append(flight.states[0].copy())
    return np.array(states), float(flight.referee.lap_time[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", nargs="?", default=str(POLICY_FILE))
    args = parser.parse_args()

    course = load_course(COURSE_FILE)
    agent = Agent.load(args.policy)
    nearest = int(agent.info.get("nearest", 0))
    lap = fly(agent, CourseEnv(course, n=1, nearest=nearest), record=True)
    result = evaluate(agent, course, n=256, nearest=nearest)
    states = lap["states"]
    speed = np.linalg.norm(states[:, 3:6], axis=1)
    reference, reference_lap = baseline_states(course)

    print(f"{course.name}, policy {args.policy} ({agent.info}):")
    if result.finished:
        print(f"  mean action: finished in {result.lap_time:.2f} s", end="")
    else:
        print(f"  mean action: {result.status} after {result.gates} gates", end="")
    print(f", closest approach {100 * result.clearance:.1f} cm, top speed ", end="")
    print(f"{speed.max():.2f} m/s")
    print(f"  with its noise: {100 * result.success_rate:.0f} % of 256 laps finish")
    print(f"  classical baseline: {reference_lap:.2f} s")

    fig, (top, side) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    draw_course(course, top, side)
    top.plot(
        reference[:, 1], reference[:, 0], "--", color="0.5", lw=1, label="baseline"
    )
    side.plot(reference[:, 0], -reference[:, 2], "--", color="0.5")
    dots = top.scatter(states[:, 1], states[:, 0], c=speed, s=4, cmap="viridis")
    side.scatter(states[:, 0], -states[:, 2], c=speed, s=4, cmap="viridis")
    fig.colorbar(dots, ax=side, label="policy speed [m/s]")
    top.legend(loc="lower right")
    top.set(xlabel="East [m]", ylabel="North [m]", title="From above", aspect="equal")
    side.set(
        xlabel="North [m]", ylabel="Altitude [m]", title="From the side", aspect="equal"
    )
    outcome = f"lap {result.lap_time:.2f} s" if result.finished else result.status
    fig.suptitle(
        f"{course.name}: learned policy ({outcome}) against the baseline "
        f"({reference_lap:.2f} s)"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
