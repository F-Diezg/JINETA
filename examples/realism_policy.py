"""How a policy flies with each effect of a real flight (phase 12).

    python -m examples.realism_policy
    python -m examples.realism_policy runs/first_course/margin/best.npz

Flies a trained policy (by default the phase 11 policy,
`policies/first_course.npz`, trained in the ideal simulator) on the first course
with each effect of `picadero.realism.Realism` on its own, and with all of them,
and prints its deterministic lap (status, lap time, closest approach) and how
many of 64 laps flown with its own noise finish.
"""

import argparse
import dataclasses

import numpy as np

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.policy import Agent
from jineta.escuadron.referee import STATUS_NAMES
from jineta.picadero.env import CourseEnv
from jineta.picadero.realism import Realism
from jineta.picadero.training import fly


def conditions() -> dict[str, Realism]:
    """Every effect on its own, then all of them."""
    effects = {"ideal": Realism()}
    every = {}
    for field in dataclasses.fields(Realism):
        if field.type in (bool, "bool"):
            effects[field.name] = Realism(**{field.name: True})
            every[field.name] = True
    effects["all"] = Realism(**every)
    return effects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("policy", nargs="?", default="policies/first_course.npz")
    args = parser.parse_args()
    agent = Agent.load(args.policy)
    nearest = int(agent.info.get("nearest", 0))
    course = load_course(COURSE_FILE)
    print(f"{args.policy} on {course.name}")
    print("  effect          mean action                       noisy laps")
    for name, realism in conditions().items():
        lap = fly(agent, CourseEnv(course, nearest=nearest, realism=realism))
        noisy = fly(
            agent,
            CourseEnv(course, n=64, nearest=nearest, realism=realism),
            np.random.default_rng(0),
        )
        status = STATUS_NAMES[lap["status"][0]]
        time = f"{lap['lap_time'][0]:5.2f} s" if status == "finished" else "       "
        finished = np.mean(noisy["status"] == 1)
        print(
            f"  {name:14s}  {status:9s} {time}  closest "
            f"{100 * lap['clearance'][0]:5.1f} cm   {100 * finished:5.1f} %"
        )


if __name__ == "__main__":
    main()
