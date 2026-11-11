"""The classical baseline on the first course: the time a policy has to beat.

The velocity autopilot follows a path through the gate centres (pure pursuit with
a speed plan for the corners, see jineta/picadero/baseline.py). The batch engine
flies every combination of its settings at once, each with four slightly changed
copies, and keeps the fastest lap that finishes robustly (no copy crashes, never
closer than 5 cm to anything). That lap is flown again and drawn: the planned
path, the flight coloured by speed, the gate crossings.

    python -m examples.baseline_course
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from examples.iris_course import COURSE_FILE, draw_course
from jineta.campo.course import load_course
from jineta.picadero.baseline import baseline_flight, sweep


def main() -> None:
    course = load_course(COURSE_FILE)
    begin = time.perf_counter()
    result = sweep(course)
    seconds = time.perf_counter() - begin
    best = result.best
    finished = np.isfinite(result.lap_times)
    print(
        f"{course.name}: {len(result.tried)} settings, 5 drones each, {seconds:.0f} s"
    )
    print(f"  finished: {finished.sum()}, robustly: {result.robust.sum()}")
    print(
        f"  best: approach {best.approach} m, cruise {float(best.cruise)} m/s, "
        f"lateral {float(best.lateral_accel)} m/s², braking "
        f"{float(best.braking_accel)} m/s², lookahead {float(best.lookahead)} m"
    )
    print(f"  time to beat: {result.lap_time:.3f} s (straight at each centre: 8.81 s)")

    flight, pilot = baseline_flight(course, best, 1)
    states = [flight.states[0].copy()]
    while not flight.referee.ended[0]:
        flight.step(pilot)
        states.append(flight.states[0].copy())
    states = np.array(states)
    speed = np.linalg.norm(states[:, 3:6], axis=1)
    referee = flight.referee
    print(f"  top speed {speed.max():.2f} m/s")

    fig, (top, side) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    draw_course(course, top, side)
    path = pilot.guidance.path.points
    top.plot(path[:, 1], path[:, 0], "--", color="0.5", lw=1, label="planned path")
    side.plot(path[:, 0], -path[:, 2], "--", color="0.5", lw=1)
    dots = top.scatter(states[:, 1], states[:, 0], c=speed, s=4, cmap="viridis")
    side.scatter(states[:, 0], -states[:, 2], c=speed, s=4, cmap="viridis")
    for gate in range(referee.passed[0]):
        point = referee.pass_point[0, gate]
        top.plot(point[1], point[0], "o", color="#eb6834")
        side.plot(point[0], -point[2], "o", color="#eb6834")
    fig.colorbar(dots, ax=side, label="speed [m/s]")
    top.legend(loc="lower right")
    top.set(xlabel="East [m]", ylabel="North [m]", title="From above", aspect="equal")
    side.set(
        xlabel="North [m]", ylabel="Altitude [m]", title="From the side", aspect="equal"
    )
    fig.suptitle(f"{course.name}: classical baseline (lap {referee.lap_time[0]:.2f} s)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
