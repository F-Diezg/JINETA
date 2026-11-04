"""The Iris flies the first course: take off, through the gate centres, finish.

A course file (courses/first_course.yaml) gives the room, the gates in order and
the obstacles. The Iris stands on the floor at the start with its autopilot
landed. Its guidance is the simplest there is: climb to take off, then ask the
velocity autopilot to fly at a constant speed straight at the centre of the next
gate, nose along the gate's heading. The referee watches the lap: gates in order,
collisions (13 spheres against walls, obstacles and gate frames; the gear against
the floor after take-off) and the time limit; the run ends with the lap.

This is a demonstration of the indoor world, not the classical baseline of phase
10: going straight at each centre ignores where the next gate is, so it brakes
and turns late.

The figure shows the course from above and from the side, with the path coloured
by speed and the gate crossings marked.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from jineta.campo.course import Course, load_course
from jineta.campo.gates import RectangularGate
from jineta.campo.indoor import Box
from jineta.campo.referee import Referee
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import run
from jineta.jinete.contact import GroundContact, resting_height
from jineta.jinete.control import Autopilot
from jineta.jinete.multirotor import Multirotor
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_HITBOX, IRIS_LANDING_GEAR

DT = 0.01  # 100 Hz, simulation and control
COURSE_FILE = Path(__file__).resolve().parent.parent / "courses" / "first_course.yaml"
TAKEOFF_ALTITUDE = 0.5  # climb straight up this far before going anywhere [m]


def fly_course(
    course: Course, speed: float = 3.0
) -> tuple[np.ndarray, np.ndarray, Referee]:
    """Fly one lap of `course` with the Iris at `speed` [m/s] towards each gate
    centre. Returns the times, the states and the referee (how the lap ended)."""
    vehicle = Multirotor(IRIS, contact=GroundContact(IRIS_LANDING_GEAR))
    autopilot = Autopilot(vehicle, IRIS_GAINS, DT, landed=True)
    referee = course.referee(
        IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points
    )
    climbed = False

    def pre_step(t: float, y: np.ndarray) -> None:
        nonlocal climbed
        gate = referee.next_gate
        climbed = climbed or -y[2] > TAKEOFF_ALTITUDE
        if gate is None:
            command = np.zeros(3)
        elif not climbed:
            command = np.array([0.0, 0.0, -1.0])  # straight up (NED: up is -z)
        else:
            to_gate = course.gates[gate].center - y[0:3]
            command = speed * to_gate / np.linalg.norm(to_gate)
        autopilot.velocity_command = command
        if gate is not None:
            autopilot.yaw_command = course.gates[gate].yaw
        autopilot(t, y)

    start = course.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    times, states = run(
        vehicle.derivative,
        start,
        DT,
        course.time_limit + 1.0,
        stop=referee,
        pre_step=pre_step,
        post_step=normalize_attitude,
    )
    return times, states, referee


def draw_course(course: Course, top, side) -> None:
    """The room, obstacles and gates from above (North up) and from the side."""
    length, width, height = course.room.size
    top.add_patch(Rectangle((0, 0), width, length, fill=False, color="0.3"))
    side.add_patch(Rectangle((0, 0), length, height, fill=False, color="0.3"))
    for obstacle in course.obstacles:
        if isinstance(obstacle, Box):
            corners = (
                obstacle.size
                / 2
                * np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]])
            )
            c, s = np.cos(obstacle.yaw), np.sin(obstacle.yaw)
            north = obstacle.center[0] + c * corners[:, 0] - s * corners[:, 1]
            east = obstacle.center[1] + s * corners[:, 0] + c * corners[:, 1]
            top.add_patch(Polygon(np.c_[east, north], color="0.75"))
            bottom = -obstacle.center[2] - obstacle.size[2] / 2
            reach = np.max(np.abs(north - obstacle.center[0]))
            side.add_patch(
                Rectangle(
                    (obstacle.center[0] - reach, bottom),
                    2 * reach,
                    obstacle.size[2],
                    color="0.75",
                )
            )
        else:
            base = obstacle.base
            top.add_patch(Circle((base[1], base[0]), obstacle.radius, color="0.75"))
            side.add_patch(
                Rectangle(
                    (base[0] - obstacle.radius, -base[2]),
                    2 * obstacle.radius,
                    obstacle.height,
                    color="0.75",
                )
            )
    for i, gate in enumerate(course.gates):
        half = gate.width / 2 if isinstance(gate, RectangularGate) else gate.radius
        tall = gate.height / 2 if isinstance(gate, RectangularGate) else gate.radius
        right = np.array([-np.sin(gate.yaw), np.cos(gate.yaw)])  # (North, East)
        ends = gate.center[0:2] + np.outer([-half, half], right)
        top.plot(ends[:, 1], ends[:, 0], color="#2a78d6", lw=3)
        top.annotate(
            f"{i + 1}",
            (gate.center[1], gate.center[0]),
            textcoords="offset points",
            xytext=(6, 6),
        )
        reach = abs(half * right[0])
        altitude = -gate.center[2]
        side.add_patch(
            Rectangle(
                (gate.center[0] - reach - 0.05, altitude - tall),
                2 * reach + 0.1,
                2 * tall,
                fill=False,
                color="#2a78d6",
                lw=2,
            )
        )


def main() -> None:
    course = load_course(COURSE_FILE)
    _, states, referee = fly_course(course)
    speed = np.linalg.norm(states[:, 3:6], axis=1)

    print(f"{course.name}: {referee.status} at t = {referee.end_time:.2f} s")
    for p in referee.passes:
        print(f"  gate {p.gate + 1} at {p.time:.2f} s")
    if referee.monitor.collision is not None:
        hit = referee.monitor.collision
        print(f"  contact: {hit.kind} {hit.index} at {hit.time:.2f} s")
    print(f"  top speed {speed.max():.2f} m/s")

    fig, (top, side) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    draw_course(course, top, side)
    path = top.scatter(states[:, 1], states[:, 0], c=speed, s=4, cmap="viridis")
    side.scatter(states[:, 0], -states[:, 2], c=speed, s=4, cmap="viridis")
    for p in referee.passes:
        top.plot(p.point[1], p.point[0], "o", color="#eb6834")
        side.plot(p.point[0], -p.point[2], "o", color="#eb6834")
    fig.colorbar(path, ax=side, label="speed [m/s]")
    top.set(xlabel="East [m]", ylabel="North [m]", title="From above", aspect="equal")
    side.set(
        xlabel="North [m]", ylabel="Altitude [m]", title="From the side", aspect="equal"
    )
    lap = f"lap {referee.lap_time:.2f} s" if referee.lap_time else referee.status
    fig.suptitle(f"{course.name}: Iris through the gate centres at 3 m/s ({lap})")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
