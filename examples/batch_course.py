"""Many Iris fly the first course at once, in the batch engine.

The same flight as examples/iris_course.py (take off from the floor, then straight
at each gate centre at a constant speed, nose along the gate's heading), for n
drones at once, each at its own speed and with its own referee: a slow drone
finishes, a fast one may clip a gate frame. Each lap ends on its own; with
`laps > 1` a drone starts its next lap (standing on the floor again) as soon as
its last one ends, while the others fly on.

    python -m examples.batch_course

prints how every drone's laps ended, and how fast the batch flew.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from examples.iris_course import COURSE_FILE, DT, TAKEOFF_ALTITUDE
from jineta.campo.course import Course, load_course
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.control import BatchAutopilot
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import STATUS_NAMES, batch_referee
from jineta.escuadron.simulation import BatchFlight
from jineta.jinete.contact import resting_height
from jineta.jinete.profiles import IRIS, IRIS_GAINS, IRIS_HITBOX, IRIS_LANDING_GEAR


@dataclass
class GateGuidance:
    """The guidance of examples/iris_course.py for n drones: climb straight up to
    TAKEOFF_ALTITUDE, then fly at `speeds` (n,) [m/s] straight at the centre of each
    drone's next gate, nose along its heading; stop once every gate is crossed.
    Callable as the batch controller: it sets the autopilot's commands and runs it.
    """

    course: Course
    autopilot: BatchAutopilot
    flight: BatchFlight
    speeds: np.ndarray
    climbed: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.speeds = np.broadcast_to(np.asarray(self.speeds, float), (self.n,))
        self.climbed = np.zeros(self.n, dtype=bool)
        self._centers = np.array([gate.center for gate in self.course.gates])
        self._yaws = np.array([gate.yaw for gate in self.course.gates])

    @property
    def n(self) -> int:
        return self.flight.n

    def reset(self, drones: np.ndarray) -> None:
        """These drones start a lap standing on the floor."""
        self.climbed[drones] = False
        self.autopilot.reset(drones)
        self.autopilot.landed[drones] = True

    def __call__(self, clock: np.ndarray, states: np.ndarray) -> None:
        gate = self.flight.referee.next_gate  # -1 once every gate is crossed
        self.climbed = self.climbed | (-states[:, 2] > TAKEOFF_ALTITUDE)
        to_gate = self._centers[np.maximum(gate, 0)] - states[:, 0:3]
        distance = np.linalg.norm(to_gate, axis=1, keepdims=True)
        command = self.speeds[:, None] * to_gate / distance
        command = np.where(self.climbed[:, None], command, [0.0, 0.0, -1.0])
        has_gate = gate >= 0
        self.autopilot.velocity_command = np.where(has_gate[:, None], command, 0.0)
        self.autopilot.yaw_command = np.where(
            has_gate, self._yaws[np.maximum(gate, 0)], self.autopilot.yaw_command
        )
        self.autopilot(clock, states)


def course_flight(
    course: Course, speeds: np.ndarray
) -> tuple[BatchFlight, GateGuidance]:
    """n = len(speeds) Iris standing at the course's start, landed, with the batch
    referee of the course and the gate guidance."""
    n = len(speeds)
    vehicles = BatchMultirotor(IRIS, n, contact=BatchGroundContact(IRIS_LANDING_GEAR))
    referee = batch_referee(
        course, IRIS_HITBOX.centers, IRIS_HITBOX.radii, IRIS_LANDING_GEAR.points, n
    )
    start = course.start_state(resting_height(IRIS_LANDING_GEAR, IRIS.mass))
    flight = BatchFlight(vehicles, referee, DT, np.tile(start, (n, 1)))
    autopilot = BatchAutopilot(vehicles, IRIS_GAINS, DT, landed=True)
    return flight, GateGuidance(course, autopilot, flight, speeds)


def fly_laps(
    course: Course, speeds: np.ndarray, laps: int = 1
) -> tuple[list[list[tuple[str, float]]], np.ndarray, int]:
    """Fly `laps` laps with n = len(speeds) drones, each starting its next lap as
    soon as its last one ends. Returns, for every drone, its laps as (status, end
    time); the states of every step until every drone has ended its first lap,
    (steps + 1, n, 13) (a drone that starts its next lap goes on in it, one that has
    no more laps stays where it ended); and the number of steps flown."""
    flight, guidance = course_flight(course, speeds)
    start = flight.states[0].copy()
    results: list[list[tuple[str, float]]] = [[] for _ in range(flight.n)]
    history = [flight.states.copy()]
    steps = 0
    while min(len(r) for r in results) < laps:
        ended = flight.step(guidance)
        steps += 1
        if steps * DT > laps * (course.time_limit + 1.0):
            raise RuntimeError("laps did not end within their time limits")
        if min(len(r) for r in results) == 0:
            history.append(flight.states.copy())
        done = np.flatnonzero(ended)
        for i in done:
            status = STATUS_NAMES[flight.referee.status[i]]
            results[i].append((status, float(flight.referee.end_time[i])))
        restart = [i for i in done if len(results[i]) < laps]
        if restart:
            flight.reset(restart, start)
            guidance.reset(np.array(restart))
    return results, np.array(history), steps


def main() -> None:
    course = load_course(COURSE_FILE)
    speeds = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
    results, _, _ = fly_laps(course, speeds, laps=2)
    print(f"{course.name}: {len(speeds)} Iris at once, two laps each")
    for speed, laps in zip(speeds, results, strict=True):
        ends = ", ".join(f"{status} at {end:.2f} s" for status, end in laps)
        print(f"  {speed:.1f} m/s: {ends}")

    print("\nSpeed of the batch (first course, gate guidance, autopilot, referee):")
    for n in (1, 16, 64, 256, 1024):
        flight, guidance = course_flight(course, np.full(n, 3.0))
        start = flight.states[0].copy()
        begin = time.perf_counter()
        for _ in range(300):
            ended = flight.step(guidance)
            if np.any(ended):
                flight.reset(ended, start)
                guidance.reset(ended)
        rate = 300 * n / (time.perf_counter() - begin)
        print(f"  n = {n:5d}: {rate:9.0f} drone-steps/s, {rate * DT:7.1f} x real time")


if __name__ == "__main__":
    main()
