"""Fly a saved agent again and record its laps, for the viewer (phase 13).

    python -m examples.record_agent                                # phase 11's policy
    python -m examples.record_agent runs/realism/margin/best.npz
    python -m examples.record_agent runs/realism/margin/agents/000004915200.npz
    python -m examples.record_agent policies/first_course.npz --realism lab

The agent flies the first course (or `--course`) as a training's evaluation
flies it: the mean action's lap and 8 laps with the policy's noise or, when the
realism draws (sensors, randomised drones), the mean action on 8 drawn drones.
By default it flies in what it was trained in (its info: the realism, the stop
after the finish, the surroundings it sees); `--realism` chooses another.

The laps are written to `--out` (by default `runs/replays/<agent>`), as
`laps/<steps>.json` with a copy of the agent in `agents/`, the same layout as a
training's folder, so the viewer shows it the same way. Any agent can be
recorded this way, also those of trainings started before recordings existed
(their `best.npz` and `last.npz`).
"""

import argparse
from pathlib import Path

from examples.iris_course import COURSE_FILE
from jineta.campo.course import load_course
from jineta.doma.policy import Agent
from jineta.picadero.finish import StopAfterFinish
from jineta.picadero.realism import IDEAL, LAB, RANDOMISED_LAB
from jineta.picadero.recording import read_recording
from jineta.picadero.training import RECORDED_LAPS, realism_from_label, record_agent

POLICY_FILE = Path(__file__).resolve().parent.parent / "policies" / "first_course.npz"
REALISMS = {"ideal": IDEAL, "lab": LAB, "randomised": RANDOMISED_LAB}


def default_out(agent_path: Path) -> Path:
    """runs/replays/<name>: the agent's file name, after its run folder's name
    when it is a training's best, last or kept agent."""
    parent = agent_path.parent
    if parent.name == "agents":  # an agent kept with a recording
        name = f"{parent.parent.name}_{agent_path.stem}"
    elif agent_path.stem in ("best", "last"):
        name = f"{parent.name}_{agent_path.stem}"
    else:
        name = agent_path.stem
    return Path("runs") / "replays" / name


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("agent", nargs="?", default=str(POLICY_FILE))
    parser.add_argument("--course", default=str(COURSE_FILE))
    parser.add_argument("--realism", choices=sorted(REALISMS), help="fly in this")
    parser.add_argument("--laps", type=int, default=RECORDED_LAPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="folder (default runs/replays/<agent>)")
    args = parser.parse_args(argv)

    agent_path = Path(args.agent)
    agent = Agent.load(agent_path)
    course = load_course(args.course)
    if args.realism is None:
        realism = realism_from_label(str(agent.info.get("realism", "")))
    else:
        realism = REALISMS[args.realism]
    effects = {
        "nearest": int(agent.info.get("nearest", 0)),
        "stop": StopAfterFinish() if agent.info.get("stop") else None,
        "realism": realism,
    }
    out = Path(args.out) if args.out else default_out(agent_path)
    path = record_agent(agent, course, out, args.laps, args.seed, **effects)
    recording = read_recording(path)
    print(f"Recorded {len(recording.laps)} laps of {agent_path} in {path}")
    for lap in recording.laps:
        time = f" {lap.lap_time:.2f} s" if lap.status == "finished" else ""
        print(f"  {lap.label}: {lap.status}{time}, {lap.gates} gates")
    return path


if __name__ == "__main__":
    main()
