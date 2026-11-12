"""Training a policy on a course, and measuring it.

`train` runs PPO (`jineta.doma`) on a `CourseEnv` and, after every update,
writes a row of the training log (CSV: how the episodes ended, the lap times, the
diagnostics of the update); every `evaluate_every` updates it flies the
deterministic policy (the mean action, as a real drone would fly it) and keeps
the best agent so far (`best.npz`: the fastest finished lap; while none
finishes, the most gates), and the last one (`last.npz`).

`evaluate` measures an agent on a course the way the baseline is measured: the
deterministic policy's lap (status, lap time, closest approach to anything), and
its robustness: the success rate of `n` laps with the actions drawn from the
policy (its exploration noise standing in for the imperfections of a real
flight, until phase 12 randomises the vehicle).
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jineta.campo.course import Course
from jineta.doma.policy import Agent
from jineta.doma.ppo import PPO, PPOSettings
from jineta.escuadron.referee import CRASHED, FINISHED, STATUS_NAMES, TIMEOUT
from jineta.picadero.env import CourseEnv
from jineta.picadero.reward import RewardWeights

# The reward the policies of phase 11 are trained with (found by measurement;
# the decision log of phase 11 has the experiments).
TRAINING_WEIGHTS = RewardWeights(progress=1.0, gate=1.0, crash=5.0, time=0.25)
# The safety margin of training [m]: the baseline's laps are robust only if they
# keep 5 cm from everything, so the policy is trained to keep it too.
TRAINING_MARGIN = 0.05

LOG_COLUMNS = (
    "iteration",
    "steps",
    "seconds",
    "episodes",
    "finished",
    "crashed",
    "timeout",
    "mean_return",
    "mean_length",
    "mean_lap_time",
    "best_lap_time",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "std",
    "learning_rate",
    "eval_status",
    "eval_lap_time",
    "eval_gates",
)


@dataclass
class Evaluation:
    """How an agent flies a course.

    status, lap_time, gates: the deterministic policy's lap (lap_time nan unless
        it finished).
    clearance: its closest approach to a wall, the ceiling, an obstacle or a
        frame [m] (negative: it touched).
    success_rate, mean_lap_time: over the laps flown with the policy's own
        noise (mean_lap_time over those that finished; nan if none did).
    lap_times: those laps' times (nan where they did not finish).
    """

    status: str
    lap_time: float
    gates: int
    clearance: float
    success_rate: float
    mean_lap_time: float
    lap_times: np.ndarray = field(repr=False)

    @property
    def finished(self) -> bool:
        return self.status == "finished"


def fly(
    agent: Agent,
    env: CourseEnv,
    rng: np.random.Generator | None = None,
    record: bool = False,
) -> dict[str, Any]:
    """One lap per drone of `env` with the agent (the mean action without `rng`,
    drawn from the policy with it). Returns statuses (n,), lap times (n,; nan
    unless finished), gates passed (n,), clearances (n,: the closest approach to
    a wall, the ceiling, an obstacle or a frame before the lap's last step) and,
    if `record`, the states of drone 0 after every environment step until its
    lap ended (steps + 1, 13)."""
    observations, _ = env.reset()
    n = env.n
    status = np.full(n, -1)
    lap_time = np.full(n, np.nan)
    gates = np.zeros(n, dtype=int)
    clearance = np.full(n, np.inf)
    referee = env.flight.referee
    states = [env.flight.states[0].copy()] if record else []
    limit = int(np.ceil(env.course.time_limit / env.step_time)) + 1
    for _ in range(limit):
        flying = status < 0
        actions = agent.act(observations, rng)
        observations, _, terminated, truncated, info = env.step(actions)
        done = (terminated | truncated) & flying
        going_on = flying & ~done
        margins = referee.monitor.margins(env.flight.states)[:, :-1].min(axis=1)
        clearance = np.where(going_on, np.minimum(clearance, margins), clearance)
        gates = np.where(going_on, referee.passed, gates)
        status[done] = info["status"][done]
        lap_time[done] = info["lap_time"][done]
        if record and flying[0]:
            states.append(info["final_state"][0].copy())
        if np.all(status >= 0):
            break
    gates = np.where(status == FINISHED, len(env.course.gates), gates)
    result = {
        "status": status,
        "lap_time": lap_time,
        "gates": gates,
        "clearance": clearance,
    }
    if record:
        result["states"] = np.array(states)
    return result


def evaluate(
    agent: Agent,
    course: Course,
    n: int = 64,
    seed: int = 0,
    **env_kwargs: Any,
) -> Evaluation:
    """The deterministic lap and the success rate of n laps flown with the
    policy's noise (seeded)."""
    single = fly(agent, CourseEnv(course, n=1, **env_kwargs))
    noisy = fly(
        agent, CourseEnv(course, n=n, **env_kwargs), np.random.default_rng(seed)
    )
    finished = noisy["status"] == FINISHED
    times = np.where(finished, noisy["lap_time"], np.nan)
    return Evaluation(
        status=STATUS_NAMES[single["status"][0]],
        lap_time=float(single["lap_time"][0]),
        gates=int(single["gates"][0]),
        clearance=float(single["clearance"][0]),
        success_rate=float(np.mean(finished)),
        mean_lap_time=float(np.nanmean(times)) if np.any(finished) else float("nan"),
        lap_times=times,
    )


def _rank(result: Evaluation, margin: float = 0.0) -> tuple[float, float]:
    """Higher is better: a finished lap that keeps `margin` from everything
    beats one that does not, which beats any unfinished lap; then a faster lap
    a slower one; between unfinished laps, more gates."""
    if result.finished:
        kept = result.clearance >= margin
        return (2.0 if kept else 1.0, -result.lap_time)
    return (0.0, float(result.gates))


def _better(new: Evaluation, old: Evaluation | None, margin: float = 0.0) -> bool:
    return old is None or _rank(new, margin) > _rank(old, margin)


Judge = Callable[[Agent], tuple[tuple[float, float], dict[str, Any], Any]]


def run_training(
    env: Any,
    total_steps: int,
    out: str | Path,
    judge: Judge,
    seed: int = 0,
    settings: PPOSettings | None = None,
    evaluate_every: int = 10,
    agent: Agent | None = None,
    info: dict[str, Any] | None = None,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Agent, Any]:
    """Train an agent on any environment with CourseEnv's shape. Every
    `evaluate_every` updates `judge(agent)` gives (a rank, higher is better; the
    log's eval_ fields; a result): the agent with the highest rank so far is
    saved as `out/best.npz`, the latest as `out/last.npz`. Writes `out/log.csv`
    and calls `report(row)` after every update. Returns the best agent and its
    result (the last agent and None if it was never judged)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ppo = PPO(env, settings, seed, agent, episode_info=("status", "lap_time"))
    info = {**(info or {}), "seed": seed}
    best_rank: tuple[float, float] | None = None
    best_result: Any = None
    best_agent = ppo.agent
    lap_times: list[float] = []

    with open(out / "log.csv", "w", newline="", encoding="ascii") as file:
        writer = csv.DictWriter(file, LOG_COLUMNS)
        writer.writeheader()

        def callback(stats: dict[str, Any]) -> None:
            nonlocal best_rank, best_result, best_agent
            episodes = stats["episodes"]
            statuses = np.array([e["status"] for e in episodes.infos], dtype=int)
            laps = [e["lap_time"] for e in episodes.infos if e["status"] == FINISHED]
            lap_times.extend(laps)
            count = len(statuses)
            row: dict[str, Any] = {
                "episodes": count,
                "finished": int(np.sum(statuses == FINISHED)),
                "crashed": int(np.sum(statuses == CRASHED)),
                "timeout": int(np.sum(statuses == TIMEOUT)),
                "mean_return": np.mean(episodes.returns) if count else np.nan,
                "mean_length": np.mean(episodes.lengths) if count else np.nan,
                "mean_lap_time": np.mean(laps) if laps else np.nan,
                "best_lap_time": min(lap_times) if lap_times else np.nan,
            }
            for key in LOG_COLUMNS:
                if key in stats and key not in row:
                    row[key] = stats[key]
            if stats["iteration"] % evaluate_every == 0:
                ppo.agent.info = {**info, "steps": ppo.steps}
                rank, fields, result = judge(ppo.agent)
                row.update(fields)
                ppo.agent.save(out / "last.npz")
                if best_rank is None or rank > best_rank:
                    best_rank, best_result = rank, result
                    ppo.agent.save(out / "best.npz")
                    best_agent = Agent.load(out / "best.npz")
            writer.writerow(
                {
                    key: (f"{value:.6g}" if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )
            file.flush()
            if report is not None:
                report(row)

        ppo.learn(total_steps, callback)
    ppo.agent.info = {**info, "steps": ppo.steps}
    ppo.agent.save(out / "last.npz")
    return best_agent, best_result


def train(
    course: Course,
    total_steps: int,
    out: str | Path,
    n: int = 256,
    seed: int = 0,
    settings: PPOSettings | None = None,
    weights: RewardWeights = TRAINING_WEIGHTS,
    evaluate_every: int = 10,
    agent: Agent | None = None,
    margin: float = TRAINING_MARGIN,
    report: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Agent, Evaluation | None]:
    """Train an agent on one course for `total_steps` environment steps with n
    drones, every sphere of their hitbox `margin` larger; the best agent is the
    one whose mean action flies the fastest lap, preferring laps that keep the
    margin (judged on the course as it is; see `run_training`). Returns the best
    agent and its evaluation.

    The policies of phase 11 are trained in two stages (`examples.train_course`):
    without the margin first (with it from the start, crashes are so frequent
    that the policy learns to wait until the time limit), then with it, starting
    from the first stage's agent."""

    def judge(candidate: Agent):
        result = evaluate(candidate, course, 16, seed)
        fields = {
            "eval_status": result.status,
            "eval_lap_time": result.lap_time,
            "eval_gates": result.gates,
        }
        return _rank(result, margin), fields, result

    env = CourseEnv(course, n=n, weights=weights, margin=margin)
    info = {"course": course.name, "drones": n}
    return run_training(
        env, total_steps, out, judge, seed, settings, evaluate_every, agent, info,
        report,
    )  # fmt: skip
