"""Several environments at once, one per process: training on every core.

Python runs one line at a time in a process, so a training uses one core however
many the computer has (loose end 21). The environment is most of the time of a
training (two thirds in phase 11), and its drones do not interact, so it can be
split: `ParallelEnv` starts one worker process per part, each with its own
environment (a `CourseEnv` with its share of the drones, or a `CourseSetEnv`
with its share of the courses), steps all of them at the same time and joins
their results as if they were one environment with all the drones. The learning
(the network, PPO) stays in the main process: it is given the joined
observations and returns the joined actions, as before.

    worker 0: env with drones 0 .. m-1     \
    worker 1: env with drones m .. 2m-1     >  ParallelEnv (n = W m drones)  <-> PPO
    ...                                    /

The joined environment has the shape of `CourseEnv` (observations (n, size),
rewards, terminated and truncated (n,), every info array (n, ...)), with the
drones in the order of the workers. Because a drone flies the same alone or in
any batch (the batch engine never mixes rows), W workers of m drones give the
same numbers bit for bit as one environment of W m drones, and a training gives
the same result with any number of workers (on the same computer).

Processes are started with "spawn" (a new Python that imports what it needs),
the only way Windows has: the environment is given as a `make_env(index)` that
can be pickled (a top-level function or class, or a `functools.partial` of one;
not a lambda), and a script that starts a `ParallelEnv` must keep its work under
`if __name__ == "__main__":`. Close it at the end (`close()`, or use it in a
`with` block): the workers are separate programs.

`SerialEnv` is the same joining done in one process (no workers), with the same
seeds: it is what a `ParallelEnv` is checked against, and what a training uses
with one worker.

Seeds: `reset(seed)` gives worker i the seed `worker_seed(seed, i)` (drawn from a
`numpy.random.SeedSequence` of (seed, i)), so each part has its own random draws
(random courses, and in phase 12 the randomised drones) and the whole repeats
with the same seed. `make_parallel` with one worker gives the bare part, reset
with `seed` itself, so a training with one worker is the training of before.
"""

import multiprocessing as mp
import traceback
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

# make_env(index) -> an environment with CourseEnv's shape (picklable).
EnvFactory = Callable[[int], Any]

StepResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]


def worker_seed(seed: int | None, index: int) -> int | None:
    """The seed of part `index` when the whole is reset with `seed` (None stays
    None: fresh randomness)."""
    if seed is None:
        return None
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _join(results: Sequence[StepResult]) -> StepResult:
    """The results of the parts' steps as one environment's (drones in order)."""
    info = {key: np.concatenate([r[4][key] for r in results]) for key in results[0][4]}
    return (
        np.concatenate([r[0] for r in results]),
        np.concatenate([r[1] for r in results]),
        np.concatenate([r[2] for r in results]),
        np.concatenate([r[3] for r in results]),
        info,
    )


def _split(action: npt.ArrayLike, sizes: Sequence[int]) -> list[np.ndarray]:
    """The joined actions (n, k) cut into the parts' actions."""
    action = np.asarray(action, dtype=float)
    if action.ndim != 2 or len(action) != sum(sizes):
        raise ValueError(f"action must have shape ({sum(sizes)}, k)")
    return np.split(action, np.cumsum(sizes)[:-1])


class SerialEnv:
    """The parts `make_env(0) .. make_env(workers - 1)` stepped one after the
    other in this process, joined as one environment (see the module)."""

    def __init__(self, make_env: EnvFactory, workers: int = 1) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.envs = [make_env(i) for i in range(workers)]
        self.sizes = [env.n for env in self.envs]
        self.n = sum(self.sizes)

    @property
    def observation_space(self) -> Any:
        return self.envs[0].observation_space

    @property
    def action_space(self) -> Any:
        return self.envs[0].action_space

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        parts = [
            env.reset(seed=worker_seed(seed, i)) for i, env in enumerate(self.envs)
        ]
        return np.concatenate([p[0] for p in parts]), {}

    def step(self, action: npt.ArrayLike) -> StepResult:
        actions = _split(action, self.sizes)
        return _join([env.step(a) for env, a in zip(self.envs, actions, strict=True)])

    def close(self) -> None:
        """Nothing to stop (the same interface as ParallelEnv)."""

    def __enter__(self) -> "SerialEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _worker(connection: Any, make_env: EnvFactory, index: int) -> None:
    """The loop of a worker process: build its part, then answer commands until
    told to close. Every answer is ("ok", result) or ("error", traceback)."""
    try:
        env = make_env(index)
        connection.send(("ok", (env.n, env.observation_space, env.action_space)))
    except Exception:
        connection.send(("error", traceback.format_exc()))
        connection.close()
        return
    while True:
        try:
            command, argument = connection.recv()
        except EOFError:  # the main process went away
            break
        if command == "close":
            connection.send(("ok", None))
            break
        try:
            if command == "reset":
                result = env.reset(seed=argument)
            elif command == "step":
                result = env.step(argument)
            else:
                raise ValueError(f"unknown command {command!r}")
            connection.send(("ok", result))
        except Exception:
            connection.send(("error", traceback.format_exc()))
    connection.close()


class WorkerError(RuntimeError):
    """An exception raised inside a worker process (its traceback in the
    message)."""


class ParallelEnv:
    """The parts `make_env(0) .. make_env(workers - 1)` each in its own process,
    stepped at the same time and joined as one environment (see the module)."""

    def __init__(self, make_env: EnvFactory, workers: int) -> None:
        if workers < 1:
            raise ValueError("workers must be >= 1")
        context = mp.get_context("spawn")
        self.connections = []
        self.processes = []
        self.closed = False
        for i in range(workers):
            mine, theirs = context.Pipe()
            process = context.Process(
                target=_worker, args=(theirs, make_env, i), daemon=True
            )
            process.start()
            theirs.close()
            self.connections.append(mine)
            self.processes.append(process)
        try:
            ready = self._gather()
        except WorkerError:
            self.close()
            raise
        self.sizes = [size for size, _, _ in ready]
        self.n = sum(self.sizes)
        self._observation_space = ready[0][1]
        self._action_space = ready[0][2]

    @property
    def observation_space(self) -> Any:
        return self._observation_space

    @property
    def action_space(self) -> Any:
        return self._action_space

    def _gather(self) -> list[Any]:
        """Every worker's answer, in order; the first error is raised here
        (after reading every answer, so the pipes stay in step)."""
        answers = [connection.recv() for connection in self.connections]
        for i, (status, result) in enumerate(answers):
            if status == "error":
                raise WorkerError(f"worker {i} failed:\n{result}")
        return [result for _, result in answers]

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        for i, connection in enumerate(self.connections):
            connection.send(("reset", worker_seed(seed, i)))
        parts = self._gather()
        return np.concatenate([p[0] for p in parts]), {}

    def step(self, action: npt.ArrayLike) -> StepResult:
        actions = _split(action, self.sizes)
        for connection, part in zip(self.connections, actions, strict=True):
            connection.send(("step", part))
        return _join(self._gather())

    def close(self) -> None:
        """Stop the workers (waits a little for each, then terminates it)."""
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for connection, process in zip(self.connections, self.processes, strict=True):
            try:
                if connection.poll(5.0):
                    connection.recv()
            except (EOFError, OSError):
                pass
            connection.close()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join()

    def __enter__(self) -> "ParallelEnv":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_parallel(make_env: EnvFactory, workers: int) -> Any:
    """One worker: the bare part `make_env(0)` in this process (no process to
    start; reset with the seed as given, as before this module); more: a
    `ParallelEnv`."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        return make_env(0)
    return ParallelEnv(make_env, workers)


class CourseEnvPart:
    """`make_env` for a `CourseEnv` split into parts: part i is
    `CourseEnv(course, n=drones, **env_kwargs)` (picklable)."""

    def __init__(self, course: Any, drones: int, **env_kwargs: Any) -> None:
        self.course, self.drones, self.env_kwargs = course, drones, env_kwargs

    def __call__(self, index: int) -> Any:
        from jineta.picadero.env import CourseEnv

        return CourseEnv(self.course, n=self.drones, **self.env_kwargs)


class CourseSetEnvPart:
    """`make_env` for a `CourseSetEnv` split into parts: part i flies `k` courses
    of `n` drones each, drawn by `make_course` (picklable)."""

    def __init__(
        self,
        make_course: Callable[[np.random.Generator], Any],
        k: int,
        n: int,
        renew_every: int,
        **env_kwargs: Any,
    ) -> None:
        self.make_course, self.k, self.n = make_course, k, n
        self.renew_every, self.env_kwargs = renew_every, env_kwargs

    def __call__(self, index: int) -> Any:
        from jineta.picadero.courses import CourseSetEnv

        return CourseSetEnv(
            self.make_course, self.k, self.n, self.renew_every, **self.env_kwargs
        )
