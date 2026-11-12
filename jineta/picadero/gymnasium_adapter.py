"""The training environment as Gymnasium sees it: `CourseGymEnv` (one drone, a
`gymnasium.Env`) and `CourseVectorEnv` (n drones, a `gymnasium.vector.VectorEnv`).

Gymnasium is the standard interface between environments and learning
libraries: any library that speaks it can train on these. `CourseEnv` already has
its shape (phase 10); this module only translates names and types:

  - spaces: `BoxSpace` becomes `gymnasium.spaces.Box` (float32, same limits);
  - one drone: `CourseGymEnv.step` takes one action (4,) and returns one
    observation (size,), a float reward and two bools. When the episode ends, the
    observation returned is its last one, as Gymnasium expects from a single
    environment; call `reset` before the next `step` (the drone inside has already
    been put back on the floor, so a `step` without it simply flies the next
    episode);
  - n drones: `CourseVectorEnv` restarts a drone whose episode ended in the same
    step (`AutoresetMode.SAME_STEP`, declared in `metadata`), as `CourseEnv` does.
    Its info follows Gymnasium's vector format: where an episode ended,
    `info["final_obs"]` (an object array (n,), each entry the last observation of
    that drone, None elsewhere) and `info["final_info"]` (status, return and steps
    of the episode that ended, and its lap time if it finished), each with its
    mask `info["_final_obs"]`,
    `info["_final_info"]`: the same final observations and final infos that
    Gymnasium's own `SyncVectorEnv` gives for n copies of `CourseGymEnv` in the
    same mode (a test checks it). Unlike it, the per-step info of the drones still
    flying is not repeated (the whole state of an episode is in `course_env`).

Gymnasium is an optional dependency: this module needs it, the rest of the
project does not (and nothing in the project imports this module).
"""

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from jineta.campo.course import Course
from jineta.escuadron.referee import STATUS_NAMES
from jineta.picadero.env import CourseEnv
from jineta.picadero.spaces import BoxSpace

# The per-episode entries of CourseEnv's info, given for the episodes that ended.
EPISODE_KEYS = ("status", "lap_time", "episode_return", "episode_steps")


def to_box(space: BoxSpace) -> gym.spaces.Box:
    """Gymnasium's `Box` with the limits of a `BoxSpace` (float32)."""
    return gym.spaces.Box(space.low, space.high, dtype=np.float32)


class CourseGymEnv(gym.Env):
    """One drone learning one course, as a `gymnasium.Env`.

    course: the course; every other keyword goes to `CourseEnv` (profile, limits,
    weights, dt, action_repeat, lookahead), with n = 1.
    """

    metadata: dict[str, Any] = {"render_modes": []}  # noqa: RUF012 (Gymnasium's)

    def __init__(self, course: Course, **kwargs: Any) -> None:
        super().__init__()
        self.course_env = CourseEnv(course, n=1, **kwargs)
        self.observation_space = to_box(self.course_env.observation_space)
        self.action_space = to_box(self.course_env.action_space)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Put the drone on the floor at the start; returns its observation (size,)
        and an empty info."""
        super().reset(seed=seed)
        observation, _ = self.course_env.reset(seed=seed, options=options)
        return observation[0], {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Fly one action (4,); returns the observation (size,), the reward, whether
        the episode terminated or was truncated, and its info (the status name,
        return and steps of the episode so far, and the lap time once it has
        finished)."""
        action = np.asarray(action, dtype=float).reshape(1, -1)
        observation, reward, terminated, truncated, info = self.course_env.step(action)
        if terminated[0] or truncated[0]:
            observation = info["final_observation"]
        episode = {key: info[key][0].item() for key in EPISODE_KEYS}
        episode["status"] = STATUS_NAMES[episode["status"]]
        if np.isnan(episode["lap_time"]):
            del episode["lap_time"]  # only a finished lap has one (nan != nan)
        return (
            observation[0],
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            episode,
        )


class CourseVectorEnv(VectorEnv):
    """n drones learning one course, as a `gymnasium.vector.VectorEnv` that
    restarts each drone in the step its episode ends (same-step autoreset).

    course, n: the course and the number of drones; every other keyword goes to
    `CourseEnv`.
    """

    metadata: dict[str, Any] = {  # noqa: RUF012 (Gymnasium's)
        "autoreset_mode": AutoresetMode.SAME_STEP,
        "render_modes": [],
    }

    def __init__(self, course: Course, n: int, **kwargs: Any) -> None:
        super().__init__()
        self.course_env = CourseEnv(course, n=n, **kwargs)
        self.num_envs = n
        self.single_observation_space = to_box(self.course_env.observation_space)
        self.single_action_space = to_box(self.course_env.action_space)
        self.observation_space = batch_space(self.single_observation_space, n)
        self.action_space = batch_space(self.single_action_space, n)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Put every drone on the floor at the start; returns the observations
        (n, size) and an empty info."""
        super().reset(seed=seed)
        observations, _ = self.course_env.reset(seed=seed, options=options)
        return observations, {}

    def step(
        self, actions: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Fly one action per drone (n, 4); returns observations (n, size) (the
        first of the next episode where one ended), rewards (n,), terminated (n,),
        truncated (n,) and info (see the module)."""
        observations, rewards, terminated, truncated, info = self.course_env.step(
            actions
        )
        done = terminated | truncated
        infos: dict[str, Any] = {}
        if np.any(done):
            final_obs = np.full(self.num_envs, None, dtype=object)
            final_info: dict[str, Any] = {}
            for i in np.flatnonzero(done):
                final_obs[i] = info["final_observation"][i]
            for key in EPISODE_KEYS:
                values = info[key]
                if key == "status":
                    values = np.array([STATUS_NAMES[s] for s in values], dtype=object)
                given = done & ~np.isnan(values) if key == "lap_time" else done
                if not np.any(given):
                    continue  # as Gymnasium: a key appears only where given
                kept = np.zeros_like(values)
                if key == "status":
                    kept[:] = None
                kept[given] = values[given]
                final_info[key], final_info[f"_{key}"] = kept, given.copy()
            infos["final_obs"], infos["_final_obs"] = final_obs, done.copy()
            infos["final_info"], infos["_final_info"] = final_info, done.copy()
        return observations, rewards, terminated, truncated, infos
