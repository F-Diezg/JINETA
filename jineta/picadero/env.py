"""The training environment: n drones flying laps of one course, commanded by a
policy through collective thrust and body rates.

The interface has the shape of Gymnasium's (without depending on it; the adapter
to the library chosen in phase 11 wraps it):

    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(action)

with one row per drone: observation (n, size) float32, action (n, 4) in [-1, 1],
reward (n,), terminated and truncated (n,). `observation_space` and
`action_space` describe one row.

Each drone is an episode of its own (a vectorised environment, n copies of one
course flown at once by the batch engine). An episode is one lap: it starts with
the drone standing on the floor at the course's start, its rate pilot landed; it
ends by a crash or the finish (terminated) or the time limit (truncated). A drone
whose episode ends in a step is restarted at once, in that same step: the
observation returned for it is the first of its next episode, and the last one of
the episode that ended is in `info["final_observation"]` (what Gymnasium calls
same-step autoreset; Stable-Baselines3's vectorised environments do the same).

Timing: the physics and the rate loop run at 1 / dt (100 Hz); the policy acts
every `action_repeat` steps (2: 50 Hz), and its action is held in between. The
reward of an environment step is the sum of the rewards of its physics steps.

Realism (`picadero.realism.Realism`, phase 12; all off by default): the motor
lag and the ground effect change the vehicles; with `sensors` the observation
is built from each drone's onboard estimate (`atalaya.estimator`) instead of
its true state; with a `latency` every command reaches the rate loop that many
physics steps late (a ring of commands on their way, cleared when a drone
restarts). The reward and the referee always judge the true flight.

`info` (arrays, one entry per drone; meaningful where `done = terminated |
truncated`):
    final_observation (n, size)   the last observation of an episode that ended
    final_state (n, size)         the drones' states at the end of the step (where
                                  an episode ended, before its drone restarted)
    status (n,)                   how it ended (codes of escuadron.referee)
    lap_time (n,)                 [s], nan unless it finished
    episode_return (n,)           the sum of its rewards
    episode_steps (n,)            its number of environment steps
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from jineta.atalaya.estimator import BatchEstimator
from jineta.campo.course import Course
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.control import BatchRatePilot
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import CRASHED, batch_referee
from jineta.escuadron.simulation import BatchFlight
from jineta.jinete.contact import resting_height
from jineta.jinete.profiles import MULTIROTOR_PROFILES, MultirotorProfile
from jineta.jinete.proximity import GroundEffect
from jineta.picadero.realism import Realism
from jineta.picadero.reward import Reward, RewardWeights, episode_end
from jineta.picadero.spaces import (
    ACTION_SIZE,
    ActionLimits,
    BoxSpace,
    Observer,
    Surroundings,
    autopilot_envelope,
)


@dataclass
class CourseEnv:
    """n drones of one profile learning one course.

    course: the course (room, gates, obstacles, start, time limit).
    n: number of drones (episodes flown at once).
    profile: the vehicle (parameters, gains, landing gear, hitbox).
    limits: the action envelope; by default the profile's autopilot envelope.
    weights: the reward weights.
    dt: the step of the physics and the rate loop [s].
    action_repeat: physics steps per action (the policy acts every
        action_repeat * dt seconds).
    lookahead: gates ahead in the observation.
    nearest: surfaces around the drone in the observation (0: none; walls,
        floor, ceiling, obstacles and gate frames, see spaces.Surroundings).
    margin: added to the radius of every sphere of the hitbox [m]: coming that
        close to a wall, the ceiling, an obstacle or a frame is a crash (a safety
        margin the policy learns to keep; the floor is still judged by the
        landing gear).
    realism: the effects of a real flight that are simulated (`Realism`; none
        by default).
    """

    course: Course
    n: int = 1
    profile: MultirotorProfile = MULTIROTOR_PROFILES["iris"]
    limits: ActionLimits | None = None
    weights: RewardWeights = field(default_factory=RewardWeights)
    dt: float = 0.01
    action_repeat: int = 2
    lookahead: int = 2
    nearest: int = 0
    margin: float = 0.0
    realism: Realism = field(default_factory=Realism)
    np_random: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        if self.action_repeat < 1:
            raise ValueError("action_repeat must be >= 1")
        if not self.dt > 0.0:
            raise ValueError("dt must be > 0")
        if not self.margin >= 0.0:
            raise ValueError("margin must be >= 0")
        params, gains, gear, hitbox = self.profile
        if self.limits is None:
            self.limits = autopilot_envelope(params, gains)
        surroundings = None
        if self.nearest > 0:
            surroundings = Surroundings(
                self.course.room,
                (*self.course.obstacles, *self.course.gates),
                self.nearest,
            )
        self.observer = Observer(self.course.gates, self.lookahead, surroundings)
        self.reward = Reward(self.course.gates, self.weights, self.dt)
        self.vehicles = BatchMultirotor(
            params,
            self.n,
            contact=BatchGroundContact(gear),
            motor_lag=self.realism.motor_lag,
            ground_effect=GroundEffect() if self.realism.ground_effect else None,
        )
        referee = batch_referee(
            self.course, hitbox.centers, hitbox.radii + self.margin, gear.points, self.n
        )
        self.start = self.vehicles.initial_state(
            self.course.start_state(resting_height(gear, params.mass))
        )
        self.start.flags.writeable = False
        self.flight = BatchFlight(
            self.vehicles, referee, self.dt, np.tile(self.start, (self.n, 1))
        )
        self.pilot = BatchRatePilot(self.vehicles, gains.rate, self.dt, landed=True)
        self.estimator = None
        if self.realism.sensors is not None:
            self.estimator = BatchEstimator(self.realism.sensors, self.n, self.dt)
        # Commands on their way to the rate loop (latency): a ring of slots,
        # one per physics step of delay.
        self.delay = self.realism.latency_steps(self.dt)
        self.pending_thrust = np.zeros((self.delay, self.n))
        self.pending_rates = np.zeros((self.delay, self.n, 3))
        self.physics_steps = 0
        self.previous_action = np.zeros((self.n, ACTION_SIZE))
        self.episode_return = np.zeros(self.n)
        self.episode_steps = np.zeros(self.n, dtype=int)
        self.np_random = np.random.default_rng()

    @property
    def observation_space(self) -> BoxSpace:
        return self.observer.space

    @property
    def action_space(self) -> BoxSpace:
        return self.limits.space

    @property
    def step_time(self) -> float:
        """Time between two actions [s]."""
        return self.action_repeat * self.dt

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode for every drone; returns the observations (n, size)
        and an empty info. `seed` seeds `np_random`, which draws the sensors'
        noise (and the randomisation of phase 12)."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._restart(np.ones(self.n, dtype=bool))
        return self._observe(), {}

    def step(
        self, action: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Fly every drone for one action; returns observations (n, size),
        rewards (n,), terminated (n,), truncated (n,) and info (see the module)."""
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n, ACTION_SIZE):
            raise ValueError(f"action must have shape ({self.n}, {ACTION_SIZE})")
        action = np.clip(action, -1.0, 1.0)
        thrust, rates = self.limits.commands(action)
        referee = self.flight.referee
        reward = np.zeros(self.n)
        for _ in range(self.action_repeat):
            active = ~referee.ended
            if not np.any(active):
                break
            if self.delay == 0:
                self.pilot.thrust_command, self.pilot.rate_command = thrust, rates
            else:  # the command sent `delay` physics steps ago arrives now
                slot = self.physics_steps % self.delay
                self.pilot.thrust_command = self.pending_thrust[slot].copy()
                self.pilot.rate_command = self.pending_rates[slot].copy()
                self.pending_thrust[slot] = thrust
                self.pending_rates[slot] = rates
            self.physics_steps += 1
            target, passed = referee.next_gate.copy(), referee.passed.copy()
            previous = self.flight.states.copy()
            self.flight.step(self.pilot)
            if self.estimator is not None:
                self.estimator.step(previous, self.flight.states, self.np_random)
            crashed = active & (referee.status == CRASHED)
            reward += self.reward(
                previous[:, 0:3],
                self.flight.states[:, 0:3],
                target,
                referee.passed - passed,
                crashed,
                active,
            )
        self.previous_action = action
        self.episode_return += reward
        self.episode_steps += 1

        terminated, truncated = episode_end(referee.status)
        done = terminated | truncated
        info: dict[str, Any] = {
            "final_observation": self._observe(),
            "final_state": self.flight.states.copy(),
            "status": referee.status.copy(),
            "lap_time": referee.lap_time,
            "episode_return": self.episode_return.copy(),
            "episode_steps": self.episode_steps.copy(),
        }
        if np.any(done):
            self._restart(done)
        return self._observe(), reward, terminated, truncated, info

    def close(self) -> None:
        """Nothing to release (Gymnasium's interface has it; `ParallelEnv`
        stops its workers)."""

    def _restart(self, drones: np.ndarray) -> None:
        """These drones (a mask) start a new lap: standing on the floor at the
        start, pilot landed with a clean loop, no previous action."""
        self.flight.reset(drones, self.start)
        self.pilot.reset(drones)
        self.pilot.landed[drones] = True
        self.pending_thrust[:, drones] = 0.0
        self.pending_rates[:, drones] = 0.0
        if self.estimator is not None:
            self.estimator.reset(drones, self.start, self.np_random)
        self.previous_action[drones] = 0.0
        self.episode_return[drones] = 0.0
        self.episode_steps[drones] = 0

    def _observe(self) -> np.ndarray:
        """What the policy sees: built from the true states, or from the
        estimated ones with sensors."""
        states = (
            self.flight.states if self.estimator is None else self.estimator.estimate
        )
        observation = self.observer(
            states, self.flight.referee.next_gate, self.previous_action
        )
        return observation.astype(np.float32)
