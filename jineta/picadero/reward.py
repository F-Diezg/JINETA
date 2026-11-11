"""The reward of a lap and how an episode ends.

The reward is what a policy learns to make as large as possible. Three terms,
computed for every step of the physics (the environment adds up the steps an
action lasts):

  - progress: how much closer the drone got to the centre of its target gate in
    this step [m] (negative when it moves away). The target is the gate that was
    next at the START of the step, also in the step that crosses it, so crossing
    a gate gives no jump: the next step measures against the next gate. Added up
    over a lap, it is the distance flown towards each gate in turn, so a lap that
    gets through pays about the length of the course whatever its time; a faster
    lap pays it sooner, and the policy discounts later rewards (phase 11), so the
    progress term rewards speed through that discount.
  - gate: a bonus for each gate crossed (in order and forward, as the referee
    counts them).
  - crash: a penalty when the lap ends in a collision.

    reward = progress_weight * progress + gate_weight * gates - crash_weight * crash

The weights are a starting point, to be tuned in training (phase 11).

The end of an episode, in the words of Gymnasium: *terminated* when the lap ends
for good, by a crash or by the finish (nothing comes after it, so the policy must
not expect more reward); *truncated* when it is cut by the time limit (the drone
could have flown on: the learning algorithm treats what would have come next as
unknown, not as zero).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from jineta.campo.gates import Gate
from jineta.escuadron.referee import CRASHED, FINISHED, TIMEOUT


class RewardTerms(NamedTuple):
    """The terms of the reward of n drones in one step, each (n,)."""

    progress: np.ndarray  # metres closer to the target gate's centre
    gate: np.ndarray  # gates crossed (0 or 1)
    crash: np.ndarray  # 1 where the lap ended in a crash


@dataclass(frozen=True)
class RewardWeights:
    """How much each term counts: per metre of progress, per gate, per crash."""

    progress: float = 1.0
    gate: float = 1.0
    crash: float = 5.0

    def __post_init__(self) -> None:
        for name in ("progress", "gate", "crash"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"the {name} weight must be finite and >= 0")

    def total(self, terms: RewardTerms) -> np.ndarray:
        """The reward (n,) for these terms."""
        return (
            self.progress * terms.progress
            + self.gate * terms.gate
            - self.crash * terms.crash
        )


@dataclass(frozen=True)
class Reward:
    """The reward of n drones flying through `gates` (in order)."""

    gates: Sequence[Gate]
    weights: RewardWeights = RewardWeights()

    def __post_init__(self) -> None:
        gates = tuple(self.gates)
        if len(gates) == 0:
            raise ValueError("a course needs at least one gate")
        object.__setattr__(self, "gates", gates)
        centers = np.array([gate.center for gate in gates])
        centers.flags.writeable = False
        object.__setattr__(self, "_centers", centers)

    def terms(
        self,
        previous_positions: np.ndarray,
        positions: np.ndarray,
        target: npt.ArrayLike,
        gates_crossed: npt.ArrayLike,
        crashed: npt.ArrayLike,
        active: npt.ArrayLike,
    ) -> RewardTerms:
        """The terms of one step for n drones: their positions before and after
        it (n, 3) [m], the gate each was flying to at its start (n,), how many
        gates each crossed in it (n,), whether its lap ended in a crash in it (n,),
        and which drones were flying at its start (n,; the others get zeros)."""
        active = np.asarray(active, dtype=bool)
        target = np.where(active, target, 0)  # a drone not flying has no target
        center = self._centers[target]
        before = _distance(previous_positions, center)
        after = _distance(positions, center)
        progress = np.where(active, before - after, 0.0)
        gate = np.where(active, np.asarray(gates_crossed, dtype=float), 0.0)
        crash = np.where(active & np.asarray(crashed, dtype=bool), 1.0, 0.0)
        return RewardTerms(progress, gate, crash)

    def __call__(
        self,
        previous_positions: np.ndarray,
        positions: np.ndarray,
        target: npt.ArrayLike,
        gates_crossed: npt.ArrayLike,
        crashed: npt.ArrayLike,
        active: npt.ArrayLike,
    ) -> np.ndarray:
        """The reward (n,) of one step (arguments as `terms`)."""
        terms = self.terms(
            previous_positions, positions, target, gates_crossed, crashed, active
        )
        return self.weights.total(terms)


def _distance(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Distances (n,) between points (n, 3) and centres (n, 3), the squares added
    in order (a drone's number does not depend on its batch)."""
    d = points - centers
    return np.sqrt(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2])


def episode_end(status: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """(terminated, truncated) (n,) for the referee's status codes (n,): a crash or
    the finish terminates an episode, the time limit truncates it."""
    status = np.asarray(status)
    return (status == CRASHED) | (status == FINISHED), status == TIMEOUT
