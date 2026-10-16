"""Dynamic stability: linearise the motion around a trim and find its natural modes.

Close to a trimmed state x0 a small perturbation dx evolves, to first order, as

    d(dx)/dt = A dx,     A = df/dx at x0   (the Jacobian of the dynamics)

so it is a sum of terms e^(lambda t) v, one per eigenvalue lambda of A (and its
eigenvector v, the shape of the motion). A real lambda < 0 is a motion that dies out
without oscillating (lambda > 0 grows: unstable); a complex pair lambda = -zeta wn
+/- i wn sqrt(1 - zeta^2) is an oscillation of natural frequency wn and damping ratio
zeta, which decays when zeta > 0.

An aircraft has five classic modes: two oscillations in the plane of symmetry
(short period: fast pitching at nearly constant speed; phugoid: slow exchange of
height and speed at nearly constant angle of attack) and three out of it (roll
subsidence: the roll rate dies out; Dutch roll: yawing and rolling oscillation;
spiral: slow divergence or convergence of bank and heading). The other eigenvalues
of the 13-element state are zero: position and heading do not change the forces
(at constant density), and the quaternion norm is not a physical direction.

The Jacobian is computed numerically with central differences.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# State components that move in the lateral modes, for level flight heading North:
# east velocity, the roll and yaw parts of the quaternion, roll and yaw rates.
LATERAL = (4, 7, 9, 10, 12)
# ... and in the longitudinal ones: north and down velocity, pitch part, pitch rate.
LONGITUDINAL = (3, 5, 8, 11)


def jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: npt.ArrayLike,
    relative_step: float = 1e-6,
) -> np.ndarray:
    """Jacobian J[i, j] = d func_i / d x_j by central differences.

    Each x_j is moved by h = relative_step * max(1, |x_j|) both ways:
    (f(x + h e_j) - f(x - h e_j)) / 2h, with an error of order h^2.
    """
    x = np.asarray(x, dtype=float)
    columns = []
    for j in range(x.size):
        h = relative_step * max(1.0, abs(x[j]))
        step = np.zeros_like(x)
        step[j] = h
        columns.append((func(x + step) - func(x - step)) / (2.0 * h))
    return np.column_stack(columns)


@dataclass(frozen=True)
class Mode:
    """One natural mode: an eigenvalue and how it looks."""

    eigenvalue: complex  # [1/s]; for a complex pair, the one with Im > 0
    lateral: bool  # True if it moves mostly the lateral components

    @property
    def natural_frequency(self) -> float:
        """wn = |lambda| [rad/s]."""
        return abs(self.eigenvalue)

    @property
    def damping_ratio(self) -> float:
        """zeta = -Re(lambda) / |lambda|: 1 for a decaying real mode, -1 for a growing
        one, in between for oscillations."""
        return -self.eigenvalue.real / abs(self.eigenvalue)

    @property
    def oscillates(self) -> bool:
        return self.eigenvalue.imag > 0.0

    @property
    def period(self) -> float:
        """Time between two peaks, 2 pi / Im(lambda) [s] (inf if it does not
        oscillate)."""
        return 2.0 * np.pi / self.eigenvalue.imag if self.oscillates else np.inf


def dynamic_modes(matrix: np.ndarray, zero_tolerance: float = 1e-6) -> list[Mode]:
    """The modes of d(dx)/dt = A dx, one per real eigenvalue or complex pair.

    Eigenvalues with |lambda| below `zero_tolerance` (neutral directions) are left
    out. Sorted from the fastest (largest |lambda|) to the slowest. For the
    13-element rigid-body state (level flight heading North) a mode is lateral when
    its eigenvector is larger in the LATERAL components than in the LONGITUDINAL
    ones, both in the units of the state; for any other system it is never lateral.
    """
    eigenvalues, vectors = np.linalg.eig(matrix)
    rigid_body = matrix.shape == (13, 13)
    modes = []
    for value, vector in zip(eigenvalues, vectors.T, strict=True):
        if abs(value) < zero_tolerance or value.imag < 0.0:
            continue
        size = np.abs(vector)
        lateral = rigid_body and np.linalg.norm(size[list(LATERAL)]) > np.linalg.norm(
            size[list(LONGITUDINAL)]
        )
        modes.append(Mode(complex(value), bool(lateral)))
    return sorted(modes, key=lambda mode: -mode.natural_frequency)


@dataclass(frozen=True)
class AircraftModes:
    """The five classic modes of an aircraft, picked out of `dynamic_modes`."""

    short_period: Mode
    phugoid: Mode
    roll: Mode
    dutch_roll: Mode
    spiral: Mode


def classify_modes(modes: list[Mode]) -> AircraftModes:
    """Name the modes: the faster longitudinal oscillation is the short period and
    the slower the phugoid; the lateral oscillation is the Dutch roll; of the two
    real lateral modes the faster is the roll subsidence, the slower the spiral.

    Raises:
        ValueError: the modes do not have that structure.
    """
    longitudinal = [m for m in modes if not m.lateral and m.oscillates]
    lateral_oscillations = [m for m in modes if m.lateral and m.oscillates]
    lateral_real = [m for m in modes if m.lateral and not m.oscillates]
    if len(longitudinal) != 2 or len(lateral_oscillations) != 1:
        raise ValueError("expected two longitudinal and one lateral oscillation")
    if len(lateral_real) != 2:
        raise ValueError("expected two real lateral modes")
    # modes are sorted fastest first
    return AircraftModes(
        short_period=longitudinal[0],
        phugoid=longitudinal[1],
        roll=lateral_real[0],
        dutch_roll=lateral_oscillations[0],
        spiral=lateral_real[1],
    )
