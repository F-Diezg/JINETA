"""The Kalman filter: the best estimate of a state from noisy measurements.

An estimate is a Gaussian: a mean `x` (the best guess of the state) and a covariance
`P` (how unsure we are, and how the errors of the components go together). The
filter alternates two steps:

  - predict: move the estimate forward in time with a motion model,
        x <- F x,    P <- F P F^T + Q;
    F is the motion over the interval and Q the uncertainty the model cannot
    explain (the body may accelerate), so P grows.
  - update: blend in a measurement z = H x + noise (noise covariance R),
        y = z - H x            innovation: what the measurement says that the
                               prediction did not
        S = H P H^T + R        how large the innovation is expected to be
        K = P H^T S^-1         gain: how much to trust the measurement
        x <- x + K y,   P <- (I - K H) P (I - K H)^T + K R K^T,
    so P shrinks. The covariance is updated in Joseph form: the same value as
    (I - K H) P for the optimal gain, but always symmetric and positive
    semi-definite, also with rounding errors.

The functions here know nothing about vehicles or sensors: they take matrices and
return new arrays (the inputs are never changed).

Motion model: constant velocity, state [x, y, z, vx, vy, vz]
(NED, as a point mass), with the body's acceleration as white noise. Two classic
ways of putting that noise into Q:
  - `white_noise_acceleration` (continuous): the acceleration is white noise of
    spectral density q [m^2/s^3]. Predicting over T in one go or in several
    pieces gives exactly the same P, so the prediction does not depend on the
    simulation step. It is the one to use when the filter predicts at every
    simulation step between measurements.
  - `piecewise_white_acceleration` (discrete): one constant random acceleration of
    standard deviation sigma [m/s^2] per interval. It has the closed-form
    steady state of the alpha-beta filter (Kalata), used in the tests.
"""

import numpy as np
import numpy.typing as npt


def _matrix(value: npt.ArrayLike, what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{what} must be a matrix (2-D array)")
    return array


def predict(
    x: npt.ArrayLike, P: npt.ArrayLike, F: npt.ArrayLike, Q: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Move an estimate forward: x <- F x, P <- F P F^T + Q.

    Args:
        x: (n,) state estimate.
        P: (n, n) its covariance.
        F: (n, n) state transition over the interval.
        Q: (n, n) process noise over the interval.

    Returns:
        The predicted (x, P).
    """
    x = np.array(x, dtype=float)
    P = _matrix(P, "P")
    F = _matrix(F, "F")
    Q = _matrix(Q, "Q")
    n = x.shape[0]
    for name, matrix in (("P", P), ("F", F), ("Q", Q)):
        if matrix.shape != (n, n):
            raise ValueError(f"{name} must be ({n}, {n}), not {matrix.shape}")
    return F @ x, F @ P @ F.T + Q


def update(
    x: npt.ArrayLike,
    P: npt.ArrayLike,
    z: npt.ArrayLike,
    H: npt.ArrayLike,
    R: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Blend a measurement z = H x + noise into an estimate.

    Args:
        x: (n,) state estimate.
        P: (n, n) its covariance.
        z: (m,) measurement.
        H: (m, n) measurement matrix.
        R: (m, m) measurement noise covariance.

    Returns:
        (x, P, y, S): the updated estimate, the innovation y = z - H x and its
        covariance S = H P H^T + R (both before the update).
    """
    x = np.array(x, dtype=float)
    P = _matrix(P, "P")
    z = np.array(z, dtype=float)
    H = _matrix(H, "H")
    R = _matrix(R, "R")
    n, m = x.shape[0], z.shape[0]
    if P.shape != (n, n):
        raise ValueError(f"P must be ({n}, {n}), not {P.shape}")
    if H.shape != (m, n):
        raise ValueError(f"H must be ({m}, {n}), not {H.shape}")
    if R.shape != (m, m):
        raise ValueError(f"R must be ({m}, {m}), not {R.shape}")
    y = z - H @ x
    S = H @ P @ H.T + R
    # K = P H^T S^-1, solved instead of inverting S: S is symmetric, so
    # K^T = S^-1 H P and K = (S^-1 (H P))^T.
    K = np.linalg.solve(S, H @ P).T
    A = np.eye(n) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    return x + K @ y, 0.5 * (P_new + P_new.T), y, S


def nees(error: npt.ArrayLike, P: npt.ArrayLike) -> float:
    """Normalised estimation error squared, e^T P^-1 e (e = estimate - truth).

    For a filter whose P is honest, its average over many runs is the number of
    state components (a chi-square variable with n degrees of freedom).
    """
    error = np.asarray(error, dtype=float)
    return float(error @ np.linalg.solve(_matrix(P, "P"), error))


def constant_velocity_transition(dt: float, dimensions: int = 3) -> np.ndarray:
    """F of the constant-velocity model over dt: position += velocity * dt.

    State [positions (dimensions), velocities (dimensions)].
    """
    if not dt >= 0.0:
        raise ValueError("dt must be >= 0")
    eye = np.eye(dimensions)
    F = np.eye(2 * dimensions)
    F[:dimensions, dimensions:] = dt * eye
    return F


def white_noise_acceleration(dt: float, q: float, dimensions: int = 3) -> np.ndarray:
    """Q of the constant-velocity model with continuous white-noise acceleration.

    q is the spectral density of the acceleration [m^2/s^3] on every axis:
        Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]  (each block times the identity).
    A rough rule: sqrt(q * T) is the size of the velocity change [m/s] the model
    allows over an interval T.
    """
    if not dt >= 0.0:
        raise ValueError("dt must be >= 0")
    if not q >= 0.0:
        raise ValueError("q must be >= 0")
    eye = np.eye(dimensions)
    return q * np.block(
        [[dt**3 / 3 * eye, dt**2 / 2 * eye], [dt**2 / 2 * eye, dt * eye]]
    )


def piecewise_white_acceleration(
    dt: float, sigma: float, dimensions: int = 3
) -> np.ndarray:
    """Q of the constant-velocity model with one random constant acceleration per
    interval, of standard deviation sigma [m/s^2] on every axis:
        Q = sigma^2 g g^T,  g = [dt^2/2, dt]  (each block times the identity).
    """
    if not dt >= 0.0:
        raise ValueError("dt must be >= 0")
    if not sigma >= 0.0:
        raise ValueError("sigma must be >= 0")
    eye = np.eye(dimensions)
    return sigma**2 * np.block(
        [[dt**4 / 4 * eye, dt**3 / 2 * eye], [dt**3 / 2 * eye, dt**2 * eye]]
    )


def alpha_beta_gains(tracking_index: float) -> tuple[float, float]:
    """Steady-state gains of the 1-D constant-velocity Kalman filter (Kalata 1984).

    With piecewise white acceleration (sigma_a) and position measurements every T
    (noise sigma_w), the gain settles at K = [alpha, beta / T], where
        lambda = sigma_a T^2 / sigma_w   (the tracking index),
        alpha  = -(lambda^2 + 8 lambda - (lambda + 4) sqrt(lambda^2 + 8 lambda)) / 8,
        beta   =  (lambda^2 + 4 lambda - lambda sqrt(lambda^2 + 8 lambda)) / 4.
    A large index (an agile body, a precise sensor) trusts the measurements
    (alpha -> 1); a small one smooths them (alpha -> 0).
    """
    if not tracking_index > 0.0:
        raise ValueError("the tracking index must be > 0")
    lam = tracking_index
    root = np.sqrt(lam**2 + 8 * lam)
    alpha = -(lam**2 + 8 * lam - (lam + 4) * root) / 8
    beta = (lam**2 + 4 * lam - lam * root) / 4
    return float(alpha), float(beta)
