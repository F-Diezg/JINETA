"""What a drone knows about itself: sensors and an onboard estimate.

A real drone does not know its state; it estimates it from its sensors. Indoors,
with a motion-capture system (the setup assumed in phase 12: cameras around the
room that see markers on the drone), the chain is:

    motion capture --(delay)--> estimator <-- IMU (gyroscope, accelerometer)
                                    |
                     estimated state (position, velocity, attitude, rates)
                                    |
                                  policy

Sensors, every physics step (dt, 100 Hz):
  - motion capture: position and attitude, each with white noise (the cameras run
    faster than 100 Hz, 240 Hz for an OptiTrack PrimeX 13, so a frame is
    available every step), reaching the estimator `mocap_delay` later;
  - gyroscope: body rates + a bias (constant during an episode, drawn at its
    start) + white noise of density N (standard deviation N / sqrt(dt) per
    sample, as PX4's Gazebo IMU plugin draws it);
  - accelerometer: the specific force in body axes, f = R^T (a - g) (what an
    accelerometer measures: everything but gravity; 1 g up when resting on the
    floor), + a bias + white noise, the same way. The acceleration a is the mean
    over the step, (v_k - v_(k-1)) / dt.

The estimator does what PX4's EKF2 does with a delayed sensor, in a simpler
form:
  1. at the delayed horizon (the time of the newest motion-capture frame, now -
     delay) a Kalman filter for position and velocity: predict with the
     accelerometer (rotated to NED with the frame's attitude, plus gravity:
     v += a dt, p += v dt + a dt^2 / 2), update with the frame's position;
  2. an output predictor carries that horizon estimate to now with the IMU
     samples since then (attitude by the gyroscope, then velocity and position
     by the accelerometer), so the policy is not given a delayed state.
The attitude at the horizon is the frame's (motion capture measures it directly;
it is not filtered); the rates given are the latest gyroscope sample.

The filter's gain is its steady state (Kalata's alpha-beta gains,
`atalaya.kalman.alpha_beta_gains`): every drone's covariance is the same (the
same sensors, a measurement every step) and settles in a few steps, so the gain
is computed once (with perfect frames the position is the frame's and the
velocity is left to the input; with a perfect input nothing is corrected). Its
process noise is what the input acceleration gets wrong
per step: the accelerometer's noise and the attitude noise's leak of gravity
(g times the attitude error), sqrt((N_a / sqrt(dt))^2 + (g sigma_att)^2).

A drone restarted on the floor starts with buffers as if it had stood still at
its start, and its biases are drawn again. The random draws come from one
generator for the whole batch: the statistics of a drone's errors do not depend
on its batch, the numbers do.

Not modelled: the accelerometer's and gyroscope's vibration noise (much larger
than the datasheet noise on a real drone; PX4 filters it), the estimation of the
biases (EKF2 estimates them; here a bias stays as drawn), marker occlusions and
dropped frames.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.atalaya.kalman import alpha_beta_gains
from jineta.core.environment import GRAVITY
from jineta.core.vector import matvec
from jineta.escuadron import quaternion

Drones = npt.ArrayLike | None


@dataclass(frozen=True)
class Sensors:
    """The sensors of a drone and how wrong they are (standard deviations; the
    defaults are a motion-capture lab, `MOCAP_LAB`).

    gyro_noise_density: [rad/s/sqrt(Hz)].
    accel_noise_density: [m/s^2/sqrt(Hz)].
    gyro_bias, accel_bias: spread of the constant bias drawn per episode on each
        axis [rad/s], [m/s^2].
    mocap_position: noise of a frame's position on each axis [m].
    mocap_attitude: noise of a frame's attitude on each axis (a small rotation
        in body axes) [rad].
    mocap_delay: from the frame's moment to its arrival at the estimator [s]
        (rounded to whole physics steps).
    """

    gyro_noise_density: float = 8.726646e-4
    accel_noise_density: float = 6.37e-3
    gyro_bias: float = 0.0
    accel_bias: float = 0.0
    mocap_position: float = 1e-4
    mocap_attitude: float = float(np.radians(0.25))
    mocap_delay: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "gyro_noise_density",
            "accel_noise_density",
            "gyro_bias",
            "accel_bias",
            "mocap_position",
            "mocap_attitude",
            "mocap_delay",
        ):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be >= 0")

    def delay_steps(self, dt: float) -> int:
        """The motion-capture delay in whole physics steps."""
        return round(self.mocap_delay / dt)


# A motion-capture lab, every number from an open source:
# - IMU noise densities: the defaults of PX4's Gazebo IMU plugin
#   (PX4-SITL_gazebo-classic, include/gazebo_imu_plugin.h, "for use with
#   IIM42653": gyroscope 0.0008726646 rad/s/sqrt(Hz), 0.05 deg/s;
#   accelerometer 0.00637 m/s^2/sqrt(Hz), 0.65 mg-rms), which the Iris model
#   uses. The plugin's turn-on biases are 0; so are these.
# - Motion capture: OptiTrack PrimeX 13 (optitrack.com specs): 3D accuracy
#   +-0.20 mm and rotational errors below 0.5 deg (distributor's page),
#   taken as two standard deviations: 0.1 mm and 0.25 deg; 240 Hz, 4.2 ms of
#   camera latency. The delay to the estimator also counts the tracking
#   software, the network and the link to PX4, for which no open number was
#   found: 20 ms is a chosen value (to be measured on the real setup, and
#   randomised in training).
MOCAP_LAB = Sensors()


def specific_force(previous: np.ndarray, states: np.ndarray, dt: float) -> np.ndarray:
    """What a perfect accelerometer measures over each step (n, 3) [m/s^2]:
    R^T (a - g) in body axes, a the mean acceleration (v_k - v_(k-1)) / dt, R the
    attitude at the end of the step."""
    acceleration = (states[:, 3:6] - previous[:, 3:6]) / dt
    acceleration[:, 2] -= GRAVITY
    rotation = quaternion.to_matrix(states[:, 6:10])
    return matvec(rotation.swapaxes(-1, -2), acceleration)


class BatchEstimator:
    """Sensors and the onboard estimate of n drones (see the module).

    `reset(drones, states, rng)` starts chosen drones from rest at `states`;
    `step(previous, states, rng)` takes every drone's true state before and after
    a physics step; `estimate` is the estimated state (n, 13) in the layout of a
    rigid-body state (position, velocity, attitude, rates), which an observer
    reads as it reads the true one.
    """

    def __init__(self, sensors: Sensors, n: int, dt: float) -> None:
        if n < 1 or not dt > 0.0:
            raise ValueError("n must be >= 1 and dt > 0")
        self.sensors, self.n, self.dt = sensors, n, dt
        self.delay = sensors.delay_steps(dt)
        slots = self.delay + 1
        # Ring buffers over the last delay + 1 steps (slot = step % slots).
        self.frame_position = np.zeros((slots, n, 3))
        self.frame_attitude = np.zeros((slots, n, 4))
        self.frame_attitude[..., 0] = 1.0
        self.gyro = np.zeros((slots, n, 3))
        self.accel = np.zeros((slots, n, 3))
        self.gyro_bias = np.zeros((n, 3))
        self.accel_bias = np.zeros((n, 3))
        # The filter at the horizon, and the estimate now.
        self.position = np.zeros((n, 3))
        self.velocity = np.zeros((n, 3))
        self.estimate = np.zeros((n, 13))
        self.estimate[:, 6] = 1.0
        self.count = 0  # physics steps so far (all drones share the ring index)
        sigma_input = np.hypot(
            sensors.accel_noise_density / np.sqrt(dt),
            GRAVITY * sensors.mocap_attitude,
        )
        if sensors.mocap_position == 0.0:
            # Perfect frames: the position is the frame's, and the velocity
            # is left to the input, which is all the filter can say about it.
            self.alpha, self.beta = 1.0, 0.0
        elif sigma_input == 0.0:  # a perfect input: no correction needed
            self.alpha, self.beta = 0.0, 0.0
        else:
            self.alpha, self.beta = alpha_beta_gains(
                sigma_input * dt * dt / sensors.mocap_position
            )

    # ---- sensors -----------------------------------------------------------------

    def _frames(self, states: np.ndarray, rng: np.random.Generator) -> tuple:
        s = self.sensors
        position = states[:, 0:3] + s.mocap_position * rng.standard_normal(
            (len(states), 3)
        )
        tilt = s.mocap_attitude * rng.standard_normal((len(states), 3))
        attitude = quaternion.multiply(
            states[:, 6:10], quaternion.from_rotation_vector(tilt)
        )
        return position, quaternion.normalize(attitude)

    def _imu(
        self, rates: np.ndarray, force: np.ndarray, rng: np.random.Generator
    ) -> tuple:
        s, root = self.sensors, np.sqrt(self.dt)
        n = len(rates)
        gyro = rates + s.gyro_noise_density / root * rng.standard_normal((n, 3))
        accel = force + s.accel_noise_density / root * rng.standard_normal((n, 3))
        return gyro, accel

    # ---- life of a drone -------------------------------------------------------

    def reset(
        self,
        drones: Drones,
        states: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """These drones (None, a mask or indices) start standing still at
        `states` (one row per chosen drone, or one state for all): every buffer
        filled as if they had stood there, new biases, the estimate theirs."""
        rows = np.arange(self.n)[_rows(drones)]
        k = len(rows)
        states = np.broadcast_to(np.asarray(states, dtype=float), (k, states.shape[-1]))
        s = self.sensors
        self.gyro_bias[rows] = s.gyro_bias * rng.standard_normal((k, 3))
        self.accel_bias[rows] = s.accel_bias * rng.standard_normal((k, 3))
        rest = specific_force(states, states, self.dt)  # standing still
        for slot in range(self.delay + 1):
            position, attitude = self._frames(states, rng)
            self.frame_position[slot, rows] = position
            self.frame_attitude[slot, rows] = attitude
            gyro, accel = self._imu(np.zeros((k, 3)), rest, rng)
            self.gyro[slot, rows] = gyro + self.gyro_bias[rows]
            self.accel[slot, rows] = accel + self.accel_bias[rows]
        horizon = self.count % (self.delay + 1)  # the oldest slot after a step
        self.position[rows] = self.frame_position[horizon, rows]
        self.velocity[rows] = 0.0
        self.estimate[rows, 0:3] = states[:, 0:3]
        self.estimate[rows, 3:6] = 0.0
        self.estimate[rows, 6:10] = states[:, 6:10]
        self.estimate[rows, 10:13] = 0.0

    def step(
        self,
        previous: np.ndarray,
        states: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """One physics step of every drone: its sensors measure, the filter
        fuses the frame that arrives now, and the output predictor brings the
        estimate to now. Returns the estimate (n, 13)."""
        self.count += 1
        slots = self.delay + 1
        now = self.count % slots
        position, attitude = self._frames(states, rng)
        self.frame_position[now] = position
        self.frame_attitude[now] = attitude
        gyro, accel = self._imu(
            states[:, 10:13], specific_force(previous, states, self.dt), rng
        )
        self.gyro[now] = gyro + self.gyro_bias
        self.accel[now] = accel + self.accel_bias

        # 1. The filter at the horizon: the frame measured `delay` steps ago.
        horizon = (self.count - self.delay) % slots
        dt = self.dt
        frame_attitude = self.frame_attitude[horizon]
        acceleration = self._ned(frame_attitude, self.accel[horizon])
        predicted = self.position + self.velocity * dt + 0.5 * acceleration * dt * dt
        velocity = self.velocity + acceleration * dt
        residual = self.frame_position[horizon] - predicted
        self.position = predicted + self.alpha * residual
        self.velocity = velocity + (self.beta / dt) * residual

        # 2. The output predictor: from the horizon to now with the IMU.
        p, v, q = self.position, self.velocity, frame_attitude
        for j in range(self.count - self.delay + 1, self.count + 1):
            mean_rate = 0.5 * (self.gyro[(j - 1) % slots] + self.gyro[j % slots])
            q = quaternion.normalize(
                quaternion.multiply(q, quaternion.from_rotation_vector(mean_rate * dt))
            )
            a = self._ned(q, self.accel[j % slots])
            p = p + v * dt + 0.5 * a * dt * dt
            v = v + a * dt
        self.estimate[:, 0:3] = p
        self.estimate[:, 3:6] = v
        self.estimate[:, 6:10] = q
        self.estimate[:, 10:13] = self.gyro[now]
        return self.estimate

    @staticmethod
    def _ned(attitude: np.ndarray, force: np.ndarray) -> np.ndarray:
        """Acceleration in NED from a specific force in body axes: R f + g."""
        acceleration = matvec(quaternion.to_matrix(attitude), force)
        acceleration[:, 2] += GRAVITY
        return acceleration


def _rows(drones: Drones) -> slice | np.ndarray:
    return slice(None) if drones is None else np.asarray(drones)
