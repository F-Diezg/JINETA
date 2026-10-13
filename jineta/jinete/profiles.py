"""Vehicle profiles: parameter sets from open sources, one constant per vehicle.

A profile is only data: the vehicle models never know which vehicle they fly. Each
profile cites where its numbers come from. In phase 8 profiles will also be loaded
from YAML files into the same dataclasses.
"""

import numpy as np
import numpy.typing as npt

from jineta.jinete.control import (
    AttitudeGains,
    MultirotorGains,
    RateGains,
    VelocityGains,
)
from jineta.jinete.multirotor import MultirotorParams, Rotor


def flu_to_frd(vector: npt.ArrayLike) -> np.ndarray:
    """Convert a vector from FLU body axes (x forward, y left, z up: Gazebo, ROS)
    to the FRD body axes used here (x forward, y right, z down)."""
    x, y, z = vector
    return np.array([x, -y, -z], dtype=float)


# 3DR Iris quadrotor as modelled in PX4 SITL: PX4-SITL_gazebo-classic,
# models/iris/iris.sdf.jinja (base_link inertial, rotor poses, motor model plugins).
# - Mass: base_link 1.5 kg + 4 rotors x 0.005 kg.
# - Inertia: base_link only. The rotors would add 1-3 % (parallel-axis terms),
#   below the uncertainty of the published values.
# - Rotor order = PX4 motor numbering (motorNumber 0-3). Positions given in FLU.
# - Rotor drag: rotorDragCoefficient of the same file.
# - Body drag: the Gazebo model has none. CdA = mass / B with B = 100 kg/m^2, the
#   default ballistic coefficient of the PX4 EKF2 drag model (EKF2_BCOEF_X/Y): a
#   generic multirotor value, not measured on the Iris. There is no open value for
#   the vertical axis; the same CdA is used as a placeholder.
IRIS = MultirotorParams(
    mass=1.52,
    inertia=np.diag([0.029125, 0.029125, 0.055225]),
    rotors=(
        Rotor(flu_to_frd([0.13, -0.22, 0.023]), spin=+1),  # 0 front right, ccw
        Rotor(flu_to_frd([-0.13, 0.20, 0.023]), spin=+1),  # 1 back left, ccw
        Rotor(flu_to_frd([0.13, 0.22, 0.023]), spin=-1),  # 2 front left, cw
        Rotor(flu_to_frd([-0.13, -0.20, 0.023]), spin=-1),  # 3 back right, cw
    ),
    thrust_coefficient=5.84e-6,
    moment_constant=0.06,
    max_rotor_speed=1100.0,
    rotor_drag_coefficient=0.000175,
    body_drag_area=np.full(3, 1.52 / 100.0),
)


# Rate-loop gains for the Iris. kp chosen here (PX4 gains are in the normalised units
# of its own mixer and do not transfer): roll and pitch time constant 1/kp = 50 ms;
# yaw 1/kp = 0.2 s, because the reaction torques give little yaw authority (about
# 0.8 N·m at hover thrust, ~15 rad/s^2). ki from the PX4 default ratios I/P, which
# are unit-free in that sense and do transfer: MC_ROLLRATE_I / MC_ROLLRATE_P =
# 0.2 / 0.15 and MC_YAWRATE_I / MC_YAWRATE_P = 0.1 / 0.2 [1/s]. A stronger integral
# (critical damping of the rate loop alone, ki = kp^2 / 4) left a slow tail in the
# attitude response once the loops were cascaded.
IRIS_RATE_GAINS = RateGains(
    kp=[20.0, 20.0, 5.0],
    ki=[20.0 * 0.2 / 0.15, 20.0 * 0.2 / 0.15, 5.0 * 0.1 / 0.2],
    integral_limit=[10.0, 10.0, 2.0],
)

# Attitude-loop gains for the Iris: kp = kp_rate / 4 gives a critically damped
# cascade (natural frequency kp_rate / 2: 10 rad/s roll and pitch, 2.5 rad/s yaw)
# when the rate loop is P only; with the PI rate loop the tests still see no
# overshoot in roll and pitch. Same order as the PX4 defaults, which are in the
# same units (MC_ROLL_P = 4.0, MC_YAW_P = 2.8 [1/s]). Rate limits: 3.5 rad/s
# (~200 deg/s; PX4 MC_ROLLRATE_MAX is 220 deg/s) on roll and pitch, 1.5 rad/s on
# yaw (little yaw authority).
IRIS_ATTITUDE_GAINS = AttitudeGains(
    kp=[5.0, 5.0, 1.25],
    rate_limit=[3.5, 3.5, 1.5],
)


# Velocity-loop gains for the Iris: the PX4 defaults, which use the same units
# (acceleration per velocity error): MPC_XY_VEL_P_ACC = 1.8, MPC_XY_VEL_I_ACC = 0.4,
# MPC_Z_VEL_P_ACC = 4.0, MPC_Z_VEL_I_ACC = 2.0, MPC_TILTMAX_AIR = 45 deg. The
# derivative term of PX4 (MPC_XY_VEL_D_ACC) is left out. Integral limits chosen
# here, 5 m/s^2: horizontally the drag at a 10 m/s cruise already needs ~3.6 m/s^2,
# vertically the thrust lost at 3000 m (ISA density 74 % of sea level) ~3.4 m/s^2.
IRIS_VELOCITY_GAINS = VelocityGains(
    kp=[1.8, 1.8, 4.0],
    ki=[0.4, 0.4, 2.0],
    integral_limit=[5.0, 5.0, 5.0],
    max_tilt=np.radians(45.0),
)

IRIS_GAINS = MultirotorGains(
    rate=IRIS_RATE_GAINS,
    attitude=IRIS_ATTITUDE_GAINS,
    velocity=IRIS_VELOCITY_GAINS,
)
