"""Vehicle profiles: parameter sets from open sources, one constant per vehicle.

A profile is only data: the vehicle models never know which vehicle they fly. Each
profile cites where its numbers come from. Files that describe a flight name the
profiles (the tables at the end); the profiles themselves stay here, in code, next
to their sources.
"""

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from jineta.jinete.contact import LandingGear, box_bottom_corners
from jineta.jinete.control import (
    AttitudeGains,
    MultirotorGains,
    RateGains,
    VelocityGains,
)
from jineta.jinete.fixed_wing import FixedWingParams
from jineta.jinete.fixed_wing_control import AutopilotDesign
from jineta.jinete.hitbox import (
    Hitbox,
    bounding_sphere,
    box_corners,
    box_spheres,
    rotor_spheres,
    union,
)
from jineta.jinete.multirotor import MultirotorParams, Rotor
from jineta.jinete.propulsion import PropulsionParams


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
# - Motor lag: timeConstantUp = 0.0125 s and timeConstantDown = 0.025 s of the
#   motor model plugins of the same file (a first-order filter on the rotor
#   speed, gazebo_motor_model.cpp), used when a vehicle has motor_lag.
# - Propeller radius: 0.128 m, the rotor_N_collision cylinders of the same file
#   (for the ground effect, and the hitbox below).
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
    motor_time_constant_up=0.0125,
    motor_time_constant_down=0.025,
    propeller_radius=0.128,
)

# Landing gear of the Iris: the four bottom corners of its collision box in the same
# file (base_link collision: box 0.47 x 0.47 x 0.11 m centred on the centre of
# mass), so it rests 0.055 m below its centre of mass.
# - Sink 1 mm at rest: the file's contact min_depth (0.001 m), the depth Gazebo lets
#   a contact sink before pushing back. Gazebo's contact is nearly rigid (solved as
#   a constraint, kp = 1e12); a penalty spring cannot be, so it is sized to sink
#   that much: k = m g / (4 x 0.001) = 3727 N/m per corner.
# - Friction mu = 1: the file's <friction><ode/> is empty, so the SDF default
#   (mu = 1) applies.
# - Damping ratio 0.5 and slip speed 0.2 m/s: chosen, not from the source, so that
#   RK4 at 100 Hz is stable on the ground (see jinete/contact.py): a damping ratio
#   of 0.7 or a slip speed of 0.1 m/s already makes it unstable.
IRIS_LANDING_GEAR = LandingGear.from_damping_ratio(
    box_bottom_corners([0.47, 0.47, 0.11]),
    mass=1.52,
    sink=0.001,
    damping_ratio=0.5,
    friction=1.0,
    slip_speed=0.2,
)

# Collision shape of the Iris, from the collision geometry of the same file:
# - each rotor: rotor_N_collision, a cylinder of radius 0.128 m and length 0.005 m
#   at the hub: one sphere per rotor at its hub, radius sqrt(0.128^2 + 0.0025^2);
# - the body: base_link collision box 0.47 x 0.47 x 0.11 m, centred on the CoM (it
#   takes in the arms and the legs). One sphere around it would be 0.337 m (0.67 m
#   high); cut into 3 x 3 cells, nine spheres of 0.124 m cover it.
# Chosen 3 x 3 (13 spheres in all): 0.275 m high (0.151 m up, the rotors; 0.124 m
# down) and 0.280 m from the CoM towards the nose, where one sphere around
# everything (IRIS_BOUNDING_SPHERE) is 0.384 m every way. 4 x 4 cells would gain
# 2 cm at the bottom only, for 7 more spheres.
IRIS_PROPELLER_RADIUS = IRIS.propeller_radius
IRIS_PROPELLER_THICKNESS = 0.005
IRIS_COLLISION_BOX = (0.47, 0.47, 0.11)
IRIS_HITBOX = union(
    rotor_spheres(
        [rotor.position for rotor in IRIS.rotors],
        IRIS_PROPELLER_RADIUS,
        IRIS_PROPELLER_THICKNESS,
    ),
    box_spheres(IRIS_COLLISION_BOX, (3, 3, 1)),
)
IRIS_BOUNDING_SPHERE = bounding_sphere(
    [rotor.position for rotor in IRIS.rotors],
    IRIS_PROPELLER_RADIUS,
    box_corners(IRIS_COLLISION_BOX),
    IRIS_PROPELLER_THICKNESS,
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


# Generic small fixed-wing UAV: Aerosonde-class numbers
# from the open-source simulator of Beard & McLain, Small Unmanned Aircraft (2012),
# mavsim_public (github.com/byu-magicc/mavsim_public),
# mavsim_python/parameters/aerosonde_parameters.py. The Aerosonde is a civilian
# weather-reconnaissance UAV; its data stand here for "a small fixed-wing drone",
# not for any particular vehicle.
# - Inertia: J_x, J_y, J_z and the product J_xz of the source; the tensor carries
#   -J_xz off the diagonal (the plane of symmetry is x-z, so J_xy = J_yz = 0).
# - Drag: the source lists two drag models; this uses its polar (C_D_p, e, AR).
#   C_D_0 = 0.0424 and C_D_alpha = 0.132 of the linear model are not used.
# - The source's own density (1.2682 kg/m^3) is not part of the profile: the density
#   comes from the environment (ISA).
SMALL_FIXED_WING = FixedWingParams(
    mass=11.0,
    inertia=[[0.8244, 0.0, -0.1204], [0.0, 1.135, 0.0], [-0.1204, 0.0, 1.759]],
    wing_area=0.55,
    span=2.8956,
    chord=0.18994,
    oswald=0.9,
    c_lift_0=0.23,
    c_lift_alpha=5.61,
    c_lift_q=7.95,
    c_lift_elevator=0.13,
    stall_sharpness=50.0,
    stall_angle=0.47,
    c_drag_p=0.043,
    c_drag_q=0.0,
    c_drag_elevator=0.0135,
    c_m_0=0.0135,
    c_m_alpha=-2.74,
    c_m_q=-38.21,
    c_m_elevator=-0.99,
    c_y_0=0.0,
    c_y_beta=-0.98,
    c_y_p=0.0,
    c_y_r=0.0,
    c_y_aileron=0.075,
    c_y_rudder=0.19,
    c_ell_0=0.0,
    c_ell_beta=-0.13,
    c_ell_p=-0.51,
    c_ell_r=0.25,
    c_ell_aileron=0.17,
    c_ell_rudder=0.0024,
    c_n_0=0.0,
    c_n_beta=0.073,
    c_n_p=0.069,
    c_n_r=-0.095,
    c_n_aileron=-0.011,
    c_n_rudder=-0.069,
)


# Propulsion of the generic small fixed-wing UAV: electric motor + fixed-pitch
# propeller, from the same source file (aerosonde_parameters.py, "Propeller thrust /
# torque parameters", propulsion addendum by McLain). Static thrust at full throttle
# and sea level is ~82 N (thrust-to-weight ~0.76).
# - Propeller: D = 20 in; C_T(J), C_Q(J) quadratic fits of measured data.
# - Motor: KV = 145 RPM/V, so K = 60 / (2 pi 145) = 0.06586 V·s/rad (the source
#   uses the same value for the torque constant KQ); R = 0.042 ohm; i0 = 1.5 A.
# - Battery: 12 cells x 3.7 V = 44.4 V at full throttle.
SMALL_FIXED_WING_PROPULSION = PropulsionParams(
    diameter=20 * 0.0254,
    motor_constant=60.0 / (2.0 * np.pi * 145.0),
    resistance=0.042,
    no_load_current=1.5,
    max_voltage=12 * 3.7,
    c_thrust=(0.09357, -0.06044, -0.1079),
    c_torque=(0.005230, 0.004970, -0.01664),
)

# Autopilot design of the generic small fixed wing (successive loop closure, Beard &
# McLain ch. 6). Chosen here and checked in simulation (tests/test_fixed_wing_control):
# - surfaces +-30 deg, a typical servo throw for a small UAV (the source suggests
#   up to 45 deg for the aileron; 30 is more conservative);
# - bank limit 35 deg: at 18 m/s and 45 deg the elevator saturated and the aircraft
#   lost 12 m in a 90 deg turn;
# - pitch limit 20 deg;
# - roll and pitch saturate their surface at 15 deg of error (the source's rule);
# - outer loops 8 (course) and 15 (altitude) times slower than their inner loop:
#   with 12 the altitude loop kept a slow oscillation going after a 20 m step;
# - damping ratios 0.7-1, airspeed loop 0.8 rad/s, sideslip loop 2 rad/s.
SMALL_FIXED_WING_AUTOPILOT = AutopilotDesign(
    surface_limit=np.radians(30.0),
    max_bank=np.radians(35.0),
    max_pitch=np.radians(20.0),
    roll_error_at_saturation=np.radians(15.0),
    roll_damping=0.9,
    course_separation=8.0,
    course_damping=1.0,
    pitch_error_at_saturation=np.radians(15.0),
    pitch_damping=0.7,
    altitude_separation=15.0,
    altitude_damping=1.0,
    airspeed_frequency=0.8,
    airspeed_damping=1.0,
    sideslip_frequency=2.0,
    sideslip_damping=1.0,
)


# --- Profiles by name -----------------------------------------------------------------
# Files that describe a flight name a vehicle; these tables say which profile each
# name means. A new vehicle is a new entry here.


class MultirotorProfile(NamedTuple):
    params: MultirotorParams
    gains: MultirotorGains
    gear: LandingGear
    hitbox: Hitbox


class FixedWingProfile(NamedTuple):
    params: FixedWingParams
    propulsion: PropulsionParams
    autopilot: AutopilotDesign


MULTIROTOR_PROFILES = {
    "iris": MultirotorProfile(IRIS, IRIS_GAINS, IRIS_LANDING_GEAR, IRIS_HITBOX)
}
FIXED_WING_PROFILES = {
    "small_fixed_wing": FixedWingProfile(
        SMALL_FIXED_WING, SMALL_FIXED_WING_PROPULSION, SMALL_FIXED_WING_AUTOPILOT
    )
}
