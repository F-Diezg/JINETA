"""Vehicle profiles: parameter sets from open sources, one constant per vehicle.

A profile is only data: the vehicle models never know which vehicle they fly. Each
profile cites where its numbers come from. In phase 8 profiles will also be loaded
from YAML files into the same dataclasses.
"""

import numpy as np
import numpy.typing as npt

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
