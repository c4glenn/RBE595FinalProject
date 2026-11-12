from typing import Tuple

from rbe595_manipulation._orientation import side_grasp_rpy
from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.poses import SHAKE_AMPLITUDE, SHAKE_CYCLES


def shake(
    arm: ArmMoveClient,
    position_xyz: Tuple[float, float, float],
    heading: float,
    tilt: float,
    amplitude: float = SHAKE_AMPLITUDE,
    cycles: int = SHAKE_CYCLES,
) -> bool:
    """Wiggle the side-held bottle by alternating its pour tilt +/- `amplitude` around `tilt`.

    The tilt turns the gripper about its own approach axis (see side_grasp_rpy), so the
    end-effector stays at `position_xyz` throughout and only the wrist rotates; each leg is a
    straight-line (pure rotation) move so the arm never swings into another configuration.

    Isolated from step5_shake_over_bowl.py's "get over the bowl" logic so the shaking motion
    itself (amplitude, cycle count, or a future smoother Cartesian-path implementation instead
    of discrete MoveGroup calls per leg) can be re-tuned independently.
    """
    for _ in range(cycles):
        if not arm.move_linear(position_xyz, side_grasp_rpy(heading, tilt + amplitude)):
            return False
        if not arm.move_linear(position_xyz, side_grasp_rpy(heading, tilt - amplitude)):
            return False
    return arm.move_linear(position_xyz, side_grasp_rpy(heading, tilt))
