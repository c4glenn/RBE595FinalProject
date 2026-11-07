from typing import Sequence, Tuple

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.poses import SHAKE_AMPLITUDE, SHAKE_CYCLES


def shake(
    arm: ArmMoveClient,
    position_xyz: Tuple[float, float, float],
    base_rpy: Sequence[float],
    amplitude: float = SHAKE_AMPLITUDE,
    cycles: int = SHAKE_CYCLES,
) -> bool:
    """Wiggle the held bottle by alternating the roll angle +/- `amplitude` around `base_rpy`.

    Isolated from step5_shake_over_bowl.py's "get over the bowl" logic so the shaking motion
    itself (amplitude, cycle count, or a future smoother Cartesian-path implementation instead
    of discrete MoveGroup calls per leg) can be re-tuned independently.
    """
    roll, pitch, yaw = base_rpy
    for _ in range(cycles):
        if not arm.move_to(position_xyz, (roll + amplitude, pitch, yaw)):
            return False
        if not arm.move_to(position_xyz, (roll - amplitude, pitch, yaw)):
            return False
    return arm.move_to(position_xyz, base_rpy)
