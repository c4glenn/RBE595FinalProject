from dataclasses import dataclass

from rclpy.impl.rcutils_logger import RcutilsLogger

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.gripper_client import GripperClient


@dataclass
class StepContext:
    """The handle every step module gets: arm/gripper clients plus a logger."""

    arm: ArmMoveClient
    gripper: GripperClient
    logger: RcutilsLogger
