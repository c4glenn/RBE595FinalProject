from dataclasses import dataclass

from rclpy.impl.rcutils_logger import RcutilsLogger

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.gripper_client import GripperClient
from rbe595_manipulation.scene_setup import BottleScene


@dataclass
class StepContext:
    """The handle every step module gets: arm/gripper clients, the moved bottle's planning
    scene tracking, plus a logger."""

    arm: ArmMoveClient
    gripper: GripperClient
    bottle: BottleScene
    logger: RcutilsLogger
