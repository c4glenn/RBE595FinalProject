from dataclasses import dataclass
from typing import Optional, Tuple

from rclpy.impl.rcutils_logger import RcutilsLogger

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.gripper_client import GripperClient
from rbe595_manipulation.scene_setup import BottleScene
from rbe595_manipulation.tag_locator import TagLocator


@dataclass
class StepContext:
    """The handle every step module gets: arm/gripper clients, the wrist camera's tag locator,
    the moved bottle's planning scene tracking, plus a logger; and the goal being worked on."""

    arm: ArmMoveClient
    gripper: GripperClient
    bottle: BottleScene
    tags: TagLocator
    logger: RcutilsLogger
    # False skips step 5: the bottle goes from the side regrasp straight back down at staging.
    shake: bool = True
    # Set per goal: the cap tag ID of the jar asked for ...
    target_tag: int = -1
    # ... and, once step 1 has found it, the centre of its top in the rack (world frame), which
    # the pick and the return both work from.
    rack_jar_top: Optional[Tuple[float, float, float]] = None
    # Where step 3 found the regrasp stand by its datum tag: the centre of its pocket floor (world
    # frame) and its yaw; the jar is set down there in steps 3 and 6.
    stand_seat: Optional[Tuple[float, float, float]] = None
    stand_yaw: float = 0.0
    # The centre of the jar's top as last seen standing in the stand: what the side grasp (step 4)
    # and the top regrasp (step 7) are aimed from.
    staging_jar_top: Optional[Tuple[float, float, float]] = None
