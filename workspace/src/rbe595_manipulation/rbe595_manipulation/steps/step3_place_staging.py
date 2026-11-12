from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "3/8 place_staging"


def run(ctx: StepContext) -> bool:
    if not ctx.arm.move_to(poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    # Straight down to set it on the table, let go, straight back up.
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(
        poses.STAGING_TOP_GRASP_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN_WIDE):
        return False
    if not ctx.arm.move_linear(
        poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    # Back to the normal opening before swinging round: wide-open fingers sweep a lot of space.
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    return ctx.bottle.standing_at_staging()
