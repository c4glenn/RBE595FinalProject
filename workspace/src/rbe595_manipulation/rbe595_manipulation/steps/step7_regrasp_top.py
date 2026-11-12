from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "7/8 regrasp_top"


def run(ctx: StepContext) -> bool:
    # Turn top-down where step 6 left the gripper, clear of the bottle, then slide over it.
    if not ctx.arm.move_to(
        poses.STAGING_SIDE_RETREAT_HIGH_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    if not ctx.arm.move_linear(
        poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    if not ctx.bottle.clear():
        return False
    if not ctx.arm.move_linear(
        poses.STAGING_TOP_GRASP_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(
        poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    return ctx.bottle.hold("top")
