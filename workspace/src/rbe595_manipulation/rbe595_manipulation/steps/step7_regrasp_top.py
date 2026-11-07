from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "7/8 regrasp_top"


def run(ctx: StepContext) -> bool:
    if not ctx.arm.move_to(poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.STAGING_TOP_GRASP_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_CLOSED):
        return False
    return ctx.arm.move_to(poses.STAGING_TOP_APPROACH_EE_XYZ, poses.STAGING_TOP_ORIENTATION_RPY)
