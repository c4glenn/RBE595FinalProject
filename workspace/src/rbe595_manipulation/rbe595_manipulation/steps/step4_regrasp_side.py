from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "4/8 regrasp_side"


def run(ctx: StepContext) -> bool:
    if not ctx.arm.move_to(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.STAGING_SIDE_GRASP_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_CLOSED):
        return False
    return ctx.arm.move_to(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY)
