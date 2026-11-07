from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "6/8 return_to_staging"


def run(ctx: StepContext) -> bool:
    # Undo the pour tilt before leaving the bowl -- BOWL_OVER_EE_XYZ is only reachable at
    # BOWL_APPROACH_ORIENTATION_RPY's yaw (pi), not SIDE_ORIENTATION_RPY's (pi/2).
    if not ctx.arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.STAGING_SIDE_GRASP_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    return ctx.arm.move_to(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY)
