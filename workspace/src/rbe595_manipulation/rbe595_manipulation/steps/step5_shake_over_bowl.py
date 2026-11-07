from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.shake import shake

NAME = "5/8 shake_over_bowl"


def run(ctx: StepContext) -> bool:
    # BOWL_OVER_EE_XYZ is only reachable at BOWL_APPROACH_ORIENTATION_RPY's yaw (pi), not
    # SIDE_ORIENTATION_RPY's (pi/2) -- the arm reorients in yaw as it carries the bottle here.
    if not ctx.arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_OVER_ORIENTATION_RPY):
        return False
    return shake(ctx.arm, poses.BOWL_OVER_EE_XYZ, poses.BOWL_OVER_ORIENTATION_RPY)
