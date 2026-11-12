from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "6/8 return_to_staging"


def run(ctx: StepContext) -> bool:
    # Turn the bottle upright again (pure rotation) before leaving the bowl.
    if not ctx.arm.move_linear(poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(poses.STAGING_SIDE_LIFT_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    # Lower it straight down onto the table, let go, back straight out and rise clear of it.
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_GRASP_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN_WIDE):
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_RETREAT_HIGH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    # Back to the normal opening before swinging round: wide-open fingers sweep a lot of space.
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    return ctx.bottle.standing_at_staging()
