from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import still_holding

NAME = "4/8 regrasp_side"


def run(ctx: StepContext) -> bool:
    # Aimed at the jar where step 3 saw it standing in the stand.
    side = poses.staging_side_poses(ctx.staging_jar_top)
    # Slide away from the bottle still top-down, turn side-on there, then come straight down
    # beside it: turning right above it, the wrist can swing the fingers into it.
    if not ctx.arm.move_linear(side["retreat_high"], poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_to(side["retreat_high"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_linear(side["approach"], poses.SIDE_ORIENTATION_RPY):
        return False
    # Slide in horizontally around the bottle, then lift it straight up out of the pocket.
    if not ctx.bottle.clear():
        return False
    if not ctx.arm.move_linear(side["grasp"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(side["lift"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "lifting it out of the stand"):
        return False
    return ctx.bottle.hold("side")
