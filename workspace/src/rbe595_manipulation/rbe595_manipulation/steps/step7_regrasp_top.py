from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import aim, still_holding

NAME = "7/8 regrasp_top"


def run(ctx: StepContext) -> bool:
    rpy = poses.STAGING_TOP_ORIENTATION_RPY
    # Step 6 left the hand over the jar, where it saw it; close in on it the way step 1 does in
    # the rack (look, move, look), then straight down onto it.
    top = aim(ctx, ctx.staging_jar_top, poses.staging_top_poses, rpy, "in the regrasp stand")
    if top is None:
        return False
    ctx.staging_jar_top = top
    approach, grasp, _ = poses.staging_top_poses(top)
    if not ctx.bottle.clear():
        return False
    if not ctx.arm.move_linear(grasp, rpy):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(approach, rpy):
        return False
    if not still_holding(ctx, "lifting it out of the stand"):
        return False
    return ctx.bottle.hold("top")
