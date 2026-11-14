from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import check_seated, measure_jar, still_holding

NAME = "6/8 return_to_stand"


def run(ctx: StepContext) -> bool:
    # Turn the bottle upright again (pure rotation) before leaving the bowl. Without the shake
    # step 4 left it upright, right above the stand.
    if ctx.shake and not ctx.arm.move_linear(
        poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY
    ):
        return False
    seated = poses.seated_top(ctx.stand_seat)
    side = poses.staging_side_poses(seated)
    # Back from the bowl: a free-space move. Without the shake the hand is still where step 4
    # lifted the jar, within a mm of this pose, so only a straight-line touch-up: a free-space
    # plan to a pose it is already at can pick another arm configuration and swing the jar
    # round the workspace to get there.
    move = ctx.arm.move_to if ctx.shake else ctx.arm.move_linear
    if not move(side["lift"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "bringing it back to the stand"):
        return False
    # Lower it straight down into the pocket, let go, back straight out and rise clear of it.
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(side["place"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "lowering it into the stand"):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    if not ctx.arm.move_linear(side["approach"], poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_linear(side["retreat_high"], poses.SIDE_ORIENTATION_RPY):
        return False

    # Turn top-down where the gripper is, clear of the jar, slide over it and look at its cap:
    # the jar may have slipped in the hand or shifted when let go, so the top regrasp (step 7)
    # is aimed from what the camera sees, not from where it was put.
    top_approach, _, _ = poses.staging_top_poses(seated)
    if not ctx.arm.move_to(side["retreat_high"], poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_linear(top_approach, poses.STAGING_TOP_ORIENTATION_RPY):
        return False
    measured = measure_jar(ctx, "back in the regrasp stand")
    if measured is None:
        return False
    top = measured[0]
    if not check_seated(ctx, top, seated, "the regrasp stand"):
        return False
    ctx.staging_jar_top = top
    return ctx.bottle.standing_at(top)
