from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import check_seated, measure_jar, still_holding

NAME = "8/8 return_to_rack"


def run(ctx: StepContext) -> bool:
    # Back to the pocket it came out of (where step 1 saw it).
    approach, _, place = poses.rack_poses(ctx.rack_jar_top)
    if not ctx.arm.move_to(approach, poses.RACK_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "carrying it back to the rack"):
        return False
    # Straight down into the pocket, let go just above the slot floor, straight back up.
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(place, poses.RACK_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "lowering it into the rack"):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    if not ctx.arm.move_linear(approach, poses.RACK_ORIENTATION_RPY):
        return False
    # The camera is over the cap again: check the jar went down into its pocket, upright, where
    # it was picked from (both seen by the same camera from the same spot).
    measured = measure_jar(ctx, "back in the rack")
    if measured is None:
        return False
    return check_seated(ctx, measured[0], ctx.rack_jar_top, "its rack pocket")
