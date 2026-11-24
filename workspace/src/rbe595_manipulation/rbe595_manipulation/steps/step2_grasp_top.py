from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import still_holding

NAME = "2/8 grasp_top"


def run(ctx: StepContext) -> bool:
    # Straight down onto the jar step 1 found, and straight back up out of the pocket.
    approach, grasp, _ = poses.rack_poses(ctx.rack_jar_top)
    if not ctx.arm.move_linear(grasp, poses.RACK_ORIENTATION_RPY):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(approach, poses.RACK_ORIENTATION_RPY):
        return False
    if not still_holding(ctx, "lifting it out of the rack"):
        return False
    return ctx.bottle.hold("top")
