from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "2/8 grasp_top"


def run(ctx: StepContext) -> bool:
    # Straight down onto the bottle and straight back up out of the pocket.
    if not ctx.arm.move_linear(poses.RACK_GRASP_EE_XYZ, poses.RACK_ORIENTATION_RPY):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(poses.RACK_APPROACH_EE_XYZ, poses.RACK_ORIENTATION_RPY):
        return False
    return ctx.bottle.hold("top")
