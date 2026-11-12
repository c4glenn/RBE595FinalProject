from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "8/8 return_to_rack"


def run(ctx: StepContext) -> bool:
    if not ctx.arm.move_to(poses.RACK_APPROACH_EE_XYZ, poses.RACK_ORIENTATION_RPY):
        return False
    # Straight down into the pocket, let go just above the slot floor, straight back up.
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(poses.RACK_PLACE_EE_XYZ, poses.RACK_ORIENTATION_RPY):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    return ctx.arm.move_linear(poses.RACK_APPROACH_EE_XYZ, poses.RACK_ORIENTATION_RPY)
