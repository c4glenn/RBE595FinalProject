from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext

NAME = "4/8 regrasp_side"


def run(ctx: StepContext) -> bool:
    # Slide away from the bottle still top-down, turn side-on there, then come straight down
    # beside it: turning right above it, the wrist can swing the fingers into it.
    if not ctx.arm.move_linear(
        poses.STAGING_SIDE_RETREAT_HIGH_XYZ, poses.STAGING_TOP_ORIENTATION_RPY
    ):
        return False
    if not ctx.arm.move_to(poses.STAGING_SIDE_RETREAT_HIGH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_APPROACH_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    # Slide in horizontally around the bottle, then lift it straight up off the table.
    if not ctx.bottle.clear():
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_GRASP_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    if not ctx.gripper.grasp(poses.GRIPPER_CLOSED, poses.GRIPPER_HOLDING_MAX):
        return False
    if not ctx.arm.move_linear(poses.STAGING_SIDE_LIFT_XYZ, poses.SIDE_ORIENTATION_RPY):
        return False
    return ctx.bottle.hold("side")
