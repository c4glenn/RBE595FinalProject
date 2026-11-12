from rbe595_manipulation import poses
from rbe595_manipulation._orientation import side_grasp_rpy
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.shake import shake

NAME = "5/8 shake_over_bowl"


def run(ctx: StepContext) -> bool:
    # Carry the bottle upright to above the bowl, then tip it over and shake. Tipping turns the
    # gripper about its own approach axis with the end effector held still: a straight-line
    # (pure rotation) move, so the arm does not swing round into another configuration.
    if not ctx.arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY):
        return False
    # The wrist can turn one way or the other depending on where joint_6 ended up.
    for tilt in (poses.BOWL_TILT_ANGLE, -poses.BOWL_TILT_ANGLE):
        if ctx.arm.move_linear(poses.BOWL_OVER_EE_XYZ, side_grasp_rpy(poses.BOWL_HEADING, tilt)):
            return shake(ctx.arm, poses.BOWL_OVER_EE_XYZ, poses.BOWL_HEADING, tilt)
        ctx.logger.info(f"cannot tip the bottle {tilt:+.2f} rad from here, trying the other way")
    return False
