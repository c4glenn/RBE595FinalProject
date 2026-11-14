import math

from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import (
    check_seated,
    fmt,
    measure_jar,
    still_holding,
    tilt_deg,
)

NAME = "3/8 place_on_stand"

# The stand's base plate lies flat on the table; a datum tag tilted more than this was misread.
STAND_TILT_TOLERANCE_DEG = 5.0


def run(ctx: StepContext) -> bool:
    rpy = poses.STAGING_TOP_ORIENTATION_RPY
    # Find the stand: the hand, still holding the jar, over where its column should be, which puts
    # the camera over its datum tag.
    if not ctx.arm.move_to(poses.STAND_LOOK_EE_XYZ, rpy):
        return False
    if not still_holding(ctx, "carrying it to the stand"):
        return False
    tag = ctx.tags.detect().get(poses.STAND_TAG_ID)
    if tag is None:
        ctx.logger.error(f"regrasp stand not found: its datum tag {poses.STAND_TAG_ID} is not in view")
        return False
    if tilt_deg(tag) > STAND_TILT_TOLERANCE_DEG:
        ctx.logger.error(f"regrasp stand tag tilted {tilt_deg(tag):.1f} deg; not trusting it")
        return False
    # The tag image's right (+x) points along the stand's -x, from the tag back to the column.
    ctx.stand_yaw = math.atan2(-tag.rotation[1, 0], -tag.rotation[0, 0])
    ctx.stand_seat = poses.stand_seat(tag.position, ctx.stand_yaw)
    ctx.logger.info(
        f"regrasp stand: pocket floor at {fmt(ctx.stand_seat)}, yaw {math.degrees(ctx.stand_yaw):.1f} deg"
    )
    if not ctx.bottle.set_stand(ctx.stand_seat, ctx.stand_yaw):
        return False

    # Over the pocket, straight down to set the jar in it, let go, straight back up.
    seated = poses.seated_top(ctx.stand_seat)
    approach, _, place = poses.staging_top_poses(seated)
    if not ctx.arm.move_linear(approach, rpy):
        return False
    if not ctx.bottle.let_go():
        return False
    if not ctx.arm.move_linear(place, rpy):
        return False
    if not still_holding(ctx, "lowering it into the stand"):
        return False
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False
    if not ctx.arm.move_linear(approach, rpy):
        return False

    # The camera is now over the cap: check the jar went into the pocket and stands upright, and
    # measure it for the side grasp.
    measured = measure_jar(ctx, "in the regrasp stand")
    if measured is None:
        return False
    top = measured[0]
    if not check_seated(ctx, top, seated, "the regrasp stand"):
        return False
    ctx.staging_jar_top = top
    return ctx.bottle.standing_at(top)
