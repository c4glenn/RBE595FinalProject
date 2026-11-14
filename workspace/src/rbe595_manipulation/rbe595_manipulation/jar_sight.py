"""What the steps learn about the jar from its cap tag, and the checks that stop the goal when the
jar is not where it should be: tipped over, not seated in a pocket, or no longer in the hand.

Every grasp is aimed from a fresh look at the cap taken right before it, never from where the jar
was put, so a jar that shifted while being set down (or slipped in the hand before that) is
grasped where it actually is.
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np

from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.tag_locator import TagPose

Xyz = Tuple[float, float, float]

# Look-then-move: after each move over the jar the camera measures it again, until the hand is
# aimed at where the jar is to within this.
AIM_TOLERANCE_M = 0.0015
MAX_AIMS = 3
# A standing jar's cap faces straight up; a tag tilted more than this means the jar is leaning on
# a pocket rim or lying over.
UPRIGHT_TOLERANCE_DEG = 5.0
# A jar seated in a pocket is within this of where it should be: the pockets leave it ~1 mm of
# play, the camera reads the cap a couple of mm low from up here, and a jar caught on the lead-in
# instead sits several mm off and higher.
SEATED_XY_TOLERANCE_M = 0.004
SEATED_Z_TOLERANCE_M = 0.005


def fmt(xyz) -> str:
    return "(" + ", ".join(f"{v:.4f}" for v in xyz) + ")"


def jar_top(tag: TagPose) -> Xyz:
    """Centre of the top of the jar a cap tag is on (world frame)."""
    return tuple(float(v) for v in tag.position + tag.rotation @ np.array(poses.CAP_CENTRE_IN_TAG))


def tilt_deg(tag: TagPose) -> float:
    """How far a tag's face is turned away from facing straight up."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(tag.rotation[2, 2])))))


def still_holding(ctx: StepContext, doing: str) -> bool:
    """The fingers still stop on the jar; if it slipped out they close on nothing."""
    if ctx.gripper.holding(poses.GRIPPER_HOLDING_MAX):
        return True
    ctx.logger.error(f"lost the jar while {doing}: the fingers closed on nothing")
    return False


def measure_jar(ctx: StepContext, where: str) -> Optional[Tuple[Xyz, TagPose]]:
    """Look at the target jar's cap from where the hand is: (centre of its top, its cap tag), or
    None (logged) if the tag is not seen or the jar is not standing upright."""
    seen = ctx.tags.detect()
    tag = seen.get(ctx.target_tag)
    if tag is None:
        ctx.logger.error(f"cannot see cap tag {ctx.target_tag} {where}")
        return None
    tilt = tilt_deg(tag)
    if tilt > UPRIGHT_TOLERANCE_DEG:
        ctx.logger.error(
            f"jar {ctx.target_tag} is not upright {where}: its cap is tilted {tilt:.1f} deg"
        )
        return None
    return jar_top(tag), tag


def check_seated(ctx: StepContext, top: Xyz, expected_top: Xyz, pocket: str) -> bool:
    """The jar's top is where it is when the jar stands in `pocket`."""
    off_xy = math.dist(top[:2], expected_top[:2])
    off_z = top[2] - expected_top[2]
    if off_xy > SEATED_XY_TOLERANCE_M or abs(off_z) > SEATED_Z_TOLERANCE_M:
        ctx.logger.error(
            f"jar {ctx.target_tag} is not seated in {pocket}: its top is {off_xy * 1000:.1f} mm "
            f"off the pocket centre and {off_z * 1000:+.1f} mm off the seated height"
        )
        return False
    ctx.logger.info(
        f"jar {ctx.target_tag} seated in {pocket} ({off_xy * 1000:.1f} mm off centre, "
        f"{off_z * 1000:+.1f} mm in height)"
    )
    return True


def aim(
    ctx: StepContext,
    top_guess: Xyz,
    poses_for: Callable[[Xyz], tuple],
    rpy,
    where: str,
    first_move_free: bool = False,
) -> Optional[Xyz]:
    """Look-move-look: move to the approach above where the jar is thought to be (poses_for(top)[0],
    which puts the camera over it), measure it there, and move again to the measurement, until the
    correction is negligible. Returns the measured centre of its top, or None (logged)."""
    top = top_guess
    correction = 0.0
    for look in range(1, MAX_AIMS + 1):
        approach = poses_for(top)[0]
        moved = (
            ctx.arm.move_to(approach, rpy)
            if first_move_free and look == 1
            else ctx.arm.move_linear(approach, rpy)
        )
        if not moved:
            return None
        measured = measure_jar(ctx, f"above it {where}")
        if measured is None:
            return None
        correction = math.dist(measured[0], top)
        ctx.logger.info(
            f"aim {look} {where}: jar {ctx.target_tag} top at {fmt(measured[0])}, "
            f"{correction * 1000:.1f} mm from where the hand was aimed"
        )
        top = measured[0]
        if correction < AIM_TOLERANCE_M:
            break
    else:
        ctx.logger.warn(f"still {correction * 1000:.1f} mm off after {MAX_AIMS} looks; going on")
    # End right over the measured jar.
    if not ctx.arm.move_linear(poses_for(top)[0], rpy):
        return None
    return top
