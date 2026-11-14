import math

import numpy as np

from rbe595_manipulation import poses
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.jar_sight import aim, fmt, jar_top

NAME = "1/8 find_jar"


def rack_pose(seen):
    """(centre (world frame), yaw) of the rack from its two datum tags, or None unless both
    were seen. Tag 6 is on the rack's +y side, tag 7 on its -y side, both on the front ledge."""
    if not all(tag_id in seen for tag_id in poses.RACK_TAG_IDS):
        return None
    plus_y, minus_y = (seen[tag_id].position for tag_id in poses.RACK_TAG_IDS)
    y_axis = (plus_y - minus_y)[:2] / np.linalg.norm((plus_y - minus_y)[:2])
    x_axis = np.array([y_axis[1], -y_axis[0]])
    centre_xy = (plus_y + minus_y)[:2] / 2 - poses.RACK_TAG_X_IN_RACK * x_axis
    top_z = (plus_y[2] + minus_y[2]) / 2 + poses.RACK_TAG_DEPTH
    centre = (centre_xy[0], centre_xy[1], top_z - poses.RACK_SIZE[2] / 2)
    return centre, math.atan2(x_axis[1], x_axis[0])


def run(ctx: StepContext) -> bool:
    if not ctx.gripper.move_to(poses.GRIPPER_OPEN):
        return False

    # Survey: the camera looks down at the whole rack, and every cap tag says which jar is where.
    if not ctx.arm.move_to(poses.RACK_SURVEY_EE_XYZ, poses.RACK_ORIENTATION_RPY):
        return False
    seen = ctx.tags.detect()
    jars = {tag_id: jar_top(tag) for tag_id, tag in seen.items() if tag_id in poses.CAP_TAG_IDS}
    ctx.logger.info(
        "survey: "
        + (", ".join(f"tag {i} at {fmt(jars[i])}" for i in sorted(jars)) or "no cap tags seen")
    )
    if ctx.target_tag not in jars:
        ctx.logger.error(f"jar with cap tag {ctx.target_tag} not in the rack")
        return False
    # The rack goes into the planning scene where its datum tags are, the other jars where
    # theirs are.
    rack = rack_pose(seen)
    if rack is None:
        ctx.logger.warn("rack datum tags not both seen; keeping the rack where it was set up")
    else:
        ctx.logger.info(f"rack at {fmt(rack[0])}, yaw {math.degrees(rack[1]):.1f} deg")
        if not ctx.bottle.set_rack(*rack):
            return False
    if not ctx.bottle.set_rack_jars(
        {tag_id: top for tag_id, top in jars.items() if tag_id != ctx.target_tag}
    ):
        return False

    # Aim: move over the jar, measure it again from close up (its tag then spans ~3x the
    # pixels), and correct, until the correction is negligible. This also checks it stands upright.
    top = aim(
        ctx,
        jars[ctx.target_tag],
        poses.rack_poses,
        poses.RACK_ORIENTATION_RPY,
        "in the rack",
        first_move_free=True,
    )
    if top is None:
        return False
    ctx.rack_jar_top = top
    return True
