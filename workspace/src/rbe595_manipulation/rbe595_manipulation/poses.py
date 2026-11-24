"""Every Cartesian waypoint the pick/place/shake sequence uses, in world-frame coordinates.

Arm: Kinova Gen3 lite 6 DoF with the gen3_lite_2f gripper (kortex_description).
"""

import math

from rbe595_manipulation._orientation import (
    _rpy_to_matrix,
    heading_from,
    offset_along_approach,
    side_grasp_rpy,
    top_grasp_rpy,
)

# The arm's base_link position in rbe595_sim's world frame (sim.launch.py's x/y/z launch args
# default to exactly this). MUST match whatever sim.launch.py actually spawned the arm at.
ARM_BASE_WORLD_XYZ = (-0.4, 0.0, 0.75)
ARM_BASE_XY = ARM_BASE_WORLD_XYZ[:2]

TABLE_TOP_Z = 0.75

# --- gripper (gen3_lite_2f, kortex_description grippers/gen3_lite_2f) ---
# Grasp point: kortex_description's tool_frame, 0.130 m out along end_effector_link +z,
# between the finger pads.
GRIPPER_FINGERTIP_REACH = 0.130

# right_finger_bottom_joint (the other finger joints mimic it). From the URDF finger geometry:
# 0.0 is fully open (~110 mm between the pads), ~0.56 touches a 48.3 mm jar, ~0.8 is shut.
# "Open" stays at 0.35 (~85 mm): wide enough to drop over a jar with ~18 mm to spare a side,
# narrow enough that the fingers stay clear of the neighbouring jars in the rack.
GRIPPER_OPEN = 0.35
# Commanded past contact so the fingers squeeze the jar; the controller stalls on it.
GRIPPER_CLOSED = 0.75
# Holding the jar stops the fingers at ~0.56-0.58; at or past this they closed on nothing.
GRIPPER_HOLDING_MAX = 0.70

APPROACH_CLEARANCE = 0.10  # [m] extra standoff for approach/retreat legs, beyond the grasp pose
PLACE_DROP = 0.004  # [m] bottle bottom above the surface (table, slot floor) when let go

DEFAULT_VELOCITY_SCALING = 0.3  # slow, safe default for every planned move

# --- bottle (B0F6LMGBQJ jar, 48.3 mm x 81.3 mm) ---
BOTTLE_HEIGHT = 0.0813
# Top grasp: grasp point this far below the bottle top. The finger pads reach ~0.03 m below
# the grasp point, so in the rack the fingertips stay above the rack top.
TOP_GRASP_BELOW_TOP = 0.012

# --- AprilTags (tag36h11): one on every jar's cap, two rack datum tags on the rack's front ledge
# (rbe595_sim models/spice_bottle.sdf.xacro, config/spice_rack.yaml). Sizes are the black
# border's edge, which is what the corners found in the image outline. ---
CAP_TAG_IDS = range(6)
CAP_TAG_SIZE = 0.016
RACK_TAG_IDS = (6, 7)
RACK_TAG_SIZE = 0.036
TAG_SIZES = {
    **{tag_id: CAP_TAG_SIZE for tag_id in CAP_TAG_IDS},
    **{tag_id: RACK_TAG_SIZE for tag_id in RACK_TAG_IDS},
}
# The cap tag sits on the uncut half of the cap, beside the pour-spout cutout: its centre 11 mm
# off the cap's centre, towards which the tag image's "up" (tag y) points, and its face 0.5 mm
# above the cap. So the centre of the jar's top, in the tag's own frame (x right, y up on the
# tag image, z out of it):
CAP_CENTRE_IN_TAG = (0.0, 0.011, -0.0005)

# --- wrist camera (RealSense D405): its lens in end_effector_link. Must match the mount:
# rbe595_sim config/wrist_camera.yaml in sim, the measured bracket on the real arm. ---
CAMERA_IN_EE_XYZ = (0.055, 0.0, 0.023)

# --- rack ---
# Straight down, fingers closing along world x: the neighbouring jars are 96 mm away along x
# (column pitch) but only 67 mm along y (row pitch).
RACK_ORIENTATION_RPY = top_grasp_rpy(math.pi / 2)
RACK_APPROACH_CLEARANCE = APPROACH_CLEARANCE

# Survey: the camera looks straight down from this high above the jar tops, over roughly where
# the rack's slots are (rack set up 0.4 m in front of the arm base, see rbe595_sim
# sim.launch.py; the slot grid sits 16 mm in front of the rack centre). From 0.25 m the D405's
# 87 x 58 deg view is ~0.47 x 0.27 m, enough for all six cap tags and both rack tags with
# margin: the tags then say where each jar, and the rack, actually are.
RACK_SURVEY_XY = (0.016, 0.0)
RACK_SURVEY_CAMERA_HEIGHT = 0.25
JAR_TOP_Z_NOMINAL = 0.754 + BOTTLE_HEIGHT  # slot floor (0.75 table + 0.004) + jar

# Rack (rbe595_sim config/spice_rack.yaml): 231 x 209 x 24 mm. In its own frame (origin under
# its centre, +x away from the front ledge) the two datum tags are centred at
# x = -len_x / 2 + tag_inset_x and y = +/-(len_y / 2 - tag_inset_y), their faces 0.3 mm below
# the rack top: tag 6 on the +y side, tag 7 on the -y side.
RACK_SIZE = (0.231, 0.209, 0.024)
RACK_TAG_X_IN_RACK = -0.231 / 2 + 0.023
RACK_TAG_DEPTH = 0.0003


def camera_over(xy, height_above_z, z, rpy):
    """End-effector position that puts the wrist camera's lens `height_above_z` straight above
    (xy, z), the gripper (and so the camera) pointing straight down with orientation rpy."""
    m = _rpy_to_matrix(rpy)
    lens_offset = [sum(m[i][k] * CAMERA_IN_EE_XYZ[k] for k in range(3)) for i in range(3)]
    return (xy[0] - lens_offset[0], xy[1] - lens_offset[1], z + height_above_z - lens_offset[2])


RACK_SURVEY_EE_XYZ = camera_over(
    RACK_SURVEY_XY, RACK_SURVEY_CAMERA_HEIGHT, JAR_TOP_Z_NOMINAL, RACK_ORIENTATION_RPY
)


def rack_poses(jar_top_xyz):
    """End-effector positions for a jar in the rack whose top centre is at jar_top_xyz (from
    its cap tag): (approach above it, top grasp, place). Placing lets go of it PLACE_DROP above
    where it was picked up from."""
    grasp_point = (jar_top_xyz[0], jar_top_xyz[1], jar_top_xyz[2] - TOP_GRASP_BELOW_TOP)
    approach = offset_along_approach(
        grasp_point, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + RACK_APPROACH_CLEARANCE
    )
    grasp = offset_along_approach(grasp_point, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH)
    place = offset_along_approach(
        grasp_point, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + PLACE_DROP
    )
    return approach, grasp, place

# --- regrasp stand (rbe595_sim config/regrasp_stand.yaml, from the stand's CAD): where the jar
#     is set down to switch from the top grasp to the side grasp and back. It lifts the jar 41 mm,
#     so the wrist of a side grasp clears the table by ~56 mm (15 mm on the bare table).
#     Stand frame: origin on the table under the column centre, +x from the column towards the
#     datum tag on the base plate. ---
STAND_TAG_ID = 8
STAND_TAG_SIZE = 0.036
STAND_TAG_X = 0.058          # tag centre from the column centre along stand +x
STAND_PLATE_SIZE = (0.136, 0.100, 0.006)
STAND_PLATE_CENTER_X = 0.018  # plate centre from the column centre along stand +x
STAND_COLUMN_RADIUS = 0.032
STAND_TOP_ABOVE_TABLE = 0.061  # column top; the pocket is a rack pocket, 20 mm deep
STAND_SEAT_ABOVE_TABLE = 0.041  # pocket floor: the jar stands on it
# Tag face in its 0.4 mm sticker recess in the plate top; the seat is this much above it.
STAND_TAG_DEPTH = 0.0003
STAND_SEAT_ABOVE_TAG = STAND_SEAT_ABOVE_TABLE - (STAND_PLATE_SIZE[2] - STAND_TAG_DEPTH)
TAG_SIZES[STAND_TAG_ID] = STAND_TAG_SIZE

# Where the stand is set up (rbe595_sim sim.launch.py STAND_XY / STAND_YAW): a clear patch of
# table within easy reach (0.5 m from the base), clear of the rack and ~0.1 m from the pinned bowl,
# its tag pad 90 deg left of the base -> stand direction. Only used to know where to look for it.
STAGING_XY = (0.0, 0.30)
STAGING_HEADING = heading_from(ARM_BASE_XY, STAGING_XY)  # base -> staging, horizontal
STAND_YAW_NOMINAL = STAGING_HEADING + math.pi / 2

# Top-down at the stand, fingers closing along the base -> stand direction: end-effector +x (the
# camera's side) then points along stand +x, so with the hand over the column the camera is right
# over the datum tag (55 mm vs 58 mm out).
STAGING_TOP_ORIENTATION_RPY = top_grasp_rpy(STAGING_HEADING + math.pi / 2)
# Look for the stand from this high above its tag, the hand (holding the jar top-down) over where
# the column should be: the jar hangs ~0.18 m below the lens, ~0.06 m above the column top.
STAND_LOOK_CAMERA_HEIGHT = 0.25
_TAG_NOMINAL_XY = (
    STAGING_XY[0] + STAND_TAG_X * math.cos(STAND_YAW_NOMINAL),
    STAGING_XY[1] + STAND_TAG_X * math.sin(STAND_YAW_NOMINAL),
)
STAND_LOOK_EE_XYZ = camera_over(
    _TAG_NOMINAL_XY,
    STAND_LOOK_CAMERA_HEIGHT,
    TABLE_TOP_Z + STAND_PLATE_SIZE[2],
    STAGING_TOP_ORIENTATION_RPY,
)


def stand_seat(stand_tag_position, stand_yaw):
    """Centre of the stand's pocket floor (world frame), from where its datum tag was seen."""
    return (
        stand_tag_position[0] - STAND_TAG_X * math.cos(stand_yaw),
        stand_tag_position[1] - STAND_TAG_X * math.sin(stand_yaw),
        stand_tag_position[2] + STAND_SEAT_ABOVE_TAG,
    )


def seated_top(seat_xyz):
    """Where the top of a jar standing in the stand's pocket is."""
    return (seat_xyz[0], seat_xyz[1], seat_xyz[2] + BOTTLE_HEIGHT)


# Standoff above the top grasp at the stand. Smaller than at the rack: the jar stands 41 mm higher
# on the stand, and top-down the arm only reaches ~1.05 m (EE z) over it, so the full
# APPROACH_CLEARANCE puts the approach out of reach. At 60 mm the open fingertips (which hang
# ~35 mm past the grasp point) are still ~14 mm over the cap, a held jar's bottom ~40 mm over
# the column top, and the camera ~0.2 m above the cap tag.
STAND_APPROACH_CLEARANCE = 0.06


def staging_top_poses(jar_top_xyz):
    """End-effector positions for a top grasp of the jar standing in the stand with its top at
    jar_top_xyz: (approach above it, grasp, place). As in the rack, the approach is where the
    camera looks down at the cap tag, and placing lets go of the jar PLACE_DROP above the seat."""
    grasp_point = (jar_top_xyz[0], jar_top_xyz[1], jar_top_xyz[2] - TOP_GRASP_BELOW_TOP)
    rpy = STAGING_TOP_ORIENTATION_RPY
    approach = offset_along_approach(
        grasp_point, rpy, GRIPPER_FINGERTIP_REACH + STAND_APPROACH_CLEARANCE
    )
    grasp = offset_along_approach(grasp_point, rpy, GRIPPER_FINGERTIP_REACH)
    place = offset_along_approach(grasp_point, rpy, GRIPPER_FINGERTIP_REACH + PLACE_DROP)
    return approach, grasp, place


# Side grasp: approach horizontally, straight out from the arm base. The finger pads (~24 mm
# tall, centred on the grasp point) go on the jar body, below the screw-on shaker lid, which
# could twist off if the jar hung from it while being tipped. The gripper base and wrist reach
# 33 mm below the grasp point side-on (kortex_description meshes): on the stand that leaves them
# ~56 mm above the table, and the fingers ~20 mm above the column top.
SIDE_ORIENTATION_RPY = side_grasp_rpy(STAGING_HEADING)
SIDE_GRASP_BELOW_TOP = 0.033


def staging_side_poses(jar_top_xyz):
    """End-effector positions for the side grasp of the jar standing in the stand with its top at
    jar_top_xyz, as a dict:
      grasp          fingers round the jar body;
      approach       APPROACH_CLEARANCE straight back from it, level;
      lift           APPROACH_CLEARANCE straight up from the grasp: the jar is lifted straight out
                     of the pocket (and lowered straight back in), never dragged;
      place          the grasp pose PLACE_DROP up: where the side-held jar is let go;
      retreat_high   above the approach, level with the top-down approach above the jar: where
                     the gripper turns between top-down and side-on. It is reached from / left for
                     the top-down approach by a level slide, which keeps the fingertips a clearance
                     above the jar, and turning this far from the jar no wrist flip can swing the
                     fingers into it."""
    point = (jar_top_xyz[0], jar_top_xyz[1], jar_top_xyz[2] - SIDE_GRASP_BELOW_TOP)
    grasp = offset_along_approach(point, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH)
    approach = offset_along_approach(
        point, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE
    )
    top_approach, _, _ = staging_top_poses(jar_top_xyz)
    return {
        "grasp": grasp,
        "approach": approach,
        "lift": (grasp[0], grasp[1], grasp[2] + APPROACH_CLEARANCE),
        "place": (grasp[0], grasp[1], grasp[2] + PLACE_DROP),
        "retreat_high": (approach[0], approach[1], top_approach[2]),
    }


# --- bowl: pin sim.launch.py's randomised bowl position to this value --
#   ros2 launch rbe595_sim sim.launch.py bowl_x:=0.0 bowl_y:=0.5
# chosen clear of the rack footprint and within random_bowl_position()'s own reach/clearance
# bounds, so it's a position that launch file would also have been willing to pick itself.
BOWL_XY = (0.0, 0.5)
BOWL_RIM_TOP_Z = 0.75 + 0.065  # bowl spawn z + bowl height (sim.launch.py BOWL_HEIGHT)

# Held side-on, the bottle is carried to sit with its grasp point above the bowl centre, high
# enough that the tipped-over bottle (cap end ~0.03 m, base end ~0.06 m from the grasp point)
# stays above the rim.
BOWL_HEADING = heading_from(ARM_BASE_XY, BOWL_XY)
BOWL_POUR_POINT = (BOWL_XY[0], BOWL_XY[1], BOWL_RIM_TOP_Z + 0.08)
BOWL_APPROACH_ORIENTATION_RPY = side_grasp_rpy(BOWL_HEADING)
# Turning about the approach axis keeps the end-effector origin fixed, so one position serves
# the upright carry pose and every tilted shake pose.
BOWL_OVER_EE_XYZ = offset_along_approach(
    BOWL_POUR_POINT, BOWL_APPROACH_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)

# Pour tilt: how far the bottle is turned over about the approach axis, from upright (0) to
# cap straight down (180 deg). 130 deg points the cap 40 deg below horizontal; shaking then
# swings it SHAKE_AMPLITUDE either way. Turning it over is a wrist (joint_6) rotation, so
# whichever direction keeps joint_6 within its +/-149 deg is used (see step5_shake_over_bowl).
BOWL_TILT_ANGLE = math.radians(130)

SHAKE_AMPLITUDE = math.radians(15)
SHAKE_CYCLES = 3
