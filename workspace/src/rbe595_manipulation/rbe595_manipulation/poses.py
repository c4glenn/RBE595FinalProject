"""Every Cartesian waypoint the pick/place/shake sequence uses, in world-frame coordinates.

Arm: Kinova Gen3 lite 6 DoF with the gen3_lite_2f gripper (kortex_description).
"""

import math

from rbe595_manipulation._orientation import (
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
# Letting go at the staging spot, where there are no neighbours: open fully (~110 mm), so the
# fingers clear the bottle as they back away from it.
GRIPPER_OPEN_WIDE = 0.0
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

# --- rack geometry (rbe595_sim/launch/sim.launch.py + config/spice_rack.yaml) ---
RACK_SLOT_XY = (0.068, -0.067)  # world (x, y) of slot S0 (rack spawned at x=0.1, y=0.0)
RACK_SLOT_FLOOR_Z = 0.754      # 0.75 rack spawn z + 0.004 floor_thickness
RACK_WALL_TOP_Z = RACK_SLOT_FLOOR_Z + 0.020   # + pocket_depth (= rack top, 0.024 above spawn z)
BOTTLE_BASE_Z = RACK_SLOT_FLOOR_Z + 0.001    # bottle spawned 1mm above the slot floor
BOTTLE_TOP_Z = BOTTLE_BASE_Z + BOTTLE_HEIGHT

# Straight down, fingers closing along world x: the neighbouring jars are 96 mm away along x
# (column pitch) but only 67 mm along y (row pitch).
RACK_ORIENTATION_RPY = top_grasp_rpy(math.pi / 2)
RACK_APPROACH_CLEARANCE = APPROACH_CLEARANCE
RACK_GRASP_Z = BOTTLE_TOP_Z - TOP_GRASP_BELOW_TOP
RACK_GRASP_POINT = (RACK_SLOT_XY[0], RACK_SLOT_XY[1], RACK_GRASP_Z)
RACK_GRASP_EE_XYZ = offset_along_approach(
    RACK_GRASP_POINT, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
RACK_APPROACH_EE_XYZ = offset_along_approach(
    RACK_GRASP_POINT, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + RACK_APPROACH_CLEARANCE
)
# Putting the bottle back: let go of it PLACE_DROP above where it was picked up from.
RACK_PLACE_EE_XYZ = offset_along_approach(
    RACK_GRASP_POINT, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH - PLACE_DROP
)

# --- staging location: a clear patch of table, used to switch grasp from top to side and
#     back. Within easy Gen3 lite reach (0.5 m from the base), clear of the rack footprint
#     (x in [-0.016, 0.216], y in [-0.105, 0.105]), of the rack's hidden floor collision
#     (x in [-0.15, 0.35], y in [-0.25, 0.25]) and ~0.1 m from the pinned bowl below. ---
STAGING_XY = (0.0, 0.30)
STAGING_HEADING = heading_from(ARM_BASE_XY, STAGING_XY)  # base -> staging, horizontal

# Top grasp at the same height on the bottle as in the rack, so the bottle is set down
# (released) just above the table.
STAGING_TOP_GRASP_Z = TABLE_TOP_Z + PLACE_DROP + BOTTLE_HEIGHT - TOP_GRASP_BELOW_TOP
STAGING_TOP_ORIENTATION_RPY = top_grasp_rpy(STAGING_HEADING + math.pi / 2)
STAGING_TOP_GRASP_POINT = (STAGING_XY[0], STAGING_XY[1], STAGING_TOP_GRASP_Z)
STAGING_TOP_GRASP_EE_XYZ = offset_along_approach(
    STAGING_TOP_GRASP_POINT, STAGING_TOP_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
STAGING_TOP_APPROACH_EE_XYZ = offset_along_approach(
    STAGING_TOP_GRASP_POINT,
    STAGING_TOP_ORIENTATION_RPY,
    GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE,
)

# Side grasp: approach horizontally, straight out from the arm base. Grasped as high on the
# bottle as the finger pads allow (they are ~24 mm tall, centred on the grasp point), since the
# wrist behind the gripper is ~0.09 m across and has to stay clear of the table below it.
SIDE_ORIENTATION_RPY = side_grasp_rpy(STAGING_HEADING)
SIDE_GRASP_BELOW_TOP = 0.013
STAGING_SIDE_GRASP_Z = TABLE_TOP_Z + BOTTLE_HEIGHT - SIDE_GRASP_BELOW_TOP
STAGING_SIDE_GRASP_POINT = (STAGING_XY[0], STAGING_XY[1], STAGING_SIDE_GRASP_Z)
STAGING_SIDE_GRASP_XYZ = offset_along_approach(
    STAGING_SIDE_GRASP_POINT, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
STAGING_SIDE_APPROACH_XYZ = offset_along_approach(
    STAGING_SIDE_GRASP_POINT, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE
)
# Held side-on, the bottle is lifted straight up off the table (and lowered straight back down
# onto it), so it is never dragged along the table.
STAGING_SIDE_LIFT_XYZ = (
    STAGING_SIDE_GRASP_XYZ[0],
    STAGING_SIDE_GRASP_XYZ[1],
    STAGING_SIDE_GRASP_XYZ[2] + APPROACH_CLEARANCE,
)
# Above the side approach pose (~0.22 m off the staging spot's axis), level with the top-down
# approach pose: where the gripper turns between top-down and side-on. It is reached from /
# left for the top-down approach by a level slide, which keeps the fingertips a clearance above
# the bottle, and turning this far from the bottle no wrist flip can swing the fingers into it.
STAGING_SIDE_RETREAT_HIGH_XYZ = (
    STAGING_SIDE_APPROACH_XYZ[0],
    STAGING_SIDE_APPROACH_XYZ[1],
    STAGING_TOP_APPROACH_EE_XYZ[2],
)

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
