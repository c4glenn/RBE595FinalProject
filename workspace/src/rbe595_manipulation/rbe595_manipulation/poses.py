"""Every Cartesian waypoint the pick/place/shake sequence uses, in world-frame coordinates.
"""

import math

from rbe595_manipulation._orientation import offset_along_approach

# --- gripper geometry (robotiq_2f_85_macro.urdf.xacro) ---
GRIPPER_FINGERTIP_REACH = 0.11  # [m] -- estimate from URDF joint offsets

GRIPPER_OPEN = 0.0   # robotiq_85_left_knuckle_joint fully open (SRDF group_state "Open")
GRIPPER_CLOSED = 0.8  # robotiq_85_left_knuckle_joint fully closed (SRDF group_state "Close")

APPROACH_CLEARANCE = 0.15  # [m] extra standoff for approach/retreat legs, beyond the grasp pose

DEFAULT_VELOCITY_SCALING = 0.3  # slow, safe default for every planned move

TOP_TILT_FROM_VERTICAL = math.pi  # roll -- straight down
SIDE_ORIENTATION_RPY = (math.pi / 2, 0.0, math.pi / 2)  # horizontal approach along world +X

# --- rack geometry (rbe595_sim/launch/sim.launch.py + config/spice_rack.yaml) ---
RACK_SLOT_XY = (0.06925, -0.064)  # world (x, y) of slot S0 (rack spawned at x=0.1, y=0.0)
RACK_SLOT_FLOOR_Z = 0.76       # 0.75 rack spawn z + 0.01 plate_thickness
RACK_WALL_TOP_Z = RACK_SLOT_FLOOR_Z + 0.024   # + wall_height
BOTTLE_BASE_Z = RACK_SLOT_FLOOR_Z + 0.001    # bottle spawned 1mm above the slot floor
BOTTLE_TOP_Z = BOTTLE_BASE_Z + 0.1           # + bottle height

RACK_ORIENTATION_RPY = (TOP_TILT_FROM_VERTICAL, 0.0, math.radians(90))
RACK_APPROACH_CLEARANCE = APPROACH_CLEARANCE
RACK_GRASP_Z = 0.83  # world z of the grasp point on the bottle itself
RACK_GRASP_POINT = (RACK_SLOT_XY[0], RACK_SLOT_XY[1], RACK_GRASP_Z)
RACK_GRASP_EE_XYZ = offset_along_approach(
    RACK_GRASP_POINT, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
RACK_APPROACH_EE_XYZ = offset_along_approach(
    RACK_GRASP_POINT, RACK_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + RACK_APPROACH_CLEARANCE
)

# --- staging location: a clear patch of table between the rack and the (pinned) bowl, used
#     to switch grasp from top to side and back. Clear of the rack footprint
#     (x in [-0.07, 0.27], y in [-0.22, 0.22]) and of the pinned bowl below. ---
STAGING_XY = (0.2, 0.35)
STAGING_TOP_GRASP_Z = 0.81  # mid-bottle height; no rack wall here so no need to grasp as high
# Same reasoning as the rack: at true vertical, yaw is arbitrary for a round bottle. 150 deg
# confirmed reachable here (grasp + full APPROACH_CLEARANCE standoff, collision-checked).
STAGING_TOP_ORIENTATION_RPY = (TOP_TILT_FROM_VERTICAL, 0.0, math.radians(150))
STAGING_TOP_GRASP_POINT = (STAGING_XY[0], STAGING_XY[1], STAGING_TOP_GRASP_Z)
STAGING_TOP_GRASP_EE_XYZ = offset_along_approach(
    STAGING_TOP_GRASP_POINT, STAGING_TOP_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
STAGING_TOP_APPROACH_EE_XYZ = offset_along_approach(
    STAGING_TOP_GRASP_POINT,
    STAGING_TOP_ORIENTATION_RPY,
    GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE,
)

# Side grasp: SIDE_ORIENTATION_RPY approaches along world +X.
STAGING_SIDE_GRASP_POINT = (STAGING_XY[0], STAGING_XY[1], STAGING_TOP_GRASP_Z)
STAGING_SIDE_GRASP_XYZ = offset_along_approach(
    STAGING_SIDE_GRASP_POINT, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH
)
STAGING_SIDE_APPROACH_XYZ = offset_along_approach(
    STAGING_SIDE_GRASP_POINT, SIDE_ORIENTATION_RPY, GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE
)

# --- bowl: pin sim.launch.py's randomised bowl position to this value --
#   ros2 launch rbe595_sim sim.launch.py bowl_x:=0.0 bowl_y:=0.5
# chosen clear of the rack footprint and within random_bowl_position()'s own reach/clearance
# bounds, so it's a position that launch file would also have been willing to pick itself.
BOWL_XY = (0.0, 0.5)
BOWL_RIM_TOP_Z = 0.75 + 0.06  # bowl spawn z + bowl height

# Confirmed reachable directly above the bowl's centre, at yaw=pi (not SIDE_ORIENTATION_RPY's
# yaw=pi/2 -- the arm has to reorient as it carries the bottle from staging to the bowl, which
# is fine since yaw doesn't change which way the gripper closes relative to a held object).
BOWL_OVER_EE_XYZ = (BOWL_XY[0], BOWL_XY[1], 0.80)
BOWL_APPROACH_ORIENTATION_RPY = (math.pi / 2, 0.0, math.pi)

# Tilt ("pour") angle added to the roll above. A roll sweep at BOWL_OVER_EE_XYZ found IK starts
# failing past ~45 degrees of added tilt and 30 degrees is right at that edge (close enough
# that real execution noise can push the resulting joint state out of bounds for the *next*
# planning request) -- so this stays well clear, oscillating within [5, 25] degrees.
BOWL_TILT_ANGLE = math.radians(15)
BOWL_OVER_ORIENTATION_RPY = (
    BOWL_APPROACH_ORIENTATION_RPY[0] + BOWL_TILT_ANGLE,
    0.0,
    BOWL_APPROACH_ORIENTATION_RPY[2],
)

SHAKE_AMPLITUDE = math.radians(10)  # oscillates the tilt between 5 and 25 degrees
SHAKE_CYCLES = 3
