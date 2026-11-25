"""Every end-effector pose of the task, in base_link, from what the camera measured. numpy only.

Port of the tran branch's rbe595_manipulation/poses.py (sim, world frame, roll/pitch/yaw) to the
real arm (base_link, rotation matrices). A pose is [x, y, z, qx, qy, qz, qw] of end_effector_link,
whose +z is the approach direction and whose y the fingers close along; the finger-pad centre
(tool_frame) is `reach` out along +z. The camera is on end_effector_link's +x side (mount
"outside") or its -x side ("inside"); `camera` arguments are its position in end_effector_link.
"""
import math

import numpy as np

from helpers.geometry import camera_box_corners, quat_to_matrix
from spice_task.rotations import matrix_to_quaternion, rotation_angle_deg

DOWN = np.array([0.0, 0.0, -1.0])
UP = np.array([0.0, 0.0, 1.0])


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def as_pose(position, R):
    return [float(v) for v in position] + [float(v) for v in matrix_to_quaternion(R)]


def top_down_R(closing_xy, R_current=None):
    """Gripper straight down, fingers closing along closing_xy (horizontal). The gripper is the same
    turned 180 deg about the approach: of the two, the one nearer R_current (joint_6 doesn't wind)."""
    y = unit([closing_xy[0], closing_xy[1], 0.0])
    options = [np.column_stack((np.cross(s * y, DOWN), s * y, DOWN)) for s in (1.0, -1.0)]
    if R_current is None:
        return options[0]
    return min(options, key=lambda R: rotation_angle_deg(R_current.T @ R))


def top_down_options(closing_xy, camera=None, at_xy=None, R_current=None):
    """Both top-down rotations for closing_xy (the gripper turned 180 deg about the approach),
    best first: the camera outside (on the side away from the robot base, seen from at_xy); if
    that is a tie (camera sideways), the one nearer R_current (joint_6 does not wind)."""
    y = unit([closing_xy[0], closing_xy[1], 0.0])
    options = [np.column_stack((np.cross(s * y, DOWN), s * y, DOWN)) for s in (1.0, -1.0)]

    def outside(R):
        if camera is None or at_xy is None or np.linalg.norm(at_xy[:2]) < 1e-6:
            return 0.0
        off = (R @ np.asarray(camera, float))[:2]
        return float(off @ unit([at_xy[0], at_xy[1]]) / max(np.linalg.norm(off), 1e-9))

    def nearness(R):
        return -rotation_angle_deg(R_current.T @ R) if R_current is not None else 0.0

    # outside-ness counts in steps of 0.5 (so "sideways either way" ties), then wrist nearness
    return sorted(options, key=lambda R: (-round(2 * outside(R)) / 2, -nearness(R)))


def turn_camera_out(pose, camera):
    """The same hand position with the gripper turned 180 deg about its approach, if that puts
    the camera further from the robot base. Returns (pose, turned?)."""
    R = quat_to_matrix(*pose[3:])
    p = np.asarray(pose[:3], float)
    off = (R @ np.asarray(camera, float))[:2]
    if np.linalg.norm(p[:2]) < 1e-6 or off @ p[:2] >= 0.0:
        return list(pose), False
    flip = np.diag([-1.0, -1.0, 1.0])                  # 180 deg about the approach (z)
    return as_pose(p, R @ flip), True


SIDE_HEADINGS_DEG = (0, 30, -30, 60, -60, 90, -90)      # tried in this order (0 = from the base)


def side_R(heading, camera=None):
    """Gripper approaching horizontally along `heading` (base_link yaw), fingers closing
    horizontally, the camera's side (end_effector_link +x, or -x for an "inside" mount) up --
    never under the gripper, where it would hit the stand: tran's side_grasp_rpy(heading)."""
    a = np.array([math.cos(heading), math.sin(heading), 0.0])
    x = UP if camera is None or camera[0] >= 0.0 else DOWN
    return np.column_stack((x, np.cross(a, x), a))


def holes_tip(R_grasp, holes_xy):
    """For a side grasp with orientation R_grasp (side_R) of a jar whose shaker holes point along
    holes_xy (base_link, horizontal, from the lid centre): -> (align, way).
    Tipping turns the hand about its approach axis, and the lid's lowest edge is then along the
    finger axis (end_effector_link y) on the side it tips towards: tipped by way * angle (turned()),
    the edge at way * x_z * y goes down. align = |cos| between the holes and the finger axis
    (1: holes straight down when tipped, 0: holes on the palm / far side, they stay level);
    way = +1 / -1: the tip direction that puts them down."""
    h = np.array([holes_xy[0], holes_xy[1], 0.0])
    h /= np.linalg.norm(h)
    v_y = float(R_grasp[:, 1] @ h)
    x_z = float(R_grasp[2, 0])                          # +1 camera side up, -1 "inside" mount down
    return abs(v_y), (1 if v_y * x_z >= 0 else -1)


def turned(pose, angle):
    """The same hand position turned `angle` (rad) about its own approach axis (end_effector_link
    z, which is joint_6's axis): on a side grasp this tips the held jar over sideways."""
    R = quat_to_matrix(*pose[3:])
    c, s = math.cos(angle), math.sin(angle)
    return as_pose(pose[:3], R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]))


POUR_HEADINGS_DEG = SIDE_HEADINGS_DEG + (120, -120, 180)


def lowest_when_tipped(R, corners, max_tilt):
    """Lowest z (relative to end_effector_link) of `corners` (end_effector_link points, e.g. the
    camera box) while the hand at orientation R turns about its approach by up to +-max_tilt."""
    low = 0.0
    for a in np.linspace(-max_tilt, max_tilt, 73):
        c, s = math.cos(a), math.sin(a)
        Rt = R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        low = min(low, float((corners @ Rt.T)[:, 2].min()))
    return low


def pour_poses(bowl_centre, cfg, camera=None, heading=None):
    """Side-held jar over the bowl (tran poses.BOWL_OVER_EE_XYZ): its grasp point straight above
    the rim centre, pour.above_rim higher; hand side-on along `heading` (default: from the base).
    Tipped past 90 deg the wrist camera swings below the hand: the point is raised further if
    needed so the camera box stays pour.camera_clear above the rim through the whole tip + shake.
      over    upright, end_effector_link side_reach back from the grasp point;
      point   the grasp point (where the jar turns about);
      raised  how much higher than above_rim that made it (m)."""
    pr = cfg['pour']
    reach = cfg['gripper'].get('side_reach', cfg['gripper']['reach'])
    c = np.asarray(bowl_centre, float)
    R = side_R(heading_to(c) if heading is None else heading, camera)
    z = c[2] + pr['above_rim']
    raised = 0.0
    if cfg.get('hand_eye'):
        low = lowest_when_tipped(R, camera_box_corners(cfg['hand_eye']),
                                 math.radians(pr['tilt_deg'] + pr['shake_deg']))
        need = c[2] + pr.get('camera_clear', 0.02) - low          # end_effector_link (= grasp point) z
        raised = max(0.0, need - z)
        z += raised
    point = np.array([c[0], c[1], z])
    return {'R': R, 'point': point, 'over': as_pose(ee_at(point, R, reach), R), 'raised': raised}


def ee_at(point, R, distance):
    """end_effector_link position `distance` back along the approach from `point`."""
    return np.asarray(point, float) - distance * R[:, 2]


def heading_to(xy):
    """Yaw of the horizontal direction from the arm base to xy."""
    return math.atan2(xy[1], xy[0])


# ------------------------------------------------------------------ rack (tags 6, 7)
def rack_pose(tags, cfg):
    """(centre of the rack top, rack +x axis (horizontal unit), rack top z) from its two datum
    tags, or None unless both were seen. Tag rack_ids[0] is on the rack's +y side, rack_ids[1] on
    its -y side, both on the front ledge; rack +x points away from that ledge."""
    ids = cfg['tags']['rack_ids']
    if not all(i in tags for i in ids):
        return None
    plus_y, minus_y = (tags[i].position for i in ids)
    y_axis = unit([*(plus_y - minus_y)[:2], 0.0])
    x_axis = np.array([y_axis[1], -y_axis[0], 0.0])
    top_z = (plus_y[2] + minus_y[2]) / 2 + cfg['rack']['tag_recess']
    centre = (plus_y + minus_y) / 2 - cfg['rack']['tag_x_in_rack'] * x_axis
    centre[2] = top_z
    return centre, x_axis, float(top_z)


BACKWARDS_HINT = ('the rack comes out backwards (its +x points at the robot): hand_eye in config/real.yaml '
                  'does not match where the camera is on the wrist (outside / inside), or tags 6 and 7 are swapped')


def stand_backwards(tag, cfg):
    """True if the stand's pocket, from its tag, comes out on the robot's side of the tag: the
    tag faces the arm (the column is beyond it), so this means the camera is mirrored."""
    d = stand_seat(tag, cfg)[:2] - tag.position[:2]
    return float(np.dot(d, tag.position[:2])) < 0.0


def rack_backwards(rack):
    """True if the measured rack's +x (away from its tag ledge) points back at the robot: the rack
    stands in front of the arm with its tags towards it, so this means the camera is mirrored."""
    centre, x_axis, _ = rack
    return float(np.dot(x_axis[:2], centre[:2])) < 0.0


def rack_region(rack, cfg, margin=0.01):
    """((x0, x1), (y0, y1)) in base_link around the rack's footprint: where to look for lids, so
    the stand's empty cup (or anything else round) is not taken for one."""
    centre, x_axis, _ = rack
    y_axis = np.cross(UP, x_axis)
    sx, sy = cfg['rack']['size'][0] / 2, cfg['rack']['size'][1] / 2
    corners = np.array([centre + a * sx * x_axis + b * sy * y_axis for a in (-1, 1) for b in (-1, 1)])
    return ((float(corners[:, 0].min() - margin), float(corners[:, 0].max() + margin)),
            (float(corners[:, 1].min() - margin), float(corners[:, 1].max() + margin)))


def lid_top_in_rack(rack_top_z, cfg):
    """Expected lid-top height of a jar standing in a rack pocket."""
    return rack_top_z - cfg['rack']['pocket_depth'] + cfg['jar']['height']


# ------------------------------------------------------------------ stand (tag 8)
def stand_seat(tag, cfg):
    """Centre of the stand's pocket floor (base_link) from its datum tag (lying flat: yaw only)."""
    s = cfg['stand']
    xh = tag.x_horizontal()
    yh = np.cross(UP, xh)
    a = math.radians(s['pocket_angle_deg'])
    return tag.position + s['tag_to_pocket'] * (math.cos(a) * xh + math.sin(a) * yh) + [0.0, 0.0,
                                                                                          s['seat_above_tag']]


def stand_offset(tag, pocket_xy):
    """The inverse of stand_seat: (tag_to_pocket, pocket_angle_deg) that put the pocket at
    pocket_xy (a measured cup centre) for this tag."""
    xh = tag.x_horizontal()
    yh = np.cross(UP, xh)
    d = np.asarray(pocket_xy, float)[:2] - tag.position[:2]
    return float(np.linalg.norm(d)), math.degrees(math.atan2(d @ yh[:2], d @ xh[:2]))


def seated_top(seat, cfg):
    """Where the lid top of a jar standing in the stand's pocket is."""
    return np.asarray(seat, float) + [0.0, 0.0, cfg['jar']['height']]


# ------------------------------------------------------------------ top grasps
def top_poses(lid_top, R, cfg, clearance, below=None):
    """(approach above it, grasp, place) for a top-down grasp of a jar whose lid top is at lid_top
    (as seated in a pocket). Placing lets go with the jar's bottom at the pocket's MOUTH
    (motion.release_above_mouth above it), not down on the floor: it drops the last 2 cm and
    the 4 mm 45 deg lead-in round the mouth (rack_cad LEAD_IN) centres it. Lowering it all the
    way instead pushes it onto the rim when it is a few mm off, and held by the lid it tips in
    the fingers. below: finger pads this far below the lid top (default jar.top_grasp_below_top);
    placing must use the one the jar is held at."""
    reach = cfg['gripper']['reach']
    below = cfg['jar']['top_grasp_below_top'] if below is None else below
    point = np.asarray(lid_top, float) - [0.0, 0.0, below]
    m = cfg['motion']
    drop = m.get('release_above_mouth')
    up = m['place_drop'] if drop is None else cfg['rack']['pocket_depth'] + drop
    return (as_pose(ee_at(point, R, reach + clearance), R),
            as_pose(ee_at(point, R, reach), R),
            as_pose(ee_at(point, R, reach + up), R))


def rack_top_poses(lid_top, R, cfg, below=None):
    return top_poses(lid_top, R, cfg, cfg['motion']['approach_clearance'], below)


def stand_top_poses(lid_top, R, cfg, below=None):
    return top_poses(lid_top, R, cfg, cfg['stand']['approach_clearance'], below)


# ------------------------------------------------------------------ side grasp (at the stand)
def side_poses(lid_top, cfg, R_top, camera=None, heading=None, clear=None):
    """Side grasp of the jar standing in the stand with its lid top at lid_top (tran
    staging_side_poses), as a dict of poses:
      grasp         fingers round the jar body, approaching from the arm base, level;
      approach      approach_clearance straight back from it;
      lift          approach_clearance straight up from the grasp (out of the pocket);
      place         the grasp pose place_drop up: where the side-held jar is let go;
      retreat_high  above `approach`, level with the top-down approach above the jar: where the
                    gripper turns between top-down and side-on, clear of the jar.
    heading: approach direction (base_link yaw); default straight out from the arm base.
    clear: stand-off along the approach; default motion.approach_clearance."""
    # side_reach: wrist -> jar axis for the side grasp (its own number: the top grasp's `reach` is
    # a height, this one a horizontal stand-off, and the real hand needed them apart)
    reach = cfg['gripper'].get('side_reach', cfg['gripper']['reach'])
    clear = cfg['motion']['approach_clearance'] if clear is None else clear
    R = side_R(heading_to(lid_top) if heading is None else heading, camera)
    point = np.asarray(lid_top, float) - [0.0, 0.0, cfg['jar']['side_grasp_below_top']]
    grasp = ee_at(point, R, reach)
    approach = ee_at(point, R, reach + clear)
    top_approach = np.array(stand_top_poses(lid_top, R_top, cfg)[0][:3])
    return {
        'R': R,
        'grasp': as_pose(grasp, R),
        'approach': as_pose(approach, R),
        'lift': as_pose(grasp + [0.0, 0.0, clear], R),
        'place': as_pose(grasp + [0.0, 0.0, cfg['motion']['place_drop']], R),
        'retreat_high': as_pose([approach[0], approach[1], top_approach[2]], R),
        'retreat_high_top': as_pose([approach[0], approach[1], top_approach[2]], R_top),
    }
