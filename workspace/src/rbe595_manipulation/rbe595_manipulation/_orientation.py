"""Shared roll/pitch/yaw <-> world-frame math, used by poses.py (to project approach/retreat
standoffs along a gripper's own approach axis) and arm_client.py (to build MoveGroup goals).

Composed as R = Rz(yaw) * Ry(pitch) * Rx(roll), matching the convention used throughout the
plan's orientation derivation.
"""

import math
from typing import Sequence, Tuple

from geometry_msgs.msg import Quaternion


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def approach_axis_world(rpy: Sequence[float]) -> Tuple[float, float, float]:
    """The gripper's local +Z axis (approach/reach direction) expressed in world coordinates.

    Used to place approach/retreat/grasp waypoints a given standoff distance apart along the
    direction the gripper actually travels as it reaches for something -- rather than always
    offsetting along world Z, which only makes sense for a purely vertical approach.
    """
    roll, pitch, yaw = rpy
    # Rx(roll) * (0, 0, 1)
    vx, vy, vz = 0.0, -math.sin(roll), math.cos(roll)
    # Ry(pitch) * v
    cp, sp = math.cos(pitch), math.sin(pitch)
    vx, vy, vz = cp * vx + sp * vz, vy, -sp * vx + cp * vz
    # Rz(yaw) * v
    cy, sy = math.cos(yaw), math.sin(yaw)
    vx, vy, vz = cy * vx - sy * vy, sy * vx + cy * vy, vz
    return (vx, vy, vz)


def offset_along_approach(
    point_xyz: Tuple[float, float, float], rpy: Sequence[float], distance: float
) -> Tuple[float, float, float]:
    """A point `distance` back from `point_xyz`, opposite the approach direction.

    i.e. if `point_xyz` is where the gripper/fingertip should end up, the returned point is
    where the end-effector origin should be commanded to in order to reach it (distance =
    GRIPPER_FINGERTIP_REACH), or a further clearance stand-off before/after that (distance =
    GRIPPER_FINGERTIP_REACH + APPROACH_CLEARANCE).
    """
    ax, ay, az = approach_axis_world(rpy)
    px, py, pz = point_xyz
    return (px - distance * ax, py - distance * ay, pz - distance * az)


def _rpy_to_matrix(rpy: Sequence[float]):
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matrix_to_rpy(m) -> Tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -m[2][0])))
    if abs(m[2][0]) > 1.0 - 1e-9:  # gimbal lock: only roll - yaw (or roll + yaw) is defined
        return (math.atan2(-m[1][2], m[1][1]), pitch, 0.0)
    return (math.atan2(m[2][1], m[2][2]), pitch, math.atan2(m[1][0], m[0][0]))


def heading_from(origin_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """World yaw of the horizontal direction from origin_xy to target_xy."""
    return math.atan2(target_xy[1] - origin_xy[1], target_xy[0] - origin_xy[0])


def top_grasp_rpy(yaw: float) -> Tuple[float, float, float]:
    """Gripper pointing straight down. The gen3_lite_2f fingers close along the end-effector
    y axis, which this puts along world (sin yaw, -cos yaw)."""
    return (math.pi, 0.0, yaw)


def side_grasp_rpy(heading: float, tilt: float = 0.0) -> Tuple[float, float, float]:
    """Gripper approaching horizontally along world `heading`, fingers closing horizontally
    (end-effector y axis horizontal), so a standing bottle is held by its sides. The bottle's
    axis then lies along the end-effector x axis.

    `tilt` turns the gripper about its own approach axis, which tips the held bottle over
    sideways about the grasp point without moving the end-effector origin -- that is the pour.
    """
    # R = Rz(heading) * Ry(pi/2) * Rz(tilt): Ry(pi/2) points the approach (+z) along world +x
    # with the fingers along world y; Rz(tilt) then rotates about that approach axis.
    ch, sh = math.cos(heading), math.sin(heading)
    ct, st = math.cos(tilt), math.sin(tilt)
    # Ry(pi/2) * Rz(tilt) = [[0, 0, 1], [st, ct, 0], [-ct, st, 0]]
    a = [[0.0, 0.0, 1.0], [st, ct, 0.0], [-ct, st, 0.0]]
    rz = [[ch, -sh, 0.0], [sh, ch, 0.0], [0.0, 0.0, 1.0]]
    m = [[sum(rz[i][k] * a[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return _matrix_to_rpy(m)
