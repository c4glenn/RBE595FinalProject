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
