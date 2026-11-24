"""Rotation helpers not in HW7's helpers/geometry.py. numpy only."""
import numpy as np


def matrix_to_quaternion(R):
    """3x3 rotation matrix -> unit quaternion (qx, qy, qz, qw) (Shepperd's method)."""
    R = np.asarray(R, float)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    k = int(np.argmax([x, y, z, w]))
    if k == 3:
        q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
    elif k == 0:
        q = [x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x), (R[2, 1] - R[1, 2]) / (4 * x)]
    elif k == 1:
        q = [(R[0, 1] + R[1, 0]) / (4 * y), y, (R[1, 2] + R[2, 1]) / (4 * y), (R[0, 2] - R[2, 0]) / (4 * y)]
    else:
        q = [(R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), z, (R[1, 0] - R[0, 1]) / (4 * z)]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def rotation_angle_deg(R):
    """Angle of a rotation matrix [deg]."""
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))
