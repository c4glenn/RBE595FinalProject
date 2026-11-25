"""Offline tests for spice_task/labels.py: labels matched to lids, and the target lid chosen
(label mode and ID mode). No ROS, no model needed.

  python3 real/test/test_labels.py          (or: python3 -m pytest real/test)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helpers.spice_ocr import Label                            # noqa: E402
from spice_task.labels import assign_labels, choose_lid, pixel_on_plane   # noqa: E402
from spice_task.lid_detector import Lid, project               # noqa: E402

K = np.array([[640.0, 0, 640], [0, 640.0, 360], [0, 0, 1]])
D = np.zeros(5)
# camera 0.40 m above the table at (0.40, 0.0), looking straight down (optical z = base -z)
T_BASE_CAM = np.eye(4)
T_BASE_CAM[:3, :3] = np.diag([1.0, -1.0, -1.0])
T_BASE_CAM[:3, 3] = [0.40, 0.0, 0.40]
LID_Z = 0.08


def lid_at(x, y):
    c = np.array([x, y, LID_Z])
    return Lid(c, 0.025, 1.0, 1.0, tuple(project(c, T_BASE_CAM, K, D)[0]))


def label_at(name, x, y, z=LID_Z, score=-0.5):
    u, v = project([x, y, z], T_BASE_CAM, K, D)[0]
    quad = np.float32([[u - 20, v - 8], [u + 20, v - 8], [u + 20, v + 8], [u - 20, v + 8]])
    return Label(name, score, 3.0, name or '', quad, (float(u), float(v)), 0.0)


def test_pixel_on_plane_round_trip():
    p = np.array([0.43, -0.05, LID_Z])
    back = pixel_on_plane(project(p, T_BASE_CAM, K, D)[0], LID_Z, T_BASE_CAM, K, D)
    assert np.allclose(back, p, atol=1e-6)


def test_labels_go_to_nearest_lid():
    lids = [lid_at(0.40, -0.06), lid_at(0.40, 0.0), lid_at(0.40, 0.06)]
    labels = [label_at('pepper', 0.41, -0.06),            # on lid 0, a bit off centre
              label_at('salt', 0.42, 0.06),               # beside lid 2
              label_at('cumin', 0.55, 0.0),               # a paper on the table, 15 cm away
              label_at(None, 0.40, 0.0)]                  # unreadable, on lid 1
    pairs = assign_labels(lids, labels, T_BASE_CAM, K, D, max_dist=0.05)
    assert [lid.label for lid in lids] == ['pepper', None, 'salt']
    assert pairs[2][1] is None and pairs[3][1] is None


def test_label_mode_picks_the_spice():
    lids = [lid_at(0.40, -0.06), lid_at(0.40, 0.06)]
    lids[0].label, lids[1].label = 'pepper', 'salt'
    lid, why = choose_lid(lids, 'salt')
    assert lid is lids[1] and 'salt' in why
    assert choose_lid(lids, 'pepper')[0] is lids[0]
    with pytest.raises(ValueError, match='no lid with a "cumin"'):
        choose_lid(lids, 'cumin')
    lids[0].label = 'salt'
    with pytest.raises(ValueError, match='2 lids read'):
        choose_lid(lids, 'salt')


def test_id_mode_ignores_labels():
    lids = [lid_at(0.40, -0.06), lid_at(0.40, 0.06)]
    lids[0].label, lids[1].label = 'pepper', 'salt'
    assert choose_lid(lids, 'salt', index=0)[0] is lids[0]
    with pytest.raises(ValueError, match='--index 5'):
        choose_lid(lids, 'salt', index=5)


def test_single_lid():
    lid = lid_at(0.40, 0.0)
    chosen, why = choose_lid([lid], 'salt')                # label not read: still that one, warned
    assert chosen is lid and 'WARNING' in why
    lid.label = 'pepper'                                   # read as something else: refuse
    with pytest.raises(ValueError):
        choose_lid([lid], 'salt')



def label_quad_at(x, y, down_xy, z=LID_Z):
    """A label at (x, y) whose text's "down" points along down_xy on the table (reading direction
    = down turned 90 deg so that reading x down follows the image's x-right / y-down rule)."""
    d = np.array([down_xy[0], down_xy[1], 0.0]) / np.linalg.norm(down_xy)
    c = np.array([x, y, z])
    # reading direction r: with "down" d, r = d turned so that (r, d) is right-handed in the IMAGE;
    # found by projecting: pick r of the two perpendiculars that gives a clockwise TL,TR,BR,BL quad
    for r in (np.array([d[1], -d[0], 0.0]), np.array([-d[1], d[0], 0.0])):
        pts = [c - 0.01 * r - 0.004 * d, c + 0.01 * r - 0.004 * d, c + 0.01 * r + 0.004 * d, c - 0.01 * r + 0.004 * d]
        q = project(np.array(pts), T_BASE_CAM, K, D)
        e1, e2 = q[1] - q[0], q[3] - q[0]
        if e1[0] * e2[1] - e1[1] * e2[0] > 0:            # y down: TL->TR then TL->BL turns clockwise
            u, v = project(c, T_BASE_CAM, K, D)[0]
            return Label('salt', -0.3, 3.0, 'salt', q.astype(np.float32), (float(u), float(v)), 0.0)
    raise AssertionError


def test_holes_below_the_text():
    from spice_task.labels import holes_direction
    for ang in (0, 60, 135, -90):
        down = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        lid = np.array([0.40, 0.0, LID_Z])
        label = label_quad_at(*(lid[:2] - 0.012 * down), down)   # label across the lid from the holes
        d, check = holes_direction(label, lid, T_BASE_CAM, K, D)
        assert np.degrees(np.arccos(np.clip(d @ down, -1, 1))) < 2, (ang, d)
        assert check is not None and check < 5, (ang, check)


def test_tipping_puts_the_holes_down():
    from spice_task import task_poses as tp
    for heading in np.radians([0, 40, -70, 150]):
        for camera in (None, np.array([-0.05, 0.0, 0.0])):          # outside / inside mount
            R = tp.side_R(heading, camera)
            for hole_ang in np.radians([0, 30, 90, 200, 300]):
                h = np.array([np.cos(hole_ang), np.sin(hole_ang)])
                align, way = tp.holes_tip(R, h)
                # tip by way * 130 deg (as turned() does) and see where the hole edge of the lid goes
                pose = tp.as_pose(np.zeros(3), R)
                Rt = tp.quat_to_matrix(*tp.turned(pose, way * np.radians(130))[3:])
                edge = Rt @ (R.T @ np.array([h[0], h[1], 0.0]))
                assert np.isclose(edge[2], -align * np.sin(np.radians(130)), atol=1e-9), (heading, hole_ang)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
