"""Offline tests for spice_task/holes.py (the shaker holes found as dark dots on the lid).

Real lids: test/data/lid_*.png, crops of step-4 pictures of jars in the stand (the drawn circle and
marks painted out), with the lid centre / radius and the true hole direction in lids.json.
Synthetic: a lid rendered through a camera, to check lid_holes' warp and base_link direction.

  python3 real/test/test_holes.py          (or: python3 -m pytest real/test)
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))

from helpers.spice_ocr import SpiceReader                      # noqa: E402
from spice_task.holes import find_dots, lid_holes              # noqa: E402
from spice_task.lid_detector import project                    # noqa: E402

DATA = os.path.join(HERE, 'data')
LIDS = json.load(open(os.path.join(DATA, 'lids.json')))


def angle_off(v, deg):
    return abs((math.degrees(math.atan2(v[1], v[0])) - deg + 180) % 360 - 180)


@pytest.fixture(scope='module')
def reader():
    return SpiceReader()


@pytest.mark.parametrize('name', ['salt_a', 'salt_b', 'paprika_a', 'paprika_b'])
def test_real_lids(name, reader):
    """Clear lids: found, and pointing at the holes (within 35 deg: the grasp accepts 45)."""
    meta = LIDS[name]
    image = cv2.imread(os.path.join(DATA, f'lid_{name}.png'))
    exclude = np.zeros(image.shape[:2], np.uint8)
    for label in reader.read(image):
        if label.name:
            cv2.fillPoly(exclude, [label.quad.astype(np.int32)], 255)
    holes = find_dots(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), meta['centre'], meta['radius'], exclude)
    assert holes is not None, name
    assert len(holes.dots) >= 3
    assert angle_off(holes.direction, meta['holes_deg']) < 35, (name, holes.direction)


def test_faint_lid_never_wrong(reader):
    """A lid whose holes hardly show: not found is fine (the text rule is used), a wrong way is not."""
    meta = LIDS['salt_faint']
    image = cv2.imread(os.path.join(DATA, 'lid_salt_faint.png'))
    holes = find_dots(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), meta['centre'], meta['radius'])
    assert holes is None or angle_off(holes.direction, meta['holes_deg']) < 35


def test_plain_lid_has_no_holes():
    """A plain shiny lid with a few scratches: no direction."""
    rng = np.random.default_rng(0)
    gray = np.full((240, 240), 60, np.uint8)
    cv2.circle(gray, (120, 120), 100, 170, -1)
    for _ in range(6):                                     # thin scratches
        p = rng.integers(40, 200, 4)
        cv2.line(gray, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 120, 1)
    assert find_dots(gray, (120, 120), 100) is None


def test_lid_holes_direction_in_base_link():
    """A lid with a label and holes rendered by a camera looking down (turned 30 deg about its
    axis, so image and base axes differ): lid_holes finds the holes' direction in base_link."""
    K = np.array([[650.0, 0, 640], [0, 650.0, 360], [0, 0, 1]])
    lid = np.array([0.40, 0.05, 0.08])
    a = math.radians(30)
    R = np.array([[math.cos(a), math.sin(a), 0], [math.sin(a), -math.cos(a), 0], [0, 0, -1]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, lid + [0.01, -0.005, 0.18]
    for holes_deg in (0, 100, -135):
        img = np.full((720, 1280, 3), 70, np.uint8)
        ring = [lid + [0.025 * math.cos(t), 0.025 * math.sin(t), 0] for t in np.linspace(0, 2 * math.pi, 90)]
        cv2.fillPoly(img, [project(np.array(ring), T, K, None).astype(np.int32)], (175, 175, 175))
        h = math.radians(holes_deg)
        u, w = np.array([math.cos(h), math.sin(h), 0]), np.array([-math.sin(h), math.cos(h), 0])
        for r, s in ((0.010, 0), (0.014, -0.004), (0.014, 0.004), (0.018, -0.007), (0.018, 0), (0.018, 0.007)):
            c = project(lid + r * u + s * w, T, K, None)[0]
            cv2.circle(img, (int(round(c[0])), int(round(c[1]))), 4, (25, 25, 25), -1)
        label = [lid - 0.016 * u + x * w + y * u for x, y in ((-0.012, -0.005), (0.012, -0.005),
                                                               (0.012, 0.005), (-0.012, 0.005))]
        quad = project(np.array(label), T, K, None)
        cv2.fillPoly(img, [quad.astype(np.int32)], (245, 245, 245))
        holes, pic = lid_holes(img, lid, 0.025, T, K, None, quad)
        assert holes is not None, holes_deg
        assert angle_off(holes.direction, holes_deg) < 10, (holes_deg, holes.direction)
        assert pic.shape[0] == pic.shape[1]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
