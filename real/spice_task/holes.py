"""Shaker holes found directly: small dark round dots on the lid, seen top-down.

find_dots() works on any grey image in which the lid is a circle of known centre / radius (pixels):
the step-4 picture warped onto the lid plane (lid_holes()), or a saved top-down picture (tests).

  1. black-hat (how much darker a pixel is than its surroundings) inside the lid, away from the
     rim and the centre, and outside the label (exclude mask);
  2. dark spots of a hole's size that are round (not the engraved lines / scratches / letters);
  3. the holes are in one wedge of the lid: the largest group of spots within `sector_deg` of each
     other (seen from the centre) is kept, the direction is their mean.
A result needs `min_dots` in the wedge and most of the dark spots there; else None (the label rule
in labels.holes_direction is then used).
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from spice_task.lid_detector import cell_to_plane, plane_homography


@dataclass
class Holes:
    direction: np.ndarray          # unit vector from the lid centre to the holes (image col, row)
    dots: list = field(default_factory=list)       # (col, row) of the dots in the wedge
    others: list = field(default_factory=list)     # dark round spots outside it
    spread_deg: float = 0.0        # angular spread of the dots in the wedge


def find_dots(gray, centre, radius, exclude=None, min_dots=3, sector_deg=100.0, hole_frac=(0.025, 0.10),
              inner=0.18, outer=0.86, contrast=18.0, dominance=1.5):
    """gray: uint8 image; centre (col, row), radius: the lid in pixels; exclude: uint8 mask (non-zero
    = ignore, e.g. the label). hole_frac: hole radius range as a fraction of the lid radius.
    -> Holes or None."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    k = max(5, int(round(0.22 * radius)) | 1)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    h, w = gray.shape
    rows, cols = np.mgrid[0:h, 0:w]
    dist = np.hypot(cols - centre[0], rows - centre[1])
    ring = (dist >= inner * radius) & (dist <= outer * radius)
    skip = paper_mask(g, dist <= outer * radius)
    if exclude is not None:
        skip |= exclude > 0
    pad = max(3, int(round(0.06 * radius)))
    ring &= cv2.dilate(skip.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) == 0
    vals = bh[ring]
    if vals.size == 0:
        return None
    thr = max(contrast, float(np.percentile(vals, 50)) + 3.0 * float(vals.std()))
    mask = ((bh > thr) & ring).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    a_min, a_max = math.pi * (hole_frac[0] * radius) ** 2, math.pi * (hole_frac[1] * radius) ** 2
    spots, weight = [], []
    for i in range(1, n):
        x, y, bw, bh_, area = stats[i]
        if not a_min * 0.5 <= area <= a_max or max(bw, bh_) > 2.0 * min(bw, bh_) or area < 0.45 * bw * bh_:
            continue
        spots.append(cents[i])
        weight.append(float(bh[lab == i].max()) / thr)      # darker spot = more likely a hole
    if len(spots) < min_dots:
        return None
    spots, weight = np.array(spots), np.array(weight)
    ang = np.arctan2(spots[:, 1] - centre[1], spots[:, 0] - centre[0])
    half = math.radians(sector_deg) / 2

    def best_wedge(use):
        best, score = np.zeros(len(spots), bool), 0.0
        for a0 in ang[use]:                          # the wedge centred on each spot in turn
            inside = use & (np.abs((ang - a0 + math.pi) % (2 * math.pi) - math.pi) <= half)
            if weight[inside].sum() > score:
                best, score = inside, float(weight[inside].sum())
        return best, score
    best, score = best_wedge(np.ones(len(spots), bool))
    _, rival = best_wedge(~best)                     # the strongest other group of dark spots
    if best.sum() < min_dots or score < dominance * rival:
        return None
    v = spots[best] - np.asarray(centre, float)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    mean = u.mean(axis=0)
    if np.linalg.norm(mean) < 0.6:                   # spread all round: not one wedge
        return None
    a = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
    m = math.degrees(math.atan2(mean[1], mean[0]))
    spread = float(np.max(np.abs((a - m + 180) % 360 - 180)))
    return Holes(mean / np.linalg.norm(mean), [tuple(p) for p in spots[best]],
                 [tuple(p) for p in spots[~best]], spread)


def paper_mask(gray, lid):
    """The label paper on the lid (its letters are far darker than the holes): the brightest
    large, compact patch inside `lid` (bool mask), holes in it (the letters) filled."""
    vals = gray[lid]
    if vals.size == 0:
        return np.zeros(gray.shape, bool)
    bright = ((gray >= np.percentile(vals, 85)) & lid).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    out = np.zeros(gray.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 0.04 * lid.sum():
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        (_, _), (w, h), _ = cv2.minAreaRect(cnts[0])
        if stats[i, cv2.CC_STAT_AREA] >= 0.6 * w * h:          # a rectangle, not a glare streak
            hull = cv2.convexHull(cnts[0])
            cv2.fillPoly(comp, [hull], 1)
            out |= comp > 0
    return out


def lid_holes(image, lid_top, lid_radius, T_base_cam, K, D, label_quad=None, res=0.00025, margin=0.004):
    """The holes of the lid whose top centre is lid_top (base_link), from a camera image: the image
    warped onto the lid plane (res m per pixel), then find_dots. -> (Holes with .direction in
    base_link xy, top-down grey picture with the dots drawn) or (None, picture)."""
    half = lid_radius + margin
    x_range = (lid_top[0] - half, lid_top[0] + half)
    y_range = (lid_top[1] - half, lid_top[1] + half)
    size = int(math.ceil(2 * half / res))
    M = plane_homography(T_base_cam, K, lid_top[2]) @ cell_to_plane(x_range, y_range, res)
    und = image
    if D is not None and np.any(np.abs(np.asarray(D, float)) > 1e-9):
        und = cv2.undistort(image, np.asarray(K, float), np.asarray(D, float))
    top = cv2.warpPerspective(und, M, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
    exclude = None
    if label_quad is not None:
        m = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.asarray(label_quad, np.int32)], 255)
        if und is not image:
            m = cv2.undistort(m, np.asarray(K, float), np.asarray(D, float))
        exclude = cv2.warpPerspective(m, M, (size, size), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    c = ((size - 1) / 2.0, (size - 1) / 2.0)        # (col, row) of the lid centre
    holes = find_dots(gray, c, lid_radius / res, exclude)
    pic = draw_holes(gray, c, lid_radius / res, holes, exclude)
    if holes is None:
        return None, pic
    # cell (row, col) -> plane: row along +x, col along +y (lid_detector.cell_to_plane)
    d = np.array([holes.direction[1], holes.direction[0]])
    holes.direction = d / np.linalg.norm(d)
    return holes, pic


def draw_holes(gray, centre, radius, holes, exclude=None):
    pic = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if exclude is not None:
        pic[exclude > 0] = (pic[exclude > 0] * 0.5 + np.array([0, 80, 0])).astype(np.uint8)
    c = tuple(int(round(v)) for v in centre)
    cv2.circle(pic, c, int(round(radius)), (0, 255, 255), 1)
    if holes is not None:
        for p in holes.others:
            cv2.circle(pic, tuple(int(round(v)) for v in p), 6, (0, 0, 255), 2)
        for p in holes.dots:
            cv2.circle(pic, tuple(int(round(v)) for v in p), 6, (0, 255, 0), 2)
        tip = (int(round(centre[0] + 0.8 * radius * holes.direction[0])),
               int(round(centre[1] + 0.8 * radius * holes.direction[1])))
        cv2.arrowedLine(pic, c, tip, (255, 0, 255), 2, tipLength=0.2)
    return pic
