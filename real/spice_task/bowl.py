"""Find the bowl (no tag, anywhere on the table) with cv2.HoughCircles + the depth. No ROS in here.

Same idea as the lid (lid_detector.py): warp the colour image onto a horizontal plane so the bowl's
rim is a true circle, then Hough. What is different:

  * its size is not pinned: any rim diameter in bowl.diameter_range;
  * its rim height is not known: the first warp is at a guess (table + height_guess), the rim
    height is then read from the depth round the circle, and the warp is redone at that height
    (a plane at the wrong height makes the circle the wrong size and off-centre);
  * it must be a BOWL: the rim at least min_height above the table and its inside at least
    min_depth lower than the rim (a flat round thing, a plate, a lid, a printed circle is not);
  * the camera may see only part of it (it is big): a circle with at least min_seen of its
    outline in the picture counts, and step 1 then looks again from right above it.

Returns the best Bowl (centre of the rim circle at rim height, radius, ...) or None + why.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np

from spice_task.lid_detector import cell_to_plane, cloud_from_depth, fit_circle, plane_homography, project, \
    view_on_plane, warp_plane


@dataclass
class Bowl:
    center: np.ndarray        # (x, y, z): rim circle centre, at rim height, base_link
    radius: float             # rim radius [m]
    support: float            # fraction of the visible outline on an edge
    seen: float               # fraction of the outline inside the picture
    inside_drop: float        # rim height - inside height [m] (nan: no depth inside)
    pixel: tuple = (0.0, 0.0)

    @property
    def rim_z(self):
        return float(self.center[2])


def heightmap(points, x_range, y_range, res):
    """Highest point per cell; rows along base +x, columns along +y; NaN = no data."""
    nx = int(math.ceil((x_range[1] - x_range[0]) / res))
    ny = int(math.ceil((y_range[1] - y_range[0]) / res))
    hm = np.full(nx * ny, -np.inf)
    if len(points):
        i = ((points[:, 0] - x_range[0]) / res).astype(int)
        j = ((points[:, 1] - y_range[0]) / res).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        np.maximum.at(hm, i[ok] * ny + j[ok], points[ok, 2])
    hm = hm.reshape(nx, ny)
    hm[~np.isfinite(hm)] = np.nan
    return hm


def _ring(hm, row, col, r_in, r_out):
    rr, cc = np.ogrid[:hm.shape[0], :hm.shape[1]]
    d2 = (rr - row) ** 2 + (cc - col) ** 2
    h = hm[(d2 >= r_in ** 2) & (d2 < r_out ** 2)]
    return h[np.isfinite(h)]


def detect_bowl(image, depth, K, D, T_base_cam, table_z, cfg, T_base_ee=None, gripper_clear=0.165,
                exclude=()):
    """(Bowl or None, why). cfg: the bowl block of config/real.yaml. exclude: [(x, y, radius)]
    discs (the stand, the rack) a bowl's centre cannot be in."""
    res = float(cfg.get('resolution', 0.002))
    rmin, rmax = (float(d) / 2 for d in cfg['diameter_range'])
    h, w = image.shape[:2]
    pts = np.zeros((0, 3))
    if depth is not None:
        pts = cloud_from_depth(depth, K, D, T_base_cam, float(cfg.get('max_depth', 1.0)))
        if T_base_ee is not None and len(pts):                  # the gripper's own fingers
            T_ee_base = np.linalg.inv(T_base_ee)
            pts = pts[pts @ T_ee_base[2, :3] + T_ee_base[2, 3] > gripper_clear]
    z = table_z + float(cfg.get('height_guess', 0.06))
    why = 'no bowl-sized circle'
    best = None
    for _ in range(3):                                        # warp at the guess, then at the rim
        x_range, y_range = view_on_plane(T_base_cam, K, (w, h), z, limit=float(cfg.get('view_limit', 0.6)))
        best, why = _find(image, pts, K, D, T_base_cam, z, table_z, x_range, y_range, res, rmin, rmax, cfg, exclude)
        if best is None or abs(best.rim_z - z) < 0.005:
            break
        z = best.rim_z
    return best, why


def _find(image, pts, K, D, T_base_cam, z, table_z, x_range, y_range, res, rmin, rmax, cfg, exclude):
    h, w = image.shape[:2]
    bev = warp_plane(image, T_base_cam, K, D, z, x_range, y_range, res)
    grey = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if bev.ndim == 3 else bev
    grey = cv2.GaussianBlur(grey, (5, 5), 0)
    canny = float(cfg.get('canny_high', 80))
    rp = (rmin / res, rmax / res)
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=1.0, minDist=rp[0],
                               param1=canny, param2=float(cfg.get('accumulator_threshold', 30)),
                               minRadius=int(math.floor(rp[0])), maxRadius=int(math.ceil(rp[1])))
    if circles is None:
        return None, f'no circle of {200 * rmin:.0f}-{200 * rmax:.0f} cm diameter at z {z:+.3f}'
    edges = cv2.dilate(cv2.Canny(grey, canny / 2, canny), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    visible = cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                                  plane_homography(T_base_cam, K, z) @ cell_to_plane(x_range, y_range, res),
                                  (grey.shape[1], grey.shape[0]), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    hm = heightmap(pts, x_range, y_range, res) if len(pts) else None
    min_h, min_drop = float(cfg.get('min_height', 0.03)), float(cfg.get('min_depth', 0.015))
    dropped = {}
    found = []
    for col, row, r_px in circles[0]:
        x, y = x_range[0] + (row + 0.5) * res, y_range[0] + (col + 0.5) * res
        if any(math.hypot(x - ex, y - ey) < er for ex, ey, er in exclude):
            dropped['on the rack / stand'] = dropped.get('on the rack / stand', 0) + 1
            continue
        a = np.linspace(0, 2 * math.pi, 144, endpoint=False)
        u = np.round(col + r_px * np.cos(a)).astype(int)
        v = np.round(row + r_px * np.sin(a)).astype(int)
        inside = (u >= 0) & (u < grey.shape[1]) & (v >= 0) & (v < grey.shape[0])
        inside[inside] = visible[v[inside], u[inside]] > 0
        seen = float(np.mean(inside))
        if seen < float(cfg.get('min_seen', 0.4)):
            dropped['mostly out of the picture'] = dropped.get('mostly out of the picture', 0) + 1
            continue
        support = float(np.mean(edges[v[inside], u[inside]] > 0))
        if support < float(cfg.get('min_edge_support', 0.5)):
            dropped['weak outline'] = dropped.get('weak outline', 0) + 1
            continue
        rim_z, drop = z, float('nan')
        if hm is not None:
            rim = _ring(hm, row, col, r_px - 0.008 / res, r_px + 0.003 / res)
            mid = _ring(hm, row, col, 0, 0.5 * r_px)
            if len(rim) > 20:
                rim_z = float(np.percentile(rim, 90))
                if rim_z - table_z < min_h:
                    dropped['flat (not raised)'] = dropped.get('flat (not raised)', 0) + 1
                    continue
            if len(mid) > 20 and len(rim) > 20:
                drop = rim_z - float(np.median(mid))
                if drop < min_drop:
                    dropped['not hollow (no lower inside)'] = dropped.get('not hollow (no lower inside)', 0) + 1
                    continue
        found.append((support * seen ** 0.5, col, row, r_px, support, seen, rim_z, drop))
    if not found:
        why = ', '.join(f'{n} {k}' for k, n in dropped.items())
        return None, f'{len(circles[0])} circle(s) at z {z:+.3f}, none a bowl' + (f' ({why})' if why else '')
    _, col, row, r_px, support, seen, rim_z, drop = max(found)
    # sub-cell centre: least-squares circle through the edge pixels near the Hough circle
    ys, xs = np.nonzero(cv2.Canny(grey, canny / 2, canny))
    if len(xs):
        e = np.column_stack((xs, ys)).astype(np.float64)
        near = e[np.abs(np.hypot(e[:, 0] - col, e[:, 1] - row) - r_px) < 3.0]
        if len(near) > 0.3 * 2 * math.pi * r_px * seen:
            c2, r2 = fit_circle(near)
            if math.hypot(c2[0] - col, c2[1] - row) < 5.0 and abs(r2 - r_px) < 0.1 * r_px:
                col, row, r_px = float(c2[0]), float(c2[1]), float(r2)
    x, y = x_range[0] + (row + 0.5) * res, y_range[0] + (col + 0.5) * res
    centre = np.array([x, y, rim_z])
    px = project(centre, T_base_cam, K, D)[0]
    return Bowl(centre, r_px * res, support, seen, drop, (float(px[0]), float(px[1]))), ''
