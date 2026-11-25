"""Find the salt jar's round lid with cv2.HoughCircles. No ROS in here.

Same idea as the main branch's vision/circle_detector.py: a top-down view in which the lid is a
true circle of known pixel radius, so HoughCircles gets a tight radius range, then edge-support
checks and overlap suppression. Two ways to get that top-down view (lid_detection.method):

colour (default) -- like the main branch: the colour image warped onto the horizontal plane at the
  lid-top height (known from the rack tags), Hough on its grey levels. A shiny metal lid often has
  no depth at all on the D405, so this does not need any. The depth is only used, where there is
  some, to throw out circles that are clearly not at lid height (the empty rack pockets, ~8 cm
  lower) and to read the lid's real height. If it finds nothing, the depth way below is tried.

depth -- the lid as a raised disc in the depth:

  1. depth image -> points in base_link (camera pose from TF when the image was taken);
     the gripper's own fingers (in view of the wrist camera) are dropped;
  2. points -> top-down height map, 1 mm cells, highest point per cell;
  3. lid mask = cells within +-band of the lid-top height (given, e.g. from the rack tags, or
     found as the highest surface big enough to be a lid); holes filled (shiny spots, the
     "salt" writing make no difference to the height);
  4. HoughCircles on the mask with radius = lid_diameter / 2 +- tolerance; keep circles that
     are mostly lid (fill) with an edge most of the way round (edge support); of overlapping
     ones keep the best; refine each centre with a least-squares circle fit to the mask edge;
  5. lid top height = median height inside the circle.
"""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Lid:
    center: np.ndarray        # (x, y, z): centre of the lid top in base_link
    radius: float             # fitted radius [m]
    fill: float
    support: float
    pixel: tuple = (0.0, 0.0)  # centre in the camera image (for drawing)
    label: str = None          # spice name read on its paper label (labels.py), None = not read
    label_score: float = None  # that reading's CTC score (0 = perfect)


@dataclass
class LidResult:
    lids: list = field(default_factory=list)
    top_z: float = float('nan')        # lid-top height the mask was cut at
    mask: np.ndarray = None            # top-down lid mask / edges (debug)
    heightmap: np.ndarray = None       # top-down height map (debug)
    birdseye: np.ndarray = None        # top-down colour view (colour method, debug)
    reason: str = ''                   # why nothing was found
    method: str = ''                   # 'colour' or 'depth': which found the lids


def plane_homography(T_base_cam, K, z):
    """3x3: (x, y, 1) on the horizontal plane at height z in base_link -> camera image pixels."""
    T = np.linalg.inv(T_base_cam)
    R, t = T[:3, :3], T[:3, 3]
    return np.asarray(K, float) @ np.column_stack((R[:, 0], R[:, 1], R[:, 2] * z + t))


def cell_to_plane(x_range, y_range, res):
    """3x3: top-down cell (col, row, 1) -> plane (x, y, 1)."""
    return np.array([[0.0, res, x_range[0] + 0.5 * res],
                     [res, 0.0, y_range[0] + 0.5 * res],
                     [0.0, 0.0, 1.0]])


def warp_plane(image, T_base_cam, K, D, z, x_range, y_range, res):
    """Top-down view of the horizontal plane at height z: rows along base +x, columns along +y,
    res metres per pixel, cell (row, col) centred at (x0 + (row + 0.5) res, y0 + (col + 0.5) res)
    -- the same grid as LidDetector.heightmap."""
    if D is not None and np.any(np.abs(np.asarray(D, float)) > 1e-9):
        image = cv2.undistort(image, np.asarray(K, float), np.asarray(D, float))
    rows = int(math.ceil((x_range[1] - x_range[0]) / res))
    cols = int(math.ceil((y_range[1] - y_range[0]) / res))
    M = plane_homography(T_base_cam, K, z) @ cell_to_plane(x_range, y_range, res)
    return cv2.warpPerspective(image, M, (cols, rows), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def view_on_plane(T_base_cam, K, image_size, z, limit=0.4):
    """x/y ranges of what the camera sees of the plane at height z (clipped to +-limit around the
    camera)."""
    w, h = image_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    rays = corners @ np.linalg.inv(np.asarray(K, float)).T @ T_base_cam[:3, :3].T
    o = T_base_cam[:3, 3]
    pts = []
    for d in rays:
        if d[2] < -1e-6:
            pts.append(o + (z - o[2]) / d[2] * d)
    if len(pts) < 4:
        return (o[0] - limit, o[0] + limit), (o[1] - limit, o[1] + limit)
    pts = np.array(pts)
    x = np.clip([pts[:, 0].min(), pts[:, 0].max()], o[0] - limit, o[0] + limit)
    y = np.clip([pts[:, 1].min(), pts[:, 1].max()], o[1] - limit, o[1] + limit)
    return (float(x[0]), float(x[1])), (float(y[0]), float(y[1]))


def cloud_from_depth(depth, K, D, T_base_cam, max_depth=0.6, step=1):
    """Depth image (m; NaN or 0 = no data) -> (N, 3) points in base_link."""
    d = depth[::step, ::step]
    v, u = np.nonzero(np.isfinite(d) & (d > 0.05) & (d < max_depth))
    if len(u) == 0:
        return np.zeros((0, 3))
    z = d[v, u].astype(np.float64)
    px = np.column_stack((u * step, v * step)).astype(np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(px, K, D).reshape(-1, 2)
    cam = np.column_stack((rays * z[:, None], z))
    return cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]


def project(points, T_base_cam, K, D):
    """base_link points (N, 3) -> image pixels (N, 2)."""
    T = np.linalg.inv(T_base_cam)
    p = np.asarray(points, float).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]
    px, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, D)
    return px.reshape(-1, 2)


def fit_circle(xy):
    """Least-squares circle through 2-D points -> (centre, radius)."""
    A = np.column_stack((2 * xy, np.ones(len(xy))))
    c, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
    return c[:2], float(np.sqrt(max(0.0, c[2] + c[:2] @ c[:2])))


class LidDetector:
    def __init__(self, cfg, lid_diameter):
        """cfg: the lid_detection block of config/real.yaml."""
        self.cfg = cfg
        self.res = float(cfg['resolution'])
        self.r = lid_diameter / 2.0
        self.r_px = self.r / self.res

    # ------------------------------------------------------------------ height map
    def heightmap(self, points, x_range, y_range):
        """Highest point per cell. Rows along base +x, columns along base +y. NaN = no data."""
        nx = int(math.ceil((x_range[1] - x_range[0]) / self.res))
        ny = int(math.ceil((y_range[1] - y_range[0]) / self.res))
        hm = np.full(nx * ny, -np.inf)
        i = ((points[:, 0] - x_range[0]) / self.res).astype(int)
        j = ((points[:, 1] - y_range[0]) / self.res).astype(int)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        np.maximum.at(hm, i[ok] * ny + j[ok], points[ok, 2])
        hm = hm.reshape(nx, ny)
        hm[~np.isfinite(hm)] = np.nan
        return hm

    def cell_to_xy(self, row, col, x_range, y_range):
        return x_range[0] + (row + 0.5) * self.res, y_range[0] + (col + 0.5) * self.res

    def auto_top_z(self, hm, band):
        """Highest height with at least half a lid's worth of cells within +-band of it."""
        h = hm[np.isfinite(hm)]
        if len(h) == 0:
            return None
        need = 0.5 * math.pi * self.r_px ** 2
        hs = np.sort(h)[::-1]
        # walk down from the top in 2 mm steps
        for z in np.arange(hs[0], hs[-1], -0.002):
            if np.count_nonzero(np.abs(h - z) < band) >= need:
                # centre the band on that surface
                return float(np.median(h[np.abs(h - z) < band]))
        return None

    # ------------------------------------------------------------------ detection
    def detect(self, depth, K, D, T_base_cam, T_base_ee=None, top_z=None, around=None, region=None, image=None):
        """Lids in one snapshot: the colour image (BGR) and the (median-filtered) depth image.

        top_z:  expected lid-top height (base_link z); None = the highest lid-sized surface in the
                depth.
        around: (x, y) to search around (search_radius); else `region` ((x0, x1), (y0, y1));
                else everything the camera sees.
        Returns a LidResult, lids sorted best first."""
        method = self.cfg.get('method', 'colour')
        why = ''
        if method == 'colour' and image is not None:
            z = top_z
            if z is None and depth is not None:
                pts = self._points(depth, K, D, T_base_cam, T_base_ee)
                if len(pts):
                    xr, yr = (pts[:, 0].min(), pts[:, 0].max()), (pts[:, 1].min(), pts[:, 1].max())
                    z = self.auto_top_z(self.heightmap(pts, xr, yr), float(self.cfg['band']))
            if z is not None:
                res = self.detect_colour(image, depth, K, D, T_base_cam, z, T_base_ee, around, region)
                if res.lids:
                    return res
                why = f'colour: {res.reason}; '
            else:
                why = 'colour: no lid height (no rack tags, no depth surface); '
        res = self.detect_depth(depth, K, D, T_base_cam, T_base_ee, top_z, around, region)
        if why and not res.lids:
            res.reason = why + 'depth: ' + res.reason
        return res

    def detect_cup(self, image, depth, K, D, T_base_cam, rim_z, T_base_ee=None, around=None):
        """A stand's cup seen from above (the raised ring of its column top): the depth first (matte
        plastic has good depth; the colour outline merges with the column's side), then colour."""
        res = self.detect_depth(depth, K, D, T_base_cam, T_base_ee, rim_z, around)
        if res.lids:
            return res
        col = self.detect_colour(image, depth, K, D, T_base_cam, rim_z, T_base_ee, around)
        if not col.lids:
            col.reason = f'depth: {res.reason}; colour: {col.reason}'
        return col

    def _points(self, depth, K, D, T_base_cam, T_base_ee):
        """Depth -> base_link points, the gripper's own fingers dropped."""
        c = self.cfg
        if depth is None:
            return np.zeros((0, 3))
        pts = cloud_from_depth(depth, K, D, T_base_cam, c.get('max_depth', 0.6))
        if T_base_ee is not None and len(pts):
            T_ee_base = np.linalg.inv(T_base_ee)
            along = pts @ T_ee_base[2, :3] + T_ee_base[2, 3]       # distance along the approach
            pts = pts[along > c['gripper_clear']]
        return pts

    def _ranges(self, around, region, fallback):
        if around is not None:
            m = self.cfg['search_radius'] + self.r + 0.01
            return (around[0] - m, around[0] + m), (around[1] - m, around[1] + m)
        if region is not None:
            return region
        return fallback

    def detect_colour(self, image, depth, K, D, T_base_cam, top_z, T_base_ee=None, around=None, region=None):
        """Hough on the colour image warped onto the plane at the lid-top height (see the top)."""
        c = self.cfg
        h, w = image.shape[:2]
        x_range, y_range = self._ranges(around, region, view_on_plane(T_base_cam, K, (w, h), top_z))
        bev = warp_plane(image, T_base_cam, K, D, top_z, x_range, y_range, self.res)
        grey = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY) if bev.ndim == 3 else bev
        kb = int(c['blur_kernel']) | 1
        grey = cv2.GaussianBlur(grey, (kb, kb), 0)
        tol = float(c['radius_tolerance'])
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(1.0, 0.6 * self.r_px),
                                   param1=float(c['canny_high']),
                                   param2=float(c.get('colour_accumulator_threshold', 20)),
                                   minRadius=int(math.floor(self.r_px * (1 - tol))),
                                   maxRadius=int(math.ceil(self.r_px * (1 + tol))))
        edges = cv2.Canny(grey, float(c['canny_high']) / 2, float(c['canny_high']))
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        visible = cv2.warpPerspective(np.full((h, w), 255, np.uint8),
                                      plane_homography(T_base_cam, K, top_z) @ cell_to_plane(x_range, y_range, self.res),
                                      (grey.shape[1], grey.shape[0]),
                                      flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        pts = self._points(depth, K, D, T_base_cam, T_base_ee)
        hm = self.heightmap(pts, x_range, y_range) if len(pts) else None
        result = LidResult(top_z=float(top_z), mask=edges, heightmap=hm, birdseye=bev, method='colour')

        reject = float(c.get('depth_reject', 0.03))
        candidates, dropped = [], {'edge': 0, 'depth': 0, 'edge of view': 0}
        for col, row, r_found in (circles[0] if circles is not None else []):
            if self._fill(visible, col, row, self.r_px) < 0.95:
                dropped['edge of view'] += 1
                continue
            support = self._edge_support(edges, col, row, r_found)   # at the radius Hough found
            if support < c.get('colour_min_edge_support', 0.6):
                dropped['edge'] += 1
                continue
            z, at_height = self._depth_inside(hm, col, row, top_z)
            if z is not None and abs(z - top_z) > reject:
                dropped['depth'] += 1                     # e.g. an empty pocket, ~8 cm lower
                continue
            candidates.append((col, row, at_height, support, z))
        candidates.sort(key=lambda k: -round(k[3], 2))
        kept = []
        for cand in candidates:                           # two lids cannot overlap
            if all(math.hypot(cand[0] - k[0], cand[1] - k[1]) >= 1.8 * self.r_px for k in kept):
                kept.append(cand)
        ys, xs = np.nonzero(cv2.Canny(grey, float(c['canny_high']) / 2, float(c['canny_high'])))
        edge_pts = np.column_stack((xs, ys)).astype(np.float64) if len(xs) else None
        for col, row, at_height, support, z in kept:
            col, row, r_px = self._refine(edge_pts, col, row)
            x, y = self.cell_to_xy(row, col, x_range, y_range)
            centre = np.array([x, y, z if z is not None else float(top_z)])
            px = project(centre, T_base_cam, K, D)[0]
            result.lids.append(Lid(centre, r_px * self.res, float(at_height), float(support),
                                   (float(px[0]), float(px[1]))))
        if not result.lids:
            n = 0 if circles is None else len(circles[0])
            why = ', '.join(f'{v} at the {k}' if k == 'edge of view' else f'{v} by {k}'
                            for k, v in dropped.items() if v)
            result.reason = (f'no lid-sized circle in the colour image on the plane z {top_z:+.3f} '
                             f'({n} Hough candidates' + (f'; dropped {why}' if why else '') + ')')
        return result

    def _depth_inside(self, hm, col, row, top_z):
        """(median height inside the circle, fraction of it with depth near top_z); the height is
        None when too little of the circle has depth (a shiny lid)."""
        if hm is None:
            return None, 0.0
        rr, cc = np.ogrid[:hm.shape[0], :hm.shape[1]]
        inside = (rr - row) ** 2 + (cc - col) ** 2 < (0.7 * self.r_px) ** 2
        h = hm[inside]
        n = max(1, np.count_nonzero(inside))
        h = h[np.isfinite(h)]
        near = np.abs(h - top_z) < float(self.cfg['band'])
        at_height = np.count_nonzero(near) / n
        if len(h) < 0.25 * n:
            return None, at_height
        if np.count_nonzero(near) > 0.5 * len(h):
            return float(np.median(h[near])), at_height
        return float(np.median(h)), at_height

    def detect_depth(self, depth, K, D, T_base_cam, T_base_ee=None, top_z=None, around=None, region=None):
        """Hough on the lid-height mask of the depth (see the top)."""
        c = self.cfg
        pts = self._points(depth, K, D, T_base_cam, T_base_ee)
        if len(pts) == 0:
            return LidResult(reason='no depth')
        x_range, y_range = self._ranges(around, region, ((float(pts[:, 0].min()), float(pts[:, 0].max())),
                                                         (float(pts[:, 1].min()), float(pts[:, 1].max()))))
        hm = self.heightmap(pts, x_range, y_range)
        band = float(c['band'])
        if top_z is None:
            top_z = self.auto_top_z(hm, band)
            if top_z is None:
                return LidResult(heightmap=hm, reason='no surface big enough to be a lid')
        mask = (np.abs(np.nan_to_num(hm, nan=-1e9) - top_z) < band).astype(np.uint8) * 255
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        result = LidResult(top_z=float(top_z), mask=filled, heightmap=hm, method='depth')

        kb = int(c['blur_kernel']) | 1
        grey = cv2.GaussianBlur(filled, (kb, kb), 0)
        tol = float(c['radius_tolerance'])
        circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, dp=1.0, minDist=max(1.0, 0.6 * self.r_px),
                                   param1=float(c['canny_high']), param2=float(c['accumulator_threshold']),
                                   minRadius=int(math.floor(self.r_px * (1 - tol))),
                                   maxRadius=int(math.ceil(self.r_px * (1 + tol))))
        edges = cv2.Canny(filled, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        candidates = []
        for col, row, r_found in (circles[0] if circles is not None else []):
            fill = self._fill(filled, col, row, min(r_found, self.r_px))
            support = self._edge_support(edges, col, row, r_found)
            if fill >= c['min_fill'] and support >= c['min_edge_support']:
                candidates.append((col, row, fill, support))
        candidates.sort(key=lambda k: (-round(k[3], 2), -k[2]))
        kept = []
        for cand in candidates:                       # two lids cannot overlap
            if all(math.hypot(cand[0] - k[0], cand[1] - k[1]) >= 1.8 * self.r_px for k in kept):
                kept.append(cand)

        edge_pts = np.vstack([cnt.reshape(-1, 2) for cnt in contours]).astype(np.float64) if contours else None
        for col, row, fill, support in kept:
            col, row, r_px = self._refine(edge_pts, col, row)
            x, y = self.cell_to_xy(row, col, x_range, y_range)
            rr, cc = np.ogrid[:hm.shape[0], :hm.shape[1]]
            inside = (rr - row) ** 2 + (cc - col) ** 2 < (0.6 * r_px) ** 2
            h = hm[inside]
            h = h[np.isfinite(h) & (np.abs(h - top_z) < band)]
            z = float(np.median(h)) if len(h) > 10 else float(top_z)
            centre = np.array([x, y, z])
            px = project(centre, T_base_cam, K, D)[0]
            result.lids.append(Lid(centre, r_px * self.res, float(fill), float(support), (float(px[0]), float(px[1]))))
        if not result.lids:
            result.reason = (f'no lid-sized circle at height {top_z:+.3f} '
                             f'({0 if circles is None else len(circles[0])} Hough candidates)')
        return result

    def _refine(self, edge_pts, col, row):
        """Least-squares circle through the mask outline near the Hough circle (sub-cell centre)."""
        if edge_pts is None:
            return col, row, self.r_px
        d = np.hypot(edge_pts[:, 0] - col, edge_pts[:, 1] - row)
        near = edge_pts[np.abs(d - self.r_px) < 3.0]
        if len(near) < 0.4 * 2 * math.pi * self.r_px:
            return col, row, self.r_px
        c2, r_fit = fit_circle(near)
        tol = float(self.cfg['radius_tolerance'])
        if math.hypot(c2[0] - col, c2[1] - row) < 3.0 and abs(r_fit - self.r_px) < tol * self.r_px:
            return float(c2[0]), float(c2[1]), float(r_fit)
        return col, row, self.r_px

    @staticmethod
    def _fill(mask, u, v, r):
        disc = np.zeros_like(mask)
        cv2.circle(disc, (int(round(u)), int(round(v))), int(round(r)), 255, -1)
        area = cv2.countNonZero(disc)
        return cv2.countNonZero(cv2.bitwise_and(mask, disc)) / area if area else 0.0

    @staticmethod
    def _edge_support(edges, u, v, r, n=72):
        h, w = edges.shape
        hits = 0
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            x, y = int(round(u + r * math.cos(a))), int(round(v + r * math.sin(a)))
            hits += 0 <= x < w and 0 <= y < h and edges[y, x] > 0
        return hits / n


def draw_circle(image, centre, radius, T_base_cam, K, D, colour, label=None, thickness=2, label_dy=20,
                below=False):
    """A horizontal circle in base_link drawn on the camera image (in place).
    below=True: the label (lines split on '\\n') centred under the circle (above it if there is no
    room), so captions of circles side by side do not run into each other."""
    a = np.linspace(0, 2 * np.pi, 60)
    ring = np.column_stack((centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a),
                            np.full_like(a, centre[2])))
    px = project(ring, T_base_cam, K, D).astype(np.int32)
    cv2.polylines(image, [px], True, colour, thickness, cv2.LINE_AA)
    c = project(centre, T_base_cam, K, D)[0].astype(int)
    cv2.drawMarker(image, tuple(int(v) for v in c), colour, cv2.MARKER_CROSS, 14, 2)
    if label and below:
        font, scale, lh = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 20
        lines = label.split('\n')
        y = int(px[:, 1].max()) + lh
        if y + lh * (len(lines) - 1) > image.shape[0] - 4:          # no room under it: above
            y = int(px[:, 1].min()) - 8 - lh * (len(lines) - 1)
        for k, line in enumerate(lines):
            w = cv2.getTextSize(line, font, scale, 2)[0][0]
            org = (int(np.clip(c[0] - w // 2, 2, image.shape[1] - w - 2)), y + k * lh)
            cv2.putText(image, line, org, font, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, line, org, font, scale, colour, 2, cv2.LINE_AA)
    elif label:
        org = (int(c[0]) + 12, int(c[1]) + label_dy)
        cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(image, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return image
