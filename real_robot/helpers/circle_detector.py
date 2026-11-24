"""Spice bottle and bowl detection with colour thresholding + Hough circles.

No ROS in here, so it runs just as well on a saved image (see __main__ below); the ROS
wrapper is rbe595_manipulation/spice_detector_node.py. Configuration: config/spice_detection.yaml.

Pipeline, per object type (bottle, bowl):
  1. Warp the camera image to a top-down view of the horizontal plane at the object's top
     height (bottle cap / bowl rim), using the camera's pose when the image was taken and its
     intrinsics. In that view
     the round top is a true circle of known pixel radius, wherever it is on the table.
  2. Threshold the warped image in HSV to keep only the object's colour (red jars / blue bowl),
     and fill the mask's holes (e.g. the tag on a cap).
  3. Run cv2.HoughCircles on the masked image, with the radius range pinned to the known
     diameter, and keep only circles that are mostly the object's colour.
  4. Map each circle centre back to world (x, y) and to camera-image pixels.
Bottles are then numbered 0.. in reading order as seen in the camera image.
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


def default_config_path() -> Path:
    """config/spice_detection.yaml: from the installed share dir, else from the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        path = Path(get_package_share_directory("rbe595_manipulation")) / "config"
    except Exception:  # noqa: BLE001 -- not installed / no ament: use the source tree
        path = Path(__file__).resolve().parents[2] / "config"
    return path / "spice_detection.yaml"


def load_config(path=None) -> dict:
    with open(path or default_config_path()) as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------------------------
# Camera model
# --------------------------------------------------------------------------------------------


class CameraModel:
    """Pinhole camera at a known world pose (for the wrist camera: its pose when the image was
    taken, from TF)."""

    def __init__(self, K, world_from_optical, position):
        """K: 3x3 intrinsics. world_from_optical: 3x3 rotation whose columns are the optical
        frame's axes (x right, y down, z forward) in the world. position: optical centre."""
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        # world -> camera optical frame: p_cam = R p_world + t
        self.R = np.asarray(world_from_optical, dtype=float).reshape(3, 3).T
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.t = -self.R @ self.position

    def to_dict(self):
        return {
            "K": self.K.reshape(-1).tolist(),
            "world_from_optical": self.R.T.reshape(-1).tolist(),
            "position": self.position.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["K"], d["world_from_optical"], d["position"])

    def image_from_plane(self, z):
        """Homography: world (x, y, 1) on the plane at height z -> camera image pixels."""
        r1, r2, r3 = self.R[:, 0], self.R[:, 1], self.R[:, 2]
        return self.K @ np.column_stack([r1, r2, r3 * z + self.t])

    def project(self, xyz):
        p = self.K @ (self.R @ np.asarray(xyz, dtype=float) + self.t)
        return p[:2] / p[2]


def height_map(depth, camera):
    """Per pixel: height (z in base_link) of the 3D point it sees; NaN where depth is missing."""
    fx, fy, cx, cy = camera.K[0, 0], camera.K[1, 1], camera.K[0, 2], camera.K[1, 2]
    h, w = depth.shape[:2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    R_wo = camera.R.T
    t_wo = camera.position
    height = R_wo[2, 0] * x + R_wo[2, 1] * y + R_wo[2, 2] * z + t_wo[2]
    height[depth <= 0] = np.nan
    return height


# --------------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------------


@dataclass
class Detection:
    kind: str                 # "bottle", "holder" or "bowl"
    world_xyz: tuple          # circle centre on the object's top plane, world frame [m]
    pixel: tuple              # centre in the camera image (u, v)
    radius_px: float          # radius found by Hough, in the top-down view
    fill: float               # fraction of the circle that is the object's colour
    number: int = -1          # bottle number in image reading order (bottles only)


@dataclass
class DetectionResult:
    bottles: list = field(default_factory=list)
    holder: Detection = None
    bowl: Detection = None
    # Debug images, keyed by object type: top-down colour view and its colour mask
    birdseye: dict = field(default_factory=dict)
    masks: dict = field(default_factory=dict)


class SpiceCircleDetector:
    def __init__(self, config: dict, camera: CameraModel):
        self.cfg = config
        self.camera = camera
        be = config["birdseye"]
        self.res = float(be["resolution"])
        self.x_min, self.x_max = be["x_range"]
        self.y_min, self.y_max = be["y_range"]
        # Top-down image: columns along world +y, rows along world +x.
        self.bev_size = (
            int(round((self.y_max - self.y_min) / self.res)),  # width
            int(round((self.x_max - self.x_min) / self.res)),  # height
        )
        self._bev_from_world = np.array(
            [
                [0, 1 / self.res, -self.y_min / self.res],
                [1 / self.res, 0, -self.x_min / self.res],
                [0, 0, 1],
            ]
        )
        self._warps = {}  # plane height -> image->top-down homography
        self.height = None  # per-pixel height map (height_map()), for objects with mask: height

    def set_camera(self, camera: CameraModel):
        self.camera = camera
        self._warps.clear()

    # --- geometry helpers ---

    def _warp_for(self, z):
        if z not in self._warps:
            H = self._bev_from_world @ np.linalg.inv(self.camera.image_from_plane(z))
            self._warps[z] = H / H[2, 2]
        return self._warps[z]

    def bev_to_world(self, u, v):
        return self.x_min + v * self.res, self.y_min + u * self.res

    # --- per-object detection ---

    def _colour_mask(self, bgr, hsv_ranges):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for r in hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(r["lower"]), np.array(r["upper"]))
        return mask

    def _height_mask(self, H, z, band):
        """Pixels of the top-down view whose measured height is within `band` of `z`. The height
        map is warped exactly like the image; nearest-neighbour keeps real measurements."""
        hbev = cv2.warpPerspective(self.height.astype(np.float32), H, self.bev_size,
                                   flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=float("nan"))
        with np.errstate(invalid="ignore"):          # NaN (no depth) -> not in the band
            return (np.abs(hbev - z) < band).astype(np.uint8) * 255

    @staticmethod
    def _clean(mask, open_kernel=5):
        """Remove speckle (optional: an open erases thin rings, e.g. a cup's or bowl's rim), close
        small gaps, then fill holes (a cap's label, specular highlights, a ring's middle) so each
        object is a solid blob."""
        if open_kernel:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def _detect(self, image, kind, max_count=None):
        obj = self.cfg[kind]
        hough = self.cfg["hough"]
        z = float(obj["top_z"])
        H = self._warp_for(z)
        bev = cv2.warpPerspective(image, H, self.bev_size, flags=cv2.INTER_LINEAR)
        if obj.get("mask", "colour") == "height":
            if self.height is None:
                raise ValueError(f"'{kind}' uses mask: height -- pass the depth (--depth)")
            raw = self._height_mask(H, z, float(obj["height_band"]))
        else:
            raw = self._colour_mask(bev, obj["hsv_ranges"])
        mask = self._clean(raw, int(obj.get("open_kernel", 5)))

        # Hough on the grey image with everything that is not the object's colour blacked out:
        # the colour mask isolates the objects from the table/arm/rack, and the grey levels
        # inside keep the edges between neighbouring objects (e.g. a cap and the body of the
        # jar behind it) that the mask alone would merge.
        grey = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        grey = cv2.bitwise_and(grey, grey, mask=mask)
        k = int(hough["blur_kernel"]) | 1
        grey = cv2.GaussianBlur(grey, (k, k), 0)

        r_px = obj["diameter"] / 2.0 / self.res
        tol = float(hough["radius_tolerance"])
        circles = cv2.HoughCircles(
            grey,
            cv2.HOUGH_GRADIENT,
            dp=float(hough["dp"]),
            # Small on purpose: only merges repeats of the same circle. Overlapping candidates
            # for different objects are resolved below, where the geometry is known.
            minDist=max(1.0, hough["duplicate_dist_factor"] * 2 * r_px),
            param1=float(hough["canny_high"]),
            param2=float(hough["accumulator_threshold"]),
            minRadius=int(math.floor(r_px * (1 - tol))),
            maxRadius=int(math.ceil(r_px * (1 + tol))),
        )

        # Edges of the masked grey image, as Hough sees them (it runs Canny at canny_high).
        edges = cv2.Canny(grey, hough["canny_high"] / 2, hough["canny_high"])
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        candidates = []
        for u, v, r in (circles[0] if circles is not None else []):
            fill = self._fill_ratio(mask, u, v, r_px)
            if fill < obj.get("min_fill", hough["min_fill"]):
                continue
            if not self._has_body_towards_camera(mask, u, v, r_px, kind):
                continue
            support = self._edge_support(edges, u, v, r_px)
            if support < obj.get("min_edge_support", hough["min_edge_support"]):   # per-object override
                continue
            candidates.append((u, v, r, fill, support))

        # Two objects cannot overlap on the table, so of any candidates closer than a diameter
        # only one is real. That is the one whose outline is an edge all the way round: a cap
        # or rim is, while circles Hough also fits along a jar's projected sides (they vote
        # for centres along its axis) only have edges at those two sides, and one fitted to
        # the far end of a body that lines up with the cap in front of it is cut off by that
        # cap. On a tie, the one nearest the camera.
        cam_u, cam_v = self._camera_bev_position()
        candidates.sort(
            key=lambda c: (-round(c[4], 2), math.hypot(c[0] - cam_u, c[1] - cam_v))
        )
        detections = []
        for c in candidates:
            if any(
                math.hypot(c[0] - d[0], c[1] - d[1]) < hough["min_dist_factor"] * 2 * r_px
                for d in detections
            ):
                continue
            detections.append(c)
        if max_count:
            # Strongest (fullest) circles first
            detections = sorted(detections, key=lambda c: -c[3])[:max_count]

        out = []
        for u, v, r, fill, _ in detections:
            wx, wy = self.bev_to_world(u, v)
            px = self.camera.project((wx, wy, z))
            out.append(
                Detection(
                    kind=kind,
                    world_xyz=(float(wx), float(wy), z),
                    pixel=(float(px[0]), float(px[1])),
                    radius_px=float(r),
                    fill=float(fill),
                )
            )
        return out, bev, mask

    def _camera_bev_position(self):
        """The camera's (x, y), in top-down-view pixels (it may lie outside the view)."""
        cx, cy, _ = self.camera.position
        p = self._bev_from_world @ np.array([cx, cy, 1.0])
        return p[0], p[1]

    def _has_body_towards_camera(self, mask, u, v, r_px, kind):
        """In the top-down view at the top plane, the sides of an upright cylinder are
        projected out from its top circle towards the camera. So a real top circle has more of
        the object's colour just beyond it on the camera side, while a circle fitted to the far
        end of that projected body (also round, and the same size) has none.

        The sides stick out by (horizontal distance to the camera) * height / (camera height
        above the object's base); for an object nearly below the camera that is too little to
        tell, and the circle is kept."""
        cam_u, cam_v = self._camera_bev_position()
        d = math.hypot(cam_u - u, cam_v - v)
        obj = self.cfg[kind]
        above_base = self.camera.position[2] - (obj["top_z"] - obj["height"])
        offset_px = d * obj["height"] / above_base if above_base > 0 else 0.0
        if offset_px < self.cfg["hough"]["body_check_min_offset"] * r_px:
            return True
        # Just past the top circle's edge, but inside the projected sides.
        k = (r_px + min(offset_px, r_px) / 2) / d
        pu, pv = u + (cam_u - u) * k, v + (cam_v - v) * k
        return self._fill_ratio(mask, pu, pv, r_px / 3.0) >= 0.5

    @staticmethod
    def _edge_support(edges, u, v, r, n=72):
        """Fraction of the circle's outline that lies on an edge."""
        h, w = edges.shape[:2]
        hits = 0
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False):
            x, y = int(round(u + r * math.cos(a))), int(round(v + r * math.sin(a)))
            hits += 0 <= x < w and 0 <= y < h and edges[y, x] > 0
        return hits / n

    @staticmethod
    def _fill_ratio(mask, u, v, r):
        disc = np.zeros_like(mask)
        cv2.circle(disc, (int(round(u)), int(round(v))), int(round(r)), 255, thickness=-1)
        area = cv2.countNonZero(disc)
        return cv2.countNonZero(cv2.bitwise_and(mask, disc)) / area if area else 0.0

    def _number_bottles(self, bottles):
        """Number bottles in reading order as seen in the camera image: rows top to bottom,
        each row left to right."""
        if not bottles:
            return bottles
        # Apparent cap diameter in the image, roughly: use the mean over the detections.
        diam_px = np.mean([self.apparent_diameter_px(b) for b in bottles])
        tol = self.cfg["row_tolerance"] * diam_px
        remaining = sorted(bottles, key=lambda b: b.pixel[1])
        rows = []
        for b in remaining:
            if rows and abs(b.pixel[1] - np.mean([o.pixel[1] for o in rows[-1]])) < tol:
                rows[-1].append(b)
            else:
                rows.append([b])
        ordered = [b for row in rows for b in sorted(row, key=lambda b: b.pixel[0])]
        for i, b in enumerate(ordered):
            b.number = i
        return ordered

    def apparent_diameter_px(self, det):
        """Width of the detected circle as seen in the camera image."""
        x, y, z = det.world_xyz
        r = self.cfg[det.kind]["diameter"] / 2.0
        a = self.camera.project((x, y - r, z))
        b = self.camera.project((x, y + r, z))
        return float(np.hypot(*(a - b)))

    def detect(self, image, height=None, kinds=("bottle", "holder", "bowl")) -> DetectionResult:
        """height: height_map() of the same snapshot; needed for objects with mask: height.
        Only `kinds` are searched for; objects missing from the config are skipped."""
        self.height = height
        result = DetectionResult()
        for kind in kinds:
            if kind not in self.cfg:
                continue
            found, bev, mask = self._detect(image, kind, max_count=None if kind == "bottle" else 1)
            result.birdseye[kind], result.masks[kind] = bev, mask
            if kind == "bottle":
                result.bottles = self._number_bottles(found)
            else:
                setattr(result, kind, found[0] if found else None)
        return result

    # --- drawing ---

    def annotate(self, image, result: DetectionResult, selected=None):
        """The camera image with each detection's outline (the projected circle), bottle
        numbers, and the selected bottle highlighted."""
        out = image.copy()
        dets = list(result.bottles) + [d for d in (result.holder, result.bowl) if d]
        for d in dets:
            is_sel = d.kind == "bottle" and d.number == selected
            colour = ((0, 255, 255) if is_sel else (0, 255, 0) if d.kind == "bottle"
                      else (255, 0, 255) if d.kind == "holder" else (255, 255, 0))
            outline = self._projected_circle(d)
            cv2.polylines(out, [outline], True, colour, 3 if is_sel else 2, cv2.LINE_AA)
            u, v = int(round(d.pixel[0])), int(round(d.pixel[1]))
            cv2.drawMarker(out, (u, v), colour, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
            if d.kind == "bottle":
                self._label(out, str(d.number), (u, v), colour)  # on the cap itself
            else:
                self._label(out, d.kind, (u, v - int(self.apparent_diameter_px(d) * 0.6)), colour)
        return out

    def _projected_circle(self, d, n=48):
        x, y, z = d.world_xyz
        r = self.cfg[d.kind]["diameter"] / 2.0
        pts = [
            self.camera.project((x + r * math.cos(a), y + r * math.sin(a), z))
            for a in np.linspace(0, 2 * math.pi, n, endpoint=False)
        ]
        return np.round(pts).astype(np.int32).reshape(-1, 1, 2)

    @staticmethod
    def _label(img, text, centre, colour):
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        x, y = centre[0] - w // 2, centre[1] + h // 2
        cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + 4), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def format_table(result: DetectionResult) -> str:
    lines = [" #   world x   world y   world z   pixel (u, v)"]
    for b in result.bottles:
        x, y, z = b.world_xyz
        lines.append(
            f"{b.number:2d}  {x:8.3f}  {y:8.3f}  {z:8.3f}   ({b.pixel[0]:.0f}, {b.pixel[1]:.0f})"
        )
    for name in ("holder", "bowl"):
        d = getattr(result, name)
        if d:
            x, y, z = d.world_xyz
            lines.append(f"{name:6s}{x:8.3f}  {y:8.3f}  {z:8.3f}   ({d.pixel[0]:.0f}, {d.pixel[1]:.0f})")
        else:
            lines.append(f"{name}: not found")
    return "\n".join(lines)


def main(argv=None):
    """Run the detector on a saved snapshot (the spice detector node saves snapshot.png and
    the camera's pose and intrinsics as snapshot_camera.yaml next to it):
    python3 -m rbe595_manipulation.vision.circle_detector IMAGE [--camera CAMERA_YAML]
        [--out annotated.png] [--debug DIR]
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("image")
    ap.add_argument(
        "--camera", default=None, help="camera YAML (default: IMAGE's name + _camera.yaml)"
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", default=None,
                    help="depth .npy (metres, undistorted, aligned to the image); needed for mask: height")
    ap.add_argument("--out", default=None, help="write the annotated image here")
    ap.add_argument("--debug", default=None, help="write top-down views and masks to this dir")
    ap.add_argument("--select", type=int, default=None, help="bottle number to highlight")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read {args.image}")
    camera_path = Path(args.camera or Path(args.image).with_name(Path(args.image).stem + "_camera.yaml"))
    with open(camera_path) as f:
        camera = CameraModel.from_dict(yaml.safe_load(f))
    detector = SpiceCircleDetector(cfg, camera)
    height = height_map(np.load(args.depth), camera) if args.depth else None
    result = detector.detect(image, height)
    print(format_table(result))
    if args.out:
        cv2.imwrite(args.out, detector.annotate(image, result, args.select))
    if args.debug:
        d = Path(args.debug)
        d.mkdir(parents=True, exist_ok=True)
        for kind in result.birdseye:
            cv2.imwrite(str(d / f"birdseye_{kind}.png"), result.birdseye[kind])
            cv2.imwrite(str(d / f"mask_{kind}.png"), result.masks[kind])


if __name__ == "__main__":
    main()
