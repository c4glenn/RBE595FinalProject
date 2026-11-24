"""AprilTag (tag36h11) detection and pose, in base_link. No ROS in here.

A tag's frame: x to the right of the tag image, y up the tag image, z out of its face (what
cv2.solvePnP's SOLVEPNP_IPPE_SQUARE returns for the corner order below).

Position: either PnP with the printed size, or (use_depth) the aligned depth along the ray to the
tag's centre, which does not depend on the printed size. The rotation always comes from the
corners (PnP's rotation does not depend on the size either).
"""
from dataclasses import dataclass

import cv2
import numpy as np

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
if hasattr(cv2.aruco, 'ArucoDetector'):                     # OpenCV >= 4.7
    _params = cv2.aruco.DetectorParameters()
else:                                                       # OpenCV 4.6 (Ubuntu 24.04)
    _params = cv2.aruco.DetectorParameters_create()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# A tag with a thin white border on a dark printed part: the border's outer edge is found as a
# second square just outside the black one, and the default minimum distance drops the real tag
# as its duplicate. 0 keeps both; the outer one then fails to decode.
_params.minMarkerDistanceRate = 0.0
if hasattr(cv2.aruco, 'ArucoDetector'):
    _detector = cv2.aruco.ArucoDetector(DICTIONARY, _params)

    def _detect(gray):
        return _detector.detectMarkers(gray)
else:
    def _detect(gray):
        return cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=_params)


def tag_corners_3d(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


@dataclass
class TagPose:
    tag_id: int
    T: np.ndarray            # 4x4 base_link <- tag
    corners_px: np.ndarray   # (4, 2) in the image
    frames: int = 1          # how many frames were averaged
    spread_mm: float = 0.0   # largest distance of one frame's position from the mean
    edge_mm: float = float('nan')   # measured black-border edge (flat tags)

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def rotation(self):
        return self.T[:3, :3]

    def tilt_deg(self):
        """How far the tag's face is turned away from facing straight up."""
        return float(np.degrees(np.arccos(np.clip(self.T[2, 2], -1.0, 1.0))))

    def x_horizontal(self):
        """The tag's +x projected on the table plane, unit length."""
        x = np.array([self.T[0, 0], self.T[1, 0], 0.0])
        return x / np.linalg.norm(x)


def detect_corners(image_bgr):
    """{tag_id: (4, 2) corner pixels} of every tag36h11 in the image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    corners, ids, _ = _detect(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(np.float64) for i, c in zip(ids.ravel(), corners)}


def tag_in_camera(corners_px, size, K, D, depth=None):
    """4x4 camera_optical <- tag from one image, or None. With a depth image (m, NaN/0 = none,
    aligned to the colour image) the position comes from the median depth inside the tag along
    the ray to its centre."""
    ok, rvec, tvec = cv2.solvePnP(tag_corners_3d(size), corners_px, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    if depth is not None:
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.round(corners_px).astype(np.int32), 1)
        z = depth[(mask > 0) & np.isfinite(depth) & (depth > 0)]
        if len(z) < 20:
            return T                                 # no depth on the tag (glare): PnP's position
        centre = corners_px.mean(axis=0).reshape(1, 1, 2)
        ray = cv2.undistortPoints(centre, K, D).ravel()
        # median depth is the z of the face along the optical axis, so scale the ray to that z
        T[:3, 3] = np.array([ray[0], ray[1], 1.0]) * float(np.median(z))
    return T


def tag_flat(corners_px, K, D, T_base_cam, z):
    """4x4 base_link <- tag for a tag lying flat (face up) at height z: the corner rays cut with
    that horizontal plane. Unlike PnP's rotation on a small tag (which can flip tens of degrees),
    the yaw this gives cannot be ambiguous. Also returns the mean edge length (a size check)."""
    rays = cv2.undistortPoints(corners_px.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    dirs = np.column_stack((rays, np.ones(4))) @ T_base_cam[:3, :3].T
    o = T_base_cam[:3, 3]
    if np.any(dirs[:, 2] >= 0):
        return None, 0.0
    c = o + ((z - o[2]) / dirs[:, 2])[:, None] * dirs                # corners: TL, TR, BR, BL
    x = (c[1] - c[0]) + (c[2] - c[3])
    x[2] = 0.0
    x /= np.linalg.norm(x)
    zz = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, :3] = np.column_stack((x, np.cross(zz, x), zz))
    T[:3, 3] = c.mean(axis=0)
    edge = float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]))
    return T, edge


def average_poses(Ts):
    """Mean of 4x4 poses (rotation projected back onto SO(3)) and the largest position spread (mm)."""
    T = np.mean(Ts, axis=0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T[:3, :3] = R
    T[3] = [0, 0, 0, 1]
    spread = 1000.0 * max(np.linalg.norm(S[:3, 3] - T[:3, 3]) for S in Ts)
    return T, spread


def measure_tags(frames, sizes, K, D, use_depth=True, flat=True, min_frames=3, size_tolerance=0.2, report=None,
                 same_plane=()):
    """Average tag poses over several still frames.

    frames: list of (image_bgr, depth_m or None, T_base_cam 4x4).
    sizes: {tag_id: size}; only these IDs are measured.
    flat: the tags lie flat, face up (rack, stand): height from depth (or PnP), yaw and position
          from the corners on that plane (tag_flat); a measured edge more than size_tolerance off
          the configured size drops the frame (misread corner, wrong size in the config).
    same_plane: groups of tag IDs on one flat surface (the rack's two tags). A tag whose own depth
          height gives a wrong edge (noisy depth on a small tag) gets a second try at the height of
          a good tag of its group in the same frame.
    Returns {tag_id: TagPose} for tags seen in at least min_frames frames. `report` (a dict), if
    given, gets {tag_id: why it was dropped} for tags detected but not returned.
    """
    per_tag, last_px, edges, rescued = {}, {}, {}, {}
    group_of = {i: set(g) for g in same_plane for i in g}
    for image, depth, T_base_cam in frames:
        found = {}                                     # tag_id: (px, z from depth/PnP, T, edge)
        for tag_id, px in detect_corners(image).items():
            if tag_id not in sizes:
                continue
            T_cam_tag = tag_in_camera(px, sizes[tag_id], K, D, depth if use_depth else None)
            if T_cam_tag is None:
                continue
            T = T_base_cam @ T_cam_tag
            if not flat:
                per_tag.setdefault(tag_id, []).append(T)
                last_px[tag_id] = px
                continue
            z = T[2, 3]
            T, edge = tag_flat(px, K, D, T_base_cam, z)
            if T is not None:
                found[tag_id] = (px, z, T, edge)

        def ok(tag_id, edge):
            return abs(edge - sizes[tag_id]) <= size_tolerance * sizes[tag_id]

        good_z = {i: f[1] for i, f in found.items() if ok(i, f[3])}
        for tag_id, (px, z, T, edge) in found.items():
            if not ok(tag_id, edge):
                for other in group_of.get(tag_id, ()):
                    if other == tag_id or other not in good_z:
                        continue
                    T2, edge2 = tag_flat(px, K, D, T_base_cam, good_z[other])
                    if T2 is not None and ok(tag_id, edge2):
                        T, edge = T2, edge2
                        rescued[tag_id] = rescued.get(tag_id, 0) + 1
                        break
            edges.setdefault(tag_id, []).append(edge)
            if not ok(tag_id, edge):
                continue
            per_tag.setdefault(tag_id, []).append(T)
            last_px[tag_id] = px
    out = {}
    for tag_id, Ts in per_tag.items():
        if len(Ts) < min(min_frames, len(frames)):
            continue
        T, spread = average_poses(Ts)
        out[tag_id] = TagPose(tag_id, T, last_px[tag_id], len(Ts), spread)
        if tag_id in edges:
            out[tag_id].edge_mm = 1000 * float(np.median(edges[tag_id]))
    for tag_id, n in rescued.items():
        if tag_id in out:
            print(f'tag {tag_id}: depth on it was off in {n} frame(s); used the height of the other tag(s) '
                  'on the same surface there')
    for tag_id, e in edges.items():
        if tag_id not in out:
            why = (f'seen, but its edge measures {1000 * float(np.median(e)):.1f} mm, not '
                   f'{1000 * sizes[tag_id]:.1f} mm (measure it, set tags.sizes)')
            if len(per_tag.get(tag_id, [])) < min(min_frames, len(frames)) and len(per_tag.get(tag_id, [])):
                why = f'seen in only {len(per_tag[tag_id])} of {len(frames)} frames'
            print(f'tag {tag_id}: {why}')
            if report is not None:
                report[tag_id] = why
    return out


def draw_tags(image, tags, K, D, T_base_cam, sizes):
    """Outline + axes of each measured tag (in place)."""
    T_cam_base = np.linalg.inv(T_base_cam)
    for tag in tags.values():
        cv2.polylines(image, [np.round(tag.corners_px).astype(np.int32)], True, (0, 255, 0), 2)
        T = T_cam_base @ tag.T
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        cv2.drawFrameAxes(image, K, D, rvec, T[:3, 3], sizes.get(tag.tag_id, 0.03) * 0.75, 2)
        u, v = tag.corners_px.mean(axis=0)
        cv2.putText(image, f'tag {tag.tag_id}', (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4)
        cv2.putText(image, f'tag {tag.tag_id}', (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
    return image
