"""A tiny ray-cast stand-in for the real cell, to run spice_task offline (no ROS, no robot):
table, rack with tags 6/7, regrasp stand with tag 8, the salt jar (no tag), and a D405-like
camera on end_effector_link (HW7 hand-eye). FakeIO / FakeMover have the interface Task uses.

Not physics: the jar sticks to the fingers when they close around it and drops straight down
onto whatever is under it when they open (into a pocket if it is within the lead-in).
"""
import math

import cv2
import numpy as np

from helpers.geometry import quat_to_matrix
from spice_task.tags import DICTIONARY

TABLE_Z = -0.045
RACK = dict(centre=(0.45, 0.0), yaw=0.0, size=(0.231, 0.209, 0.024), tag_inset=(0.023, 0.028),
            pitch=(0.096, 0.067), col_offset=0.016, mouth=0.059, depth=0.020)
STAND = dict(column=(0.32, 0.26), yaw=math.radians(-90), radius=0.0325, top=0.061, seat=0.041,
             plate=(0.136, 0.100, 0.006), plate_cx=0.018, tag_x=0.058, mouth=0.059)
BOWL = dict(centre=(0.30, -0.27), radius=0.095, height=0.055, wall=0.006, floor=0.008)
JAR = dict(radius=0.0242, lid_radius=0.025, height=0.0813, lid_h=0.014)
TAG = 0.036
W, H, HFOV = 640, 360, math.radians(87)


def marker_texture(tag_id, cells=8):
    """tag36h11 image with a one-cell white margin; the black border edge is `cells` cells."""
    side = cells * 20
    if hasattr(cv2.aruco, 'generateImageMarker'):
        m = cv2.aruco.generateImageMarker(DICTIONARY, tag_id, side, borderBits=1)
    else:
        m = cv2.aruco.drawMarker(DICTIONARY, tag_id, side, borderBits=1)
    return cv2.copyMakeBorder(m, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def rot(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


class World:
    def __init__(self, slot=(1, 0), bowl=None):
        r, s = RACK, STAND
        self.bowl = dict(BOWL, **bowl) if isinstance(bowl, dict) else (dict(BOWL) if bowl else None)
        self.rack_R = rot(r['yaw'])
        self.rack_c = np.array(r['centre'])
        self.rack_top = TABLE_Z + r['size'][2]
        self.slots = []
        for row in range(3):
            for col in range(2):
                p = np.array([(col - 0.5) * r['pitch'][0] + r['col_offset'], (row - 1) * r['pitch'][1]])
                self.slots.append(self.rack_c + self.rack_R @ p)
        lx, ly = r['size'][0] / 2 - r['tag_inset'][0], r['size'][1] / 2 - r['tag_inset'][1]
        # tag frame x/y in base: rack tags read upright from the arm (tag y along rack +x)
        rx = np.append(self.rack_R @ [1, 0], 0)
        ry = np.append(self.rack_R @ [0, 1], 0)
        self.tags = []
        for tag_id, sign in ((6, 1), (7, -1)):
            c = self.rack_c + self.rack_R @ [-lx, sign * ly]
            self.tags.append((tag_id, np.array([*c, self.rack_top - 0.0003]), -ry, rx))
        self.col = np.array(s['column'])
        sx = np.array([math.cos(s['yaw']), math.sin(s['yaw'])])
        self.stand_x = sx
        tag_c = self.col + s['tag_x'] * sx
        tx = np.append(sx, 0)                        # tag +x points away from the column (our real stand)
        ty = np.cross([0, 0, 1], tx)
        self.tags.append((8, np.array([*tag_c, TABLE_Z + s['plate'][2] - 0.0003]), tx, ty))
        self.textures = {i: marker_texture(i) for i in (6, 7, 8)}
        self.seat_z = TABLE_Z + s['seat']
        self.jar = np.array([*self.slots[slot[0] * 2 + slot[1]], TABLE_Z + r['size'][2] - r['depth']])  # bottom
        self.held = None                              # jar bottom in end_effector_link when held

    # ------------------------------------------------------------------ nudges (a grasp pushed it)
    def move_stand(self, dx, dy):
        """Slide the stand (and a jar standing in it) along the table."""
        d = np.array([dx, dy])
        in_stand = self.held is None and math.dist(self.jar[:2], self.col) < 0.005
        self.col = self.col + d
        self.tags = [(i, c + [dx, dy, 0.0] if i == 8 else c, tx, ty) for i, c, tx, ty in self.tags]
        if in_stand:
            self.jar = self.jar + [dx, dy, 0.0]

    def move_rack(self, dx, dy):
        """Slide the rack (and a jar standing in it) along the table."""
        d = np.array([dx, dy])
        in_rack = self.held is None and any(math.dist(self.jar[:2], c) < 0.005 for c in self.slots)
        self.rack_c = self.rack_c + d
        self.slots = [c + d for c in self.slots]
        self.tags = [(i, c + [dx, dy, 0.0] if i in (6, 7) else c, tx, ty) for i, c, tx, ty in self.tags]
        if in_rack:
            self.jar = self.jar + [dx, dy, 0.0]

    # ------------------------------------------------------------------ jar handling
    def jar_bottom(self, T_ee):
        if self.held is None:
            return self.jar
        return T_ee[:3, :3] @ self.held + T_ee[:3, 3]

    def grasp(self, T_ee, reach=0.130):
        pad = T_ee[:3, :3] @ [0, 0, reach] + T_ee[:3, 3]
        b = self.jar
        if math.dist(pad[:2], b[:2]) < 0.012 and b[2] + 0.01 < pad[2] < b[2] + JAR['height'] + 0.002:
            self.held = np.linalg.inv(T_ee[:3, :3]) @ (b - T_ee[:3, 3])
            return True
        return False

    def release(self, T_ee):
        if self.held is None:
            return
        b = self.jar_bottom(T_ee)
        self.held = None
        for c, floor in [(s, self.rack_top - RACK['depth']) for s in self.slots] + [(self.col, self.seat_z)]:
            d = math.dist(b[:2], c)
            if d < RACK['mouth'] / 2 - JAR['radius']:            # caught by the lead-in
                self.jar = np.array([c[0], c[1], floor])
                return
            if d < RACK['mouth'] / 2:                            # its bottom on the lead-in: stuck, high
                self.jar = np.array([b[0], b[1], floor + 0.004])
                return
        self.jar = np.array([b[0], b[1], TABLE_Z])

    # ------------------------------------------------------------------ rendering
    def render(self, T_base_cam, T_ee):
        f = W / 2 / math.tan(HFOV / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d_cam = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f, np.ones_like(u)], -1).reshape(-1, 3)
        d = d_cam @ T_base_cam[:3, :3].T
        o = T_base_cam[:3, 3]
        n = len(d)
        best = np.full(n, np.inf)
        col = np.full(n, 90.0)

        def plane(z):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (z - o[2]) / d[:, 2]
            t[~np.isfinite(t) | (t <= 0)] = np.inf
            return t, o + t[:, None] * d

        def take(t, ok, shade):
            nonlocal best, col
            sel = ok & (t < best)
            best[sel] = t[sel]
            col[sel] = shade if np.isscalar(shade) else shade[sel]

        t, p = plane(TABLE_Z)
        take(t, np.ones(n, bool), 150.0)
        # rack top with pocket holes, pocket floors
        t, p = plane(self.rack_top)
        local = (p[:, :2] - self.rack_c) @ self.rack_R
        in_rack = (np.abs(local[:, 0]) < RACK['size'][0] / 2) & (np.abs(local[:, 1]) < RACK['size'][1] / 2)
        in_hole = np.zeros(n, bool)
        for c in self.slots:
            in_hole |= np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < RACK['mouth'] / 2
        take(t, in_rack & ~in_hole, 60.0)
        t, p = plane(self.rack_top - RACK['depth'])
        hole = np.zeros(n, bool)
        for c in self.slots:
            hole |= np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < RACK['mouth'] / 2
        take(t, hole, 40.0)
        # stand plate, column top ring, seat
        s = STAND
        t, p = plane(TABLE_Z + s['plate'][2])
        lp = p[:, :2] - self.col
        ax = lp @ self.stand_x
        ay = lp @ np.array([-self.stand_x[1], self.stand_x[0]])
        take(t, (np.abs(ax - s['plate_cx']) < s['plate'][0] / 2) & (np.abs(ay) < s['plate'][1] / 2), 60.0)
        t, p = plane(TABLE_Z + s['top'])
        r = np.hypot(p[:, 0] - self.col[0], p[:, 1] - self.col[1])
        take(t, (r < s['radius']) & (r > s['mouth'] / 2), 60.0)
        t, p = plane(self.seat_z)
        take(t, np.hypot(p[:, 0] - self.col[0], p[:, 1] - self.col[1]) < s['mouth'] / 2, 40.0)
        t_side = self._cylinder(o, d, self.col, s['radius'], TABLE_Z, TABLE_Z + s['top'])
        take(t_side, np.isfinite(t_side), 50.0)
        # bowl: outer wall, rim ring, inside wall and floor
        if self.bowl:
            bw = self.bowl
            c, R0 = np.array(bw['centre']), bw['radius']
            Ri, top = R0 - bw['wall'], TABLE_Z + bw['height']
            t_side = self._cylinder(o, d, c, R0, TABLE_Z, top)
            take(t_side, np.isfinite(t_side), 110.0)
            t, p = plane(top)
            rr = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
            take(t, (rr < R0) & (rr >= Ri), 225.0)
            through = rr < Ri                                       # rays into the bowl
            t_in = self._cylinder(o, d, c, Ri, TABLE_Z + bw['floor'], top, inside=True)
            take(t_in, through & np.isfinite(t_in), 95.0)
            t, p = plane(TABLE_Z + bw['floor'])
            take(t, through & (np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) < Ri), 125.0)
        # tags
        for tag_id, c, tx, ty in self.tags:
            t, p = plane(c[2] + 0.0005)          # drawn just above the face it is recessed in
            q = p - c
            a, b = q @ tx, q @ ty
            half = TAG / 2 * 10 / 8
            ok = (np.abs(a) < half) & (np.abs(b) < half)
            tex = self.textures[tag_id]
            N = tex.shape[0]
            ci = np.clip(((a / (2 * half) + 0.5) * N).astype(int), 0, N - 1)
            ri = np.clip(((0.5 - b / (2 * half)) * N).astype(int), 0, N - 1)
            take(t, ok, tex[ri, ci].astype(float))
        # jar: body + lid side + lid top
        b = self.jar_bottom(T_ee)
        top = b[2] + JAR['height']
        t_side = self._cylinder(o, d, b[:2], JAR['radius'], b[2], top - JAR['lid_h'])
        take(t_side, np.isfinite(t_side), 200.0)
        t_side = self._cylinder(o, d, b[:2], JAR['lid_radius'], top - JAR['lid_h'], top)
        take(t_side, np.isfinite(t_side), 230.0)
        t, p = plane(top)
        take(t, np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < JAR['lid_radius'], 235.0)

        depth = (best * d_cam[:, 2]).reshape(H, W).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        img = np.clip(col.reshape(H, W), 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), depth, K

    @staticmethod
    def _cylinder(o, d, c, r, z0, z1, inside=False):
        ox, oy = o[0] - c[0], o[1] - c[1]
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        b = 2 * (ox * d[:, 0] + oy * d[:, 1])
        cc = ox ** 2 + oy ** 2 - r ** 2
        disc = b ** 2 - 4 * a * cc
        with np.errstate(invalid='ignore', divide='ignore'):
            t = (-b + (1 if inside else -1) * np.sqrt(disc)) / (2 * a)
        z = o[2] + t * d[:, 2]
        t[(disc < 0) | (t <= 0) | (z < z0) | (z > z1) | ~np.isfinite(t)] = np.inf
        return t


# ---------------------------------------------------------------------- fakes for Task
HAND_EYE_T = (0.06041, -0.01027, 0.06052)
HAND_EYE_Q = (0.706184, -0.014006, 0.707857, 0.006797)
R_LINK_OPTICAL = np.column_stack(([0, -1, 0], [0, 0, -1], [1, 0, 0]))


def pose_to_T(p):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*p[3:])
    T[:3, 3] = p[:3]
    return T


def ee_cam(he):
    """end_effector_link -> camera optical frame for a hand_eye dict (or HW7's)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*(he['rotation'] if he else HAND_EYE_Q)) @ R_LINK_OPTICAL
    T[:3, 3] = he['translation'] if he else HAND_EYE_T
    return T


class FakeIO:
    def __init__(self, world, start_pose, noise=0.0004, seed=0, cfg=None, mount=None):
        """cfg: the task's config (its hand_eye = where the code THINKS the camera is);
        mount: where the camera really is ('outside' / 'inside' of cfg's hand_eye_mounts)."""
        self.world = world
        self.cfg = cfg
        self.T_ee = pose_to_T(start_pose)
        self.T_ee_cam = ee_cam(cfg['hand_eye'] if cfg else None)                 # believed
        self.T_ee_cam_true = ee_cam(cfg['hand_eye_mounts'][mount]) if mount else self.T_ee_cam
        self.D = np.zeros(5)
        self.K = None
        self.finger = 0.0
        self.last_grip = None
        self.rng = np.random.default_rng(seed)
        self.noise = noise

    def ee_pose(self):
        return self.T_ee.copy()

    def camera_in_ee(self):
        return self.T_ee_cam[:3, 3].copy()

    def mount_name(self):
        he = self.cfg['hand_eye']
        return next((n for n, m in self.cfg['hand_eye_mounts'].items()
                     if np.allclose(m['translation'], he['translation'])), None)

    def set_mount(self, name):
        self.cfg['hand_eye'] = self.cfg['hand_eye_mounts'][name]
        self.T_ee_cam = ee_cam(self.cfg['hand_eye'])
        return True

    def _frame(self):
        T_base_cam = self.T_ee @ self.T_ee_cam                      # what the code believes
        img, depth, self.K = self.world.render(self.T_ee @ self.T_ee_cam_true, self.T_ee)
        depth = depth + self.rng.normal(0, self.noise, depth.shape).astype(np.float32)
        return img, depth, T_base_cam

    def frames(self, n, timeout=5.0):
        f = self._frame()
        return [f] * min(n, 3)

    def snapshot(self, n_depth=5, timeout=5.0):
        return self._frame()

    def set_gripper(self, position):
        if position >= 0.5:
            held = self.world.grasp(self.T_ee)
            self.finger = 0.57 if held else position - 0.005
            self.last_grip = {'position': self.finger, 'reached': not held, 'stalled': held}
        else:
            self.world.release(self.T_ee)
            self.finger = position
        return True


class FakeMover:
    def __init__(self, io):
        self.io = io
        self.moves = []

    def move_cartesian(self, pose, quiet=False):
        self.moves.append(('line', list(pose)))
        self.io.T_ee = pose_to_T(pose)
        return True

    def reachable(self, pose):
        return True

    def set_scene(self, cfg, table_top_z=None, rack=None, stand=None, bowl=None):
        self.scene = table_top_z
        self.scene_objects = (rack, stand, bowl)
        return True

    def move_free(self, pose):
        self.last_planner = 'PTP'
        return self.move_rrtconnect(pose)

    def move_rrtconnect(self, pose):
        self.moves.append(('free', list(pose)))
        self.io.T_ee = pose_to_T(pose)
        return True
