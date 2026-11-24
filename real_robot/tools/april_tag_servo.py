#!/usr/bin/env python3
"""AprilTag grasp: coarse visual servoing to a good view, then a MoveIt grasp from the measured tag pose.

  Stage 1  servo    IBVS on the tag's 4 corners (twist_controller) until the tag is centered at
                    VIEW_DISTANCE. It only needs a good view (VIEW_OK_PX), not an exact match.
  Stage 2  measure  hold still and average the tag pose over fresh frames -> T_base_tag.
  Stage 3  grasp    MoveIt: pre-grasp -> open -> straight in along the lid's axis -> close -> pull out.
                    Asks before every motion.

Keys (click the camera window first): s = servo on/off, r = record on/off, q = quit.

Needs the robot + MoveIt launch, the RealSense, and the hand-eye transform
(end_effector_link -> camera_link, the command in config/hand_eye_result.txt).
"""
import argparse
import os
import sys
import threading
import time
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import rclpy
import yaml
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_ros import Buffer, TransformListener

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from helpers.geometry import matrix_to_quaternion, rotation_angle_deg, transform_to_matrix  # noqa: E402
from helpers.drawing import put_outlined_text  # noqa: E402
from helpers.robot_motion import ArmMover  # noqa: E402

# ---- tag -------------------------------------------------------------------------------------
TAG_ID = 0
# m, edge of the BLACK square (not the white border); 0.0147 = depth-camera estimate, measure it
TAG_SIZE = 0.0147
# Grasp point in the tag frame (m): x = red axis, y = green axis, z = out of the tag face.
# Tune x/y until the yellow cross sits on the lid center; z = how far below the lid the fingers close.
# This shaker: lid centre ~13 mm along red (x); fingers 20 mm below the lid top.
GRASP_IN_TAG = np.array([0.013, 0.003, -0.020])

# ---- ROS names -------------------------------------------------------------------------------
BASE_FRAME = 'base_link'
EE_FRAME = 'end_effector_link'
TWIST_FRAME = 'tool_frame'            # Kortex interprets twists in the Kinova tool frame
IMAGE_TOPIC = '/camera/camera/color/image_raw'
INFO_TOPIC = '/camera/camera/color/camera_info'
TWIST_TOPIC = '/twist_controller/commands'
SWITCH_SERVICE = '/controller_manager/switch_controller'
ARM_CONTROLLER = 'joint_trajectory_controller'
TWIST_CONTROLLER = 'twist_controller'
GRIPPER_CONTROLLER = 'gen3_lite_2f_gripper_controller'
GRIPPER_ACTION = f'/{GRIPPER_CONTROLLER}/gripper_cmd'

# ---- stage 1: servo --------------------------------------------------------------------------
VIEW_DISTANCE = 0.20      # m, camera -> tag at the goal (the fingertips are ~11 cm in front of the camera)
VIEW_OK_PX = 20.0         # mean corner error that counts as a good view
# Servo goal view of the tag:
#   False: square to the tag (camera looking straight down on a flat lid), keeping the tag's current
#          in-plane rotation from its corners, so the wrist doesn't spin. Use this for upright shakers.
#   True:  keep the tag's tilt from when servoing starts (mostly translation). That tilt comes from the
#          small tag's 3D pose, which is often tens of degrees off, so the view can end up oblique.
KEEP_START_TILT = False
GAIN = 0.5
V_MAX, W_MAX = 0.02, 0.2  # m/s, rad/s
SERVO_TIMEOUT = 60.0      # s
STUCK_SEC = 3.0           # no 1 px improvement for this long = stuck
STUCK_OK_PX = 40.0        # stuck or at a joint limit below this error: the view is still good enough to measure
STALE_FRAME_SEC = 0.3     # no new camera frame for this long while servoing: stop the arm

# Gen3 Lite joint limits (rad), from kortex_description gen3_lite_macro.xacro
JOINT_LIMITS = {'joint_1': 2.68, 'joint_2': 2.61, 'joint_3': 2.61, 'joint_4': 2.6, 'joint_5': 2.53, 'joint_6': 2.6}
NEAR_LIMIT_DEG = 10.0     # shown in red
LIMIT_STOP_DEG = 5.0      # servoing stops this close to a limit

# ---- stage 2: measure ------------------------------------------------------------------------
MEASURE_FRAMES = 15

# ---- stage 3: grasp --------------------------------------------------------------------------
DO_GRASP = True
APPROACH = 'vertical'     # 'tag_normal': along the lid's axis (tilted cylinder); 'vertical': straight down
PREGRASP = 0.10           # m, back along the approach; also how far it pulls out
FINGERTIP = 0.130         # m, end_effector_link -> finger-pad centre (URDF tool_frame)
PUT_DOWN = True           # PLACE_TARGET = 'back': hold it up, then put it back where it was and let go
HOLD_SEC = 2.0            # s, held up this long before putting it back
PLACE_DROP = 0.003        # m, let go this far short of the grasp pose (back along the approach)

# ---- stage 4: place in the holder (regrasp stand, tag 8). Geometry from the team's stand CAD
# (RBE595FinalProject rbe595_sim/config/regrasp_stand.yaml): the pocket the jar stands in is
# STAND_TAG_X along the tag image's +x (red axis) from the tag centre, its floor STAND_SEAT_ABOVE_TAG
# above the tag face. Stick the real tag that way: its red axis pointing at the cup.
PLACE_TARGET = 'stand'    # 'stand': into the holder; 'back': put it back where it was; None: keep holding
STAND_TAG_ID = 8
STAND_TAG_X = 0.058       # m, pocket centre from the tag centre (distance)
STAND_TAG_ANGLE = -90.0   # deg, direction of the pocket from the tag, from the tag's +x towards +y.
                          # The CAD says 0 (red axis at the cup); the real sticker is turned: measured
                          # 57 mm at -90 deg (along the green axis, negative side) from view 0.
STAND_SEAT_ABOVE_TAG = 0.0353   # m, pocket floor above the tag face (0.041 above the table - 0.0057 tag face)
STAND_RIM_ABOVE_SEAT = 0.020    # m, pocket depth (for drawing the rim)
STAND_POCKET_MOUTH = 0.059      # m, pocket diameter at the rim (45 deg lead-in catches ~5 mm off-centre)
STAND_CLEARANCE = 0.06    # m, jar bottom this far above the pocket floor before lowering it in
STAND_DROP = 0.004        # m, let go with the jar bottom this far above the pocket floor
STAND_LOOK_VIEW = 0       # views.yaml view to look for the holder from, if it wasn't seen at the servo view
SHAKER_HEIGHT = 0.0813    # m, jar base to lid top -- MEASURE the real shaker (sets how low it is let go)
STAND_DIR = os.path.join(HERE, '..', 'captures', 'stand_checks')
VIEWS_FILE = os.path.join(HERE, '..', 'config', 'views.yaml')
# Open = 0.05, not 0.0: fully open, the real gripper reads slightly negative (-0.04), outside the URDF's
# [0.0, 0.85] finger limits on the real robot. MoveIt then warns "out of bounds" and, past ~0.05 rad,
# refuses to plan from that start state; Kortex also rejected it (VALUE_IS_BELOW_MINIMUM). 0.05 still
# opens ~105 mm (vs ~110) -- plenty for a 50 mm lid.
GRIPPER_OPEN, GRIPPER_CLOSED, GRIPPER_EFFORT = 0.05, 0.8, 100.0

# ---- stage 3b: close-up lid correction, at the pre-grasp before going in
# From the pre-grasp the camera sees the whole lid up close. Its centre is found from the depth image
# (the highest flat surface around the tag, outline fitted with a circle), so the grasp no longer
# depends on the tag size, the sticker's place on the lid or the measured tilt.
LID_CORRECTION = False
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'   # aligned to the color image, mm
LID_RADIUS = 0.025        # m, expected lid radius (50 mm lid); fits outside 0.6x..1.5x are rejected
LID_SEARCH = 0.07         # m, look this far (horizontally) around the tag for the lid
LID_BAND = 0.008          # m, lid top = surface within this of the highest one there
LID_HEIGHT_BELOW = 0.04   # m, the lid top is searched from this far below the tag's measured height ...
LID_HEIGHT_ABOVE = 0.03   # m, ... to this far above it (the tag is on the lid; its height is off by ~cm)
GRIPPER_CLEAR = 0.18      # m, ignore points closer than this to end_effector_link along its approach
                          #    (the fingertips end ~0.16 m out and are in the camera's view)
LID_DEPTH_FRAMES = 10     # depth images averaged (median)
GRASP_BELOW_LID = 0.020   # m, finger-pad centre (tool_frame) this far below the lid top
LID_DIR = os.path.join(HERE, '..', 'captures', 'lid_checks')

# ---- run options (set from the command line: --auto, --record)
AUTO = False              # no human in the loop: servo starts by itself, every prompt is answered yes
RECORD_ALL = False        # record the camera for the whole run (one video), no pop-up windows
AUTO_TAG_WAIT = 15.0      # s, --auto: give up if the tag is not seen this long after start
LID_SHOW_SEC = 2.0        # s, --record: the lid check picture is shown this long in the video

# ---- display / recording ---------------------------------------------------------------------
WINDOW = 'AprilTag servo'
VIDEO_DIR = os.path.join(HERE, '..', 'captures', 'servo_videos')
VIDEO_FPS = 15            # frames are repeated/dropped to match wall-clock time: playback runs at real speed
GREEN, RED, YELLOW, WHITE = (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)

FACING = np.array([np.pi, 0.0, 0.0])        # rvec of a tag square to the camera, upright
_h = TAG_SIZE / 2.0
TAG_CORNERS = np.array([[-_h, _h, 0], [_h, _h, 0], [_h, -_h, 0], [-_h, -_h, 0]])   # SOLVEPNP_IPPE_SQUARE order


# ============================================================================== small helpers
def wait(future, timeout=10.0):
    """Block until a future (completed by the spinning thread) is done; None on timeout."""
    end = time.monotonic() + timeout
    while not future.done() and time.monotonic() < end:
        time.sleep(0.05)
    return future.result() if future.done() else None


def ask(question):
    if AUTO:
        print(f'{question} [auto: yes]')
        return True
    return input(f'{question} [y/N] ').strip().lower() == 'y'


# ============================================================================== vision
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'ArucoDetector')
           else cv2.aruco.DetectorParameters_create())
# A tag with a thin white border stuck on a dark (blue) part: the white border's outer edge is found as a
# second square just outside the tag's black one, and with the default min distance the real one is
# dropped as its duplicate (seen with the holder/rack tags on the blue printed parts). 0 keeps both;
# the white one then fails to decode and only the tag remains.
_params.minMarkerDistanceRate = 0.0
if hasattr(cv2.aruco, 'ArucoDetector'):                       # OpenCV >= 4.7
    _detector = cv2.aruco.ArucoDetector(DICTIONARY, _params)
    detect_markers = _detector.detectMarkers
else:
    def detect_markers(gray):
        return cv2.aruco.detectMarkers(gray, DICTIONARY, parameters=_params)


@dataclass
class Tag:
    corners: np.ndarray    # (4, 2) pixels, TAG_CORNERS order
    rvec: np.ndarray
    tvec: np.ndarray
    T_cam_tag: np.ndarray  # 4x4


def find_tag(image, K, D):
    """-> (Tag for TAG_ID or None, all detected corners, all ids) for drawing."""
    corners, ids, _ = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None or TAG_ID not in ids.ravel():
        return None, corners, ids
    px = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(TAG_CORNERS, px, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None, corners, ids
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    return Tag(px, rvec, tvec, T), corners, ids


def square_goal_rvec(corners_px):
    """Goal orientation: tag square to the camera (FACING), turned about the optical axis to match the
    tag's current in-plane rotation in the image (edge corner 0 -> 1 is the tag's +x). Only the corners'
    2-D layout is used, not the tag's (ambiguous) 3-D tilt."""
    du, dv = corners_px[1] - corners_px[0]
    theta = np.arctan2(dv, du)
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    return cv2.Rodrigues(Rz @ cv2.Rodrigues(FACING)[0])[0].ravel()


def goal_corners(K, D, rvec):
    """Pixels of the tag corners at the goal: centered, VIEW_DISTANCE ahead, orientation rvec."""
    pts, _ = cv2.projectPoints(TAG_CORNERS, rvec, np.array([0.0, 0.0, VIEW_DISTANCE]), K, D)
    return pts.reshape(4, 2)


def corner_depths(R, t):
    return (TAG_CORNERS @ R.T + t)[:, 2]


def normalized(px, K, D):
    """Pixels (N, 2) -> undistorted normalized image coordinates [x1, y1, x2, y2, ...]."""
    return cv2.undistortPoints(px.reshape(-1, 1, 2), K, D).reshape(-1)


# ============================================================================== servo math
def interaction_matrix(s, Z):
    """Image Jacobian of normalized point features s = [x1, y1, ...] at depths Z."""
    rows = []
    for (x, y), z in zip(s.reshape(-1, 2), Z):
        rows.append([-1 / z, 0, x / z, x * y, -(1 + x * x), y])
        rows.append([0, -1 / z, y / z, 1 + y * y, -x * y, -x])
    return np.array(rows)


def ibvs_twist(s, s_goal, Z, Z_goal):
    """Camera-frame (v, w) from IBVS with the averaged current/goal Jacobian, clamped to V_MAX / W_MAX."""
    L = 0.5 * (interaction_matrix(s, Z) + interaction_matrix(s_goal, Z_goal))
    twist = -GAIN * np.linalg.pinv(L) @ (s - s_goal)
    twist /= max(1.0, np.linalg.norm(twist[:3]) / V_MAX, np.linalg.norm(twist[3:]) / W_MAX)
    return twist[:3], twist[3:]


def camera_to_tool(v, w, T_tool_cam):
    """Re-express a camera twist in the tool frame (the twist of the tool origin)."""
    R, p = T_tool_cam[:3, :3], T_tool_cam[:3, 3]
    w_tool = R @ w
    return R @ v + np.cross(p, w_tool), w_tool


# ============================================================================== ROS node
class TagServoNode(Node):
    """ROS I/O only: camera, joints, TF, twist commands, controller switching, gripper."""

    def __init__(self):
        super().__init__('apriltag_servo')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.image, self.camera_frame, self.K, self.D = None, None, None, None
        self.frame_count, self.frame_time = 0, 0.0
        self.joint_pos, self.joint_vel = {}, {}
        self.create_subscription(Image, IMAGE_TOPIC, self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, INFO_TOPIC, self.on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.depth, self.depth_count = None, 0
        self.last_tag_xyz = None             # tag position (base_link) from the measure stage
        self.stand_T = None                  # holder tag pose (base_link), if measured
        self.used_lid_correction = False
        self.create_subscription(Image, DEPTH_TOPIC, self.on_depth, qos_profile_sensor_data)
        self.twist_pub = self.create_publisher(Twist, TWIST_TOPIC, 10)
        self.switch_client = self.create_client(SwitchController, SWITCH_SERVICE)
        self.gripper_client = ActionClient(self, GripperCommand, GRIPPER_ACTION)

    # ---- callbacks
    def on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
        self.camera_frame = msg.header.frame_id
        self.frame_count += 1
        self.frame_time = time.monotonic()

    def on_depth(self, msg):
        self.depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) * 0.001   # m
        self.depth_count += 1

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_joints(self, msg):
        for i, name in enumerate(msg.name):
            if name in JOINT_LIMITS:
                if i < len(msg.position):
                    self.joint_pos[name] = msg.position[i]
                if i < len(msg.velocity):
                    self.joint_vel[name] = msg.velocity[i]

    # ---- robot
    def send_twist(self, v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0)):
        """Tool-frame twist: v in m/s, w in rad/s. The Kinova twist API takes deg/s, converted here.
        The twist controller holds the last command forever, so every stop must be explicit."""
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = (float(c) for c in v)
        msg.angular.x, msg.angular.y, msg.angular.z = (float(c) for c in np.degrees(w))
        self.twist_pub.publish(msg)

    def stop(self):
        self.send_twist()

    def switch_controllers(self, activate, deactivate):
        if not self.switch_client.wait_for_service(timeout_sec=5.0):
            return False
        req = SwitchController.Request()
        req.activate_controllers, req.deactivate_controllers = list(activate), list(deactivate)
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        res = wait(self.switch_client.call_async(req))
        return res is not None and res.ok

    def set_gripper(self, position):
        """Send a gripper goal and print what the controller reports. True if it reached the goal or
        stalled on an object (both count as done)."""
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            print(f'Gripper: {GRIPPER_ACTION} not available (is {GRIPPER_CONTROLLER} active?)')
            return False
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(position), GRIPPER_EFFORT
        handle = wait(self.gripper_client.send_goal_async(goal))
        if handle is None or not handle.accepted:
            print('Gripper: goal rejected')
            return False
        response = wait(handle.get_result_async(), timeout=15.0)
        if response is None:
            print('Gripper: no result within 15 s')
            return False
        r = response.result
        print(f'Gripper: asked {position:.2f}, now at {r.position:.3f}, reached_goal={r.reached_goal}, '
              f'stalled={r.stalled}, status={response.status}')
        return r.reached_goal or r.stalled

    def lookup(self, target, source):
        return transform_to_matrix(self.tf_buffer.lookup_transform(target, source, Time()).transform)

    def wait_for_transform(self, target, source, tries=20):
        for _ in range(tries):
            try:
                return self.lookup(target, source)
            except Exception as e:
                print(f'Waiting for TF {target} -> {source}: {e}')
                time.sleep(0.5)
        return None

    # ---- joints
    def joint_margins_deg(self):
        return {name: np.degrees(limit - abs(self.joint_pos[name]))
                for name, limit in JOINT_LIMITS.items() if name in self.joint_pos}

    def joint_near_limit(self, margin_deg):
        return next((name for name, m in self.joint_margins_deg().items() if m < margin_deg), None)

    def joint_report(self):
        """[(text, near_limit)] per joint, plus one line on whether the arm is moving."""
        margins = self.joint_margins_deg()
        lines = [(f'{name} {np.degrees(self.joint_pos[name]):7.1f} deg ({m:5.1f} to limit)'
                  + ('  NEAR LIMIT' if m < NEAR_LIMIT_DEG else ''), m < NEAR_LIMIT_DEG)
                 for name, m in margins.items()]
        speed = np.degrees(max((abs(v) for v in self.joint_vel.values()), default=0.0))
        return lines, f'arm {"MOVING" if speed > 0.5 else "still"} (max joint speed {speed:.1f} deg/s)'


# ============================================================================== recording
class Recorder:
    """Records annotated frames to an MP4 that plays back at real speed."""

    def __init__(self):
        self.writer, self.path = None, None
        self.lock = threading.Lock()
        self.thread, self.node, self.running = None, None, False
        self.stage, self.still, self.still_until = '', None, 0.0

    @property
    def active(self):
        return self.writer is not None

    def toggle(self, frame):
        self.stop() if self.active else self.start(frame)

    def start(self, frame):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        self.path = os.path.abspath(os.path.join(VIDEO_DIR, time.strftime('servo_%Y%m%d_%H%M%S.mp4')))
        h, w = frame.shape[:2]
        for codec in ('avc1', 'mp4v'):          # H.264 previews in browsers/Slack; mp4v as a fallback
            writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*codec), VIDEO_FPS, (w, h))
            if writer.isOpened():
                self.writer = writer
                self.t0, self.frames = time.monotonic(), 0
                print(f'Recording to {self.path} ({codec})')
                return
        print(f'Could not open a video writer for {self.path}')

    def write(self, frame):
        with self.lock:
            if self.writer is None:
                return
            due = max(self.frames + 1, int((time.monotonic() - self.t0) * VIDEO_FPS))
            while self.frames < due:
                self.writer.write(frame)
                self.frames += 1

    def stop(self):
        self.stop_live()
        with self.lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
                print(f'Saved video ({self.frames / VIDEO_FPS:.1f} s): {self.path}')

    # ---- whole-run recording outside the servo loop: live camera + current step, in a thread
    def start_live(self, node):
        if self.running or not self.active:
            return
        self.node, self.running = node, True
        self.thread = threading.Thread(target=self._live_loop, daemon=True)
        self.thread.start()

    def stop_live(self):
        if self.running:
            self.running = False
            self.thread.join(timeout=2.0)

    def set_stage(self, text):
        self.stage = text

    def show_still(self, picture, seconds=LID_SHOW_SEC):
        """Put a picture (e.g. the lid check) in the video for a while instead of the live camera."""
        self.still, self.still_until = picture.copy(), time.monotonic() + seconds

    def _live_loop(self):
        while self.running:
            if self.still is not None and time.monotonic() < self.still_until:
                frame = self.still.copy()
            elif self.node.image is not None:
                frame = self.node.image.copy()
            else:
                time.sleep(0.05)
                continue
            if self.stage:
                put_outlined_text(frame, self.stage, (10, 30), 0.8, YELLOW)
            self.write(frame)
            time.sleep(1.0 / VIDEO_FPS)


# ============================================================================== display
def draw_overlay(image, node, tag, all_corners, ids, goal_px, lines):
    """Markers, tag axes, grasp point, goal corners, status lines (top-left), joints (bottom-left)."""
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image, all_corners, ids)
    for u, v in goal_px:
        cv2.circle(image, (int(u), int(v)), 5, GREEN, -1)
    if tag is not None:
        cv2.drawFrameAxes(image, node.K, node.D, tag.rvec, tag.tvec, TAG_SIZE)
        for u, v in tag.corners:
            cv2.circle(image, (int(u), int(v)), 5, RED, -1)
        surface = np.array([[GRASP_IN_TAG[0], GRASP_IN_TAG[1], 0.0]])      # on the lid surface, so it's visible
        (gu, gv), = cv2.projectPoints(surface, tag.rvec, tag.tvec, node.K, node.D)[0].reshape(-1, 2)
        cv2.drawMarker(image, (int(gu), int(gv)), YELLOW, cv2.MARKER_CROSS, 25, 2)
        cv2.putText(image, 'grasp', (int(gu) + 8, int(gv) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2)
    for row, (text, color) in enumerate(lines):
        cv2.putText(image, text, (10, 25 + 25 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    joint_lines, moving = node.joint_report()
    y0 = image.shape[0] - 20 * (len(joint_lines) + 1) - 5
    cv2.putText(image, moving, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
    for row, (text, near) in enumerate(joint_lines, start=1):
        cv2.putText(image, text, (10, y0 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED if near else WHITE, 2)


def draw_rec_marker(image):
    cv2.circle(image, (image.shape[1] - 70, 25), 8, RED, -1)
    cv2.putText(image, 'REC', (image.shape[1] - 55, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)


# ============================================================================== stages 1 + 2
def servo_and_measure(node, T_tool_cam, recorder):
    """Keyboard-driven loop: idle -> servo -> measure. Returns (tag pose samples, stop reason).
    Every exit path leaves the arm stopped; the caller still sends a final stop."""
    mode = 'idle'
    goal_rvec = FACING
    samples, last_sample_frame = [], -1
    best_error, best_time, deadline = np.inf, time.monotonic(), np.inf
    started = time.monotonic()

    while rclpy.ok():
        frame_no, image = node.frame_count, node.image.copy()
        now = time.monotonic()
        tag, all_corners, ids = find_tag(image, node.K, node.D)
        goal_px = goal_corners(node.K, node.D, goal_rvec)

        error_px, v, w = np.inf, np.zeros(3), np.zeros(3)
        if tag is not None:
            error_px = np.linalg.norm(tag.corners - goal_px, axis=1).mean()
            R_goal = cv2.Rodrigues(goal_rvec)[0]
            v_cam, w_cam = ibvs_twist(normalized(tag.corners, node.K, node.D), normalized(goal_px, node.K, node.D),
                                      corner_depths(tag.T_cam_tag[:3, :3], tag.T_cam_tag[:3, 3]),
                                      corner_depths(R_goal, np.array([0.0, 0.0, VIEW_DISTANCE])))
            v, w = camera_to_tool(v_cam, w_cam, T_tool_cam)

        sent = False
        if mode == 'servo':
            if error_px < best_error - 1.0:
                best_error, best_time = error_px, now
            limited = node.joint_near_limit(LIMIT_STOP_DEG)
            if error_px < VIEW_OK_PX:
                why = f'good view ({error_px:.1f} px)'
            elif limited:
                why = f'{limited} within {LIMIT_STOP_DEG:.0f} deg of its limit'
            elif now - best_time > STUCK_SEC:
                why = 'tag lost' if tag is None else f'no progress for {STUCK_SEC:.0f} s'
            elif now > deadline:
                why = f'timeout after {SERVO_TIMEOUT:.0f} s'
            else:
                why = None

            if why is not None:
                node.stop()
                if error_px < STUCK_OK_PX:
                    print(f'{why}: holding still and measuring the tag pose')
                    mode = 'measure'
                else:
                    lines, moving = node.joint_report()
                    print('Joint state:\n  ' + '\n  '.join(t for t, _ in lines) + f'\n  {moving}')
                    return samples, f'{why}, error still {error_px:.1f} px'
            elif tag is None or now - node.frame_time > STALE_FRAME_SEC:
                node.stop()                                   # nothing fresh to act on
            else:
                node.send_twist(v, w)
                sent = True

        elif mode == 'measure':
            if tag is not None and frame_no != last_sample_frame:   # each camera frame counts once
                samples.append(tag.T_cam_tag)
                last_sample_frame = frame_no
            if len(samples) >= MEASURE_FRAMES:
                return samples, f'tag pose measured over {len(samples)} frames'
            if now > deadline:
                return samples, f'timeout while measuring ({len(samples)} frames)'

        status = (f'tag at {np.linalg.norm(tag.tvec) * 100:.0f} cm' if tag is not None else 'tag NOT found')
        mode_line = {'idle': ('SERVO OFF (s = start)', GREEN), 'servo': ('SERVO ON', RED),
                     'measure': (f'MEASURING {len(samples)}/{MEASURE_FRAMES} - holding still', YELLOW)}[mode]
        lines = [(status + ' | s=servo r=record q=quit', GREEN)]
        if tag is not None:
            lines += [(f'error: {error_px:.1f} px', GREEN),
                      (f'vel: {np.linalg.norm(v):.3f} m/s, {np.linalg.norm(w):.3f} rad/s'
                       + ('' if sent else ' (not sent)'), GREEN)]
        lines.append(mode_line)
        draw_overlay(image, node, tag, all_corners, ids, goal_px, lines)
        recorder.write(image)                                 # the video gets the frame without the REC marker
        if recorder.active:
            draw_rec_marker(image)
        cv2.imshow(WINDOW, image)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            return samples, 'user pressed q'
        if key == ord('r') and not RECORD_ALL:
            recorder.toggle(image)
        if RECORD_ALL and not recorder.active:
            recorder.start(image)
        if AUTO and mode == 'idle':
            if tag is not None:
                key = ord('s')                               # start servoing by itself
            elif now - started > AUTO_TAG_WAIT:
                return samples, f'tag not seen within {AUTO_TAG_WAIT:.0f} s'
        if key == ord('s'):
            if mode != 'idle':
                node.stop()
                mode, samples = 'idle', []
                print('Servoing OFF')
            elif tag is None:
                print('Tag not visible; servoing not started')
            else:
                goal_rvec = tag.rvec.ravel().copy() if KEEP_START_TILT else square_goal_rvec(tag.corners)
                mode = 'servo'
                best_error, best_time, deadline = np.inf, now, now + SERVO_TIMEOUT
                print(f'Servoing ON: goal = tag centered at {VIEW_DISTANCE * 100:.0f} cm'
                      + (', keeping its current tilt' if KEEP_START_TILT else ', looking square at it'))
    return samples, 'ROS shut down'


def run_servo_stage(node, T_tool_cam, recorder):
    """Switch to twist control, run the loop, and always stop + switch back afterwards."""
    if not node.switch_controllers([TWIST_CONTROLLER], [ARM_CONTROLLER]):
        print(f'Could not switch to {TWIST_CONTROLLER}; not servoing.')
        return []
    samples, reason = [], 'exception (traceback below)'
    try:
        samples, reason = servo_and_measure(node, T_tool_cam, recorder)
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        try:
            node.stop()
            time.sleep(0.1)                                   # let the stop command go out
        except Exception as e:
            print(f'WARNING: stop command failed ({e}); the arm may still be moving! Use the e-stop or:\n'
                  f'  ros2 topic pub --once {TWIST_TOPIC} geometry_msgs/msg/Twist "{{}}"')
        print(f'Servoing stopped: {reason}')
        if not RECORD_ALL:
            recorder.stop()                                   # --record keeps recording to the end
        if node.switch_controllers([ARM_CONTROLLER], [TWIST_CONTROLLER]):
            print(f'Switched back to {ARM_CONTROLLER}.')
        else:
            print(f'WARNING: switching back to {ARM_CONTROLLER} failed; check: ros2 control list_controllers')
    return samples


# ============================================================================== stage 3
def average_pose(samples):
    """Mean position; mean rotation projected back onto a valid rotation (SVD). Also the max spread in mm."""
    T = np.mean(samples, axis=0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    spread_mm = 1000 * max(np.linalg.norm(S[:3, 3] - T[:3, 3]) for S in samples)
    return T, spread_mm


def grasp_midpoint(T_base_tag, approach=APPROACH):
    """Where the finger pads close (base_link): GRASP_IN_TAG in the full tag frame for 'tag_normal',
    in the levelled tag frame (the tag's yaw only, z straight up) for 'vertical'. The measured tilt of a
    small tag is often tens of degrees off; in the full frame the "20 mm into the lid" part of
    GRASP_IN_TAG would then shift the grasp sideways by ~1 cm, which the yellow cross doesn't show."""
    if approach == 'tag_normal':
        return (T_base_tag @ np.append(GRASP_IN_TAG, 1.0))[:3]
    tag_x = T_base_tag[:3, 0]
    xh = np.array([tag_x[0], tag_x[1], 0.0]) / np.linalg.norm(tag_x[:2])
    R_level = np.column_stack((xh, np.cross([0.0, 0.0, 1.0], xh), [0.0, 0.0, 1.0]))
    return T_base_tag[:3, 3] + R_level @ GRASP_IN_TAG


def grasp_poses(T_base_tag, R_current, approach=APPROACH):
    """end_effector_link poses (pre-grasp, grasp, pull-out) as [x, y, z, qx, qy, qz, qw] in base_link."""
    midpoint = grasp_midpoint(T_base_tag, approach)  # where the fingers close
    tag_x, tag_z = T_base_tag[:3, 0], T_base_tag[:3, 2]
    if approach == 'tag_normal':
        a = -tag_z                                   # into the lid, along the cylinder's axis
        y = tag_x - (tag_x @ a) * a                  # fingers close along the tag's x
    else:
        a = np.array([0.0, 0.0, -1.0])               # straight down
        y = np.array([tag_x[0], tag_x[1], 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, a)
    # The gripper is symmetric under 180 deg about the approach: take the wrist angle closer to the
    # current one, so joint_6 doesn't wind toward its limit.
    R = min(np.column_stack((x, y, a)), np.column_stack((-x, -y, a)),
            key=lambda R: rotation_angle_deg(R_current.T @ R))
    q = matrix_to_quaternion(R)
    grasp = midpoint - FINGERTIP * a                 # end_effector_link sits FINGERTIP behind the fingertips
    pre = grasp - PREGRASP * a
    return [*pre, *q], [*grasp, *q], [*pre, *q]      # pull out the same way it went in


# ============================================================================== stage 3b
def fit_circle(xy):
    """Least-squares circle through 2-D points -> (center, radius)."""
    A = np.column_stack((2 * xy, np.ones(len(xy))))
    c, *_ = np.linalg.lstsq(A, (xy ** 2).sum(axis=1), rcond=None)
    return c[:2], float(np.sqrt(c[2] + c[:2] @ c[:2]))


def project(points_base, T_base_cam, K, D):
    """base_link points (N, 3) -> pixels (N, 2)."""
    T = np.linalg.inv(T_base_cam)
    px, _ = cv2.projectPoints(np.asarray(points_base, float) @ T[:3, :3].T + T[:3, 3], np.zeros(3), np.zeros(3), K, D)
    return px.reshape(-1, 2)


def find_lid(node, tag_xyz):
    """Lid top from the aligned depth image. -> (center xyz in base_link, radius, None, picture) or
    (None, None, reason, picture)."""
    frames, last, end = [], node.depth_count, time.monotonic() + 3.0
    while len(frames) < LID_DEPTH_FRAMES and time.monotonic() < end:
        if node.depth is not None and node.depth_count != last:
            last = node.depth_count
            frames.append(node.depth.astype(np.float32))
        time.sleep(0.01)
    picture = node.image.copy()
    if not frames:
        return None, None, f'no depth images on {DEPTH_TOPIC}', picture
    stack = np.stack(frames)
    stack[stack == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                  # all-NaN pixels (no depth) stay NaN
        depth = np.nanmedian(stack, axis=0)
    h, w = depth.shape
    T_base_cam = node.lookup(BASE_FRAME, node.camera_frame)
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0.03))
    rays = cv2.undistortPoints(np.column_stack((u, v)).astype(np.float64).reshape(-1, 1, 2),
                               node.K, node.D).reshape(-1, 2)
    z = depth[v, u]
    pts = np.column_stack((rays * z[:, None], z)) @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    # Not the gripper: the fingers are in view and higher than the lid. Drop everything within
    # GRIPPER_CLEAR of end_effector_link along its approach axis.
    T_ee_base = np.linalg.inv(node.lookup(BASE_FRAME, EE_FRAME))
    along_approach = pts @ T_ee_base[2, :3] + T_ee_base[2, 3]
    # The lid top is where the tag is (the tag is stuck on it), within the tag's height error.
    near = ((np.linalg.norm(pts[:, :2] - tag_xyz[:2], axis=1) < LID_SEARCH)
            & (along_approach > GRIPPER_CLEAR)
            & (pts[:, 2] > tag_xyz[2] - LID_HEIGHT_BELOW) & (pts[:, 2] < tag_xyz[2] + LID_HEIGHT_ABOVE))
    if near.sum() < 100:
        return None, None, 'no depth around the tag', picture
    top_guess = np.percentile(pts[near, 2], 98)
    on_top = near & (pts[:, 2] > top_guess - LID_BAND)
    mask = np.zeros((h, w), np.uint8)
    mask[v[on_top], u[on_top]] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels = cv2.connectedComponents(mask)
    if count < 2:
        return None, None, 'no flat top found around the tag', picture
    # The lid is the region the tag sits on (a neighbouring jar's lid can be just as high).
    tag_px = project([tag_xyz], T_base_cam, node.K, node.D)[0]
    best, best_d = None, np.inf
    for k in range(1, count):
        ys, xs = np.nonzero(labels == k)
        if len(xs) < 50:
            continue
        d = np.min(np.hypot(xs - tag_px[0], ys - tag_px[1]))
        if d < best_d:
            best, best_d = k, d
    if best is None:
        return None, None, 'lid region too small', picture
    blob = labels == best
    top_z = float(np.median(pts[on_top][blob[v[on_top], u[on_top]], 2]))
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    inside = ((outline[:, 0] > 2) & (outline[:, 0] < w - 3) & (outline[:, 1] > 2) & (outline[:, 1] < h - 3))
    outline = outline[inside]                            # image-border edges are not the lid's edge
    # Outline pixels -> rays -> where they cross the lid-top plane.
    rays = cv2.undistortPoints(outline.reshape(-1, 1, 2), node.K, node.D).reshape(-1, 2)
    dirs = np.column_stack((rays, np.ones(len(rays)))) @ T_base_cam[:3, :3].T
    origin = T_base_cam[:3, 3]
    t = (top_z - origin[2]) / dirs[:, 2]
    rim = origin + t[:, None] * dirs
    rim = rim[np.linalg.norm(rim[:, :2] - tag_xyz[:2], axis=1) < LID_SEARCH - 0.003]   # not the search cut
    if len(rim) < 20:
        return None, None, 'too little of the lid edge visible', picture
    center_xy, radius = fit_circle(rim[:, :2])
    center = np.array([center_xy[0], center_xy[1], top_z])

    tint = picture.copy()
    tint[blob] = (0.5 * tint[blob] + [0, 127, 0]).astype(np.uint8)
    picture = tint
    ring = np.linspace(0, 2 * np.pi, 60)
    circle = np.column_stack((center[0] + radius * np.cos(ring), center[1] + radius * np.sin(ring),
                              np.full_like(ring, top_z)))
    cv2.polylines(picture, [project(circle, T_base_cam, node.K, node.D).astype(np.int32)], True, YELLOW, 2)
    c_px = project([center], T_base_cam, node.K, node.D)[0].astype(int)
    cv2.drawMarker(picture, tuple(c_px), RED, cv2.MARKER_CROSS, 20, 2)
    cv2.putText(picture, 'lid centre', tuple(c_px + [10, -10]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

    if not 0.6 * LID_RADIUS <= radius <= 1.5 * LID_RADIUS:
        return None, None, f'fitted radius {1000 * radius:.0f} mm, expected ~{1000 * LID_RADIUS:.0f} mm', picture
    if np.linalg.norm(center[:2] - tag_xyz[:2]) > radius + 0.005:
        return None, None, 'the fitted lid does not contain the tag', picture
    return center, radius, None, picture


def lid_correction(node, midpoint, recorder):
    """At the pre-grasp: measure the lid centre up close; show it; on yes return the corrected
    (pre, grasp, pull_out): straight down onto the measured centre, finger pads GRASP_BELOW_LID below
    the measured lid top. None = keep the original targets."""
    time.sleep(0.5)                                      # let the arm and the image settle
    center, radius, problem, picture = find_lid(node, node.last_tag_xyz)
    os.makedirs(LID_DIR, exist_ok=True)
    path = os.path.join(LID_DIR, time.strftime('lid_%Y%m%d_%H%M%S.png'))
    if center is not None:
        T_base_cam = node.lookup(BASE_FRAME, node.camera_frame)
        m_px = project([[midpoint[0], midpoint[1], center[2]]], T_base_cam, node.K, node.D)[0].astype(int)
        cv2.drawMarker(picture, tuple(m_px), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(picture, 'tag estimate', tuple(m_px + [10, 15]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(path, picture)
    recorder.show_still(picture)
    popup = not AUTO                  # shown live unless nobody is at the keyboard; the video gets it too
    if popup:
        cv2.imshow('lid check', picture)
        cv2.waitKey(500)
    if center is None:
        print(f'Lid correction: {problem} (image: {os.path.normpath(path)}). Keeping the tag-based grasp.')
        return None
    shift = center[:2] - midpoint[:2]
    print(f'Lid centre (base_link) {np.round(center, 4)}, radius {1000 * radius:.1f} mm; '
          f'tag-based estimate was {1000 * np.linalg.norm(shift):.1f} mm off '
          f'(dx {1000 * shift[0]:+.1f}, dy {1000 * shift[1]:+.1f}); lid top z {center[2]:.4f}. '
          f'Image: {os.path.normpath(path)}')
    use = ask('Does the yellow circle sit on the lid? Use the measured centre?')
    if popup:
        cv2.destroyWindow('lid check')
    if not use:
        return None
    reach = float(node.lookup(EE_FRAME, TWIST_FRAME)[2, 3])     # end_effector_link -> finger-pad centre
    R_cur = node.lookup(BASE_FRAME, EE_FRAME)[:3, :3]
    a = np.array([0.0, 0.0, -1.0])                              # straight down: the lid is flat
    y = np.array([R_cur[0, 1], R_cur[1, 1], 0.0])               # keep the wrist's closing direction
    y /= np.linalg.norm(y)
    R = np.column_stack((np.cross(y, a), y, a))
    q = matrix_to_quaternion(R)
    pad = center + [0.0, 0.0, -GRASP_BELOW_LID]
    grasp = pad - reach * a
    pre = grasp - PREGRASP * a
    print(f'Corrected grasp: finger-pad centre at {np.round(pad, 4)} (reach {1000 * reach:.0f} mm)')
    return [*pre, *q], [*grasp, *q], [*pre, *q]


# ============================================================================== stage 4: holder
def measure_tag_depth(node, tag_id, frames=MEASURE_FRAMES, timeout=5.0):
    """Pose of tag `tag_id` in base_link, from a still arm, without needing its printed size: the
    rotation from its corners (solvePnP's rotation doesn't depend on the size), the position from the
    aligned depth image along the ray to its centre. -> 4x4 or None."""
    unit_square = TAG_CORNERS / TAG_SIZE            # any size gives the same rotation
    T_base_cam = node.lookup(BASE_FRAME, node.camera_frame)
    poses, last, end = [], node.frame_count, time.monotonic() + timeout
    while len(poses) < frames and time.monotonic() < end:
        if node.frame_count == last or node.depth is None:
            time.sleep(0.01)
            continue
        last, image, depth = node.frame_count, node.image.copy(), node.depth
        corners, ids, _ = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if ids is None or tag_id not in ids.ravel():
            continue
        px = corners[list(ids.ravel()).index(tag_id)].reshape(4, 2).astype(np.float64)
        ok, rvec, _ = cv2.solvePnP(unit_square, px, node.K, node.D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, px.astype(np.int32), 1)
        z = depth[(mask > 0) & (depth > 0)]
        if not ok or len(z) < 20:
            continue
        ray = cv2.undistortPoints(px.mean(axis=0).reshape(1, 1, 2), node.K, node.D).ravel()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = cv2.Rodrigues(rvec)[0], np.array([ray[0], ray[1], 1.0]) * np.median(z)
        poses.append(T_base_cam @ T)
    if len(poses) < max(3, frames // 2):
        return None
    T, _ = average_pose(poses)
    return T


def stand_seat(T_base_tag):
    """Pocket floor centre of the holder (base_link) from its tag; the tag's yaw only (it lies flat)."""
    tag_x = T_base_tag[:3, 0]
    xh = np.array([tag_x[0], tag_x[1], 0.0]) / np.linalg.norm(tag_x[:2])
    yh = np.cross([0.0, 0.0, 1.0], xh)
    a = np.radians(STAND_TAG_ANGLE)
    return T_base_tag[:3, 3] + STAND_TAG_X * (np.cos(a) * xh + np.sin(a) * yh) + [0.0, 0.0, STAND_SEAT_ABOVE_TAG]


def check_stand(node, seat, recorder):
    """Draw the predicted pocket rim on the camera image, save it, show it (video / pop-up), ask."""
    T_base_cam = node.lookup(BASE_FRAME, node.camera_frame)
    ring = np.linspace(0, 2 * np.pi, 60)
    rim = np.column_stack((seat[0] + STAND_POCKET_MOUTH / 2 * np.cos(ring),
                           seat[1] + STAND_POCKET_MOUTH / 2 * np.sin(ring),
                           np.full_like(ring, seat[2] + STAND_RIM_ABOVE_SEAT)))
    picture = node.image.copy()
    px = project(rim, T_base_cam, node.K, node.D).astype(np.int32)
    cv2.polylines(picture, [px], True, (255, 255, 0), 2)
    cv2.putText(picture, 'holder pocket (predicted)', tuple(px.min(axis=0) + [0, -8]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    os.makedirs(STAND_DIR, exist_ok=True)
    path = os.path.join(STAND_DIR, time.strftime('stand_%Y%m%d_%H%M%S.png'))
    cv2.imwrite(path, picture)
    recorder.show_still(picture)
    print(f'Holder pocket floor (base_link) {np.round(seat, 4)}. Image: {os.path.normpath(path)}')
    popup = not AUTO                  # shown live unless nobody is at the keyboard; the video gets it too
    if popup:
        cv2.imshow('holder check', picture)
        cv2.waitKey(500)
    ok = ask('Does the cyan circle sit on the holder\'s cup?')
    if popup:
        cv2.destroyWindow('holder check')
    return ok


def place_in_stand(node, mover, stage, recorder, grasp_depth):
    """Carry the held shaker over the holder's pocket (level, straight line), lower it in, let go a few
    mm above the pocket floor, back off upwards. grasp_depth: finger-pad centre below the lid top."""
    if node.stand_T is None:
        with open(VIEWS_FILE) as f:
            look = [float(v) for v in yaml.safe_load(f)['views'][STAND_LOOK_VIEW]]
        if not ask(f'Holder (tag {STAND_TAG_ID}) not seen yet. Move to view {STAND_LOOK_VIEW} to look for it?'):
            return
        stage('looking for the holder')
        if not mover.move_to(look):
            print('Could not reach the look pose.')
            return
        time.sleep(1.0)
        node.stand_T = measure_tag_depth(node, STAND_TAG_ID)
        if node.stand_T is None:
            print(f'Holder tag {STAND_TAG_ID} not found (or no depth on it); still holding the shaker.')
            return
    seat = stand_seat(node.stand_T)
    stage('checking the holder')
    if not check_stand(node, seat, recorder):
        print('Holder position not confirmed; still holding the shaker.')
        return

    R_cur = node.lookup(BASE_FRAME, EE_FRAME)[:3, :3]
    a = np.array([0.0, 0.0, -1.0])                               # straight down
    y = np.array([R_cur[0, 1], R_cur[1, 1], 0.0])                # keep the wrist's angle: no joint_6 spin
    y /= np.linalg.norm(y)
    q = matrix_to_quaternion(np.column_stack((np.cross(y, a), y, a)))
    pad_place = seat + [0.0, 0.0, STAND_DROP + SHAKER_HEIGHT - grasp_depth]   # jar bottom STAND_DROP up
    place = pad_place - FINGERTIP * a
    above = place + [0.0, 0.0, STAND_CLEARANCE - STAND_DROP]
    place, above = [*place, *q], [*above, *q]

    stage('carrying it to the holder')
    if not mover.move_cartesian(above):
        if not ask('No straight-line carry to the holder. Plan a free move instead (watch the hanging shaker)?'):
            return
        if not mover.move_rrtconnect(above):
            print('Could not get over the holder; still holding the shaker.')
            return
    if not ask(f'Over the holder. Lower it {100 * (STAND_CLEARANCE - STAND_DROP):.0f} cm into the pocket?'):
        return
    stage('lowering it into the holder')
    if not mover.move_cartesian(place):
        print('Lowering failed; still holding the shaker above the holder.')
        return
    stage('letting go')
    if not node.set_gripper(GRIPPER_OPEN):
        print('Gripper did not report opening; not backing off.')
        return
    time.sleep(0.5)
    stage('backing off')
    if not mover.move_cartesian(above):
        print('Back-off failed.')
    print('Shaker placed in the holder.')


def put_down(node, mover, grasp, pull_out, stage):
    """Put the held shaker back where it was picked up: hold it up for HOLD_SEC, go back down the same
    straight line to PLACE_DROP short of the grasp pose (the jar doesn't jam into its pocket), let go,
    back out the same way."""
    stage('holding it up')
    time.sleep(HOLD_SEC)
    if not ask('Put it back down where it was?'):
        return
    approach = np.subtract(grasp[:3], pull_out[:3])
    approach /= np.linalg.norm(approach)                        # pull-out -> grasp direction
    place = [*(np.array(grasp[:3]) - PLACE_DROP * approach), *grasp[3:]]
    stage('putting it back down')
    if not mover.move_cartesian(place):
        print('Could not go back down; still holding it.')
        return
    stage('letting go')
    if not node.set_gripper(GRIPPER_OPEN):
        print('Gripper did not report opening; not backing out.')
        return
    time.sleep(0.5)                                             # let the jar settle before the fingers leave
    stage('backing out')
    if not mover.move_cartesian(pull_out):
        print('Back-out failed.')


def run_grasp(node, pre, grasp, pull_out, midpoint, recorder):
    """Pre-grasp (planned), open, straight in, close, straight out. Asks before each motion
    (--auto: answers yes itself)."""
    stage = recorder.set_stage
    mover = ArmMover(node, ee_link=EE_FRAME, base_frame=BASE_FRAME, speed=0.1, motion='rrtconnect')
    if not ask('Move to the pre-grasp with MoveIt?'):
        return
    stage('moving to the pre-grasp')
    if not mover.move_to(pre):
        print('Pre-grasp failed (see the MoveIt message above).')
        return
    if LID_CORRECTION:
        stage('measuring the lid up close')
        corrected = lid_correction(node, midpoint, recorder)
        if corrected is None and AUTO:
            print('Auto mode: no lid measurement, so no grasp (the tag-based grasp alone misses).')
            return
        node.used_lid_correction = corrected is not None
        if corrected is not None:
            pre, grasp, pull_out = corrected
            if not ask(f'Move above the measured lid centre (straight down, {PREGRASP * 100:.0f} cm up)?'):
                return
            stage('moving above the measured lid centre')
            if not mover.move_to(pre):
                print('Could not reach the corrected pre-grasp (see the MoveIt message above).')
                return
    stage('opening the gripper')
    if not node.set_gripper(GRIPPER_OPEN):
        print('Could not open the gripper; stopping here.')
        return
    if not ask(f'At the pre-grasp. Go in {PREGRASP * 100:.0f} cm along the approach (straight line)?'):
        return
    stage('going in')
    if not mover.move_cartesian(grasp):
        print('Straight-line approach failed; staying where it is.')
        return
    if not ask('Close the gripper?'):
        return
    stage('closing the gripper')
    if not node.set_gripper(GRIPPER_CLOSED) and not ask('The gripper did not report closing. Pull out anyway?'):
        return
    if not ask('Pull out along the axis?'):
        return
    stage('pulling out')
    if not mover.move_cartesian(pull_out):
        print('Pull-out failed; still holding it.')
        return
    if PLACE_TARGET == 'stand':
        grasp_depth = GRASP_BELOW_LID if (LID_CORRECTION and node.used_lid_correction) else -GRASP_IN_TAG[2]
        place_in_stand(node, mover, stage, recorder, grasp_depth)
    elif PLACE_TARGET == 'back' and PUT_DOWN:
        put_down(node, mover, grasp, pull_out, stage)
    stage('done')
    time.sleep(1.0)                                           # a moment of the result in the video
    print('Grasp sequence finished.')


# ============================================================================== main
def run(node, recorder):
    if AUTO:
        print('AUTO mode: servoing starts as soon as the tag is seen and every step runs without asking. '
              'Keep a hand on the e-stop.')
    else:
        print('Click the image window, then press: s = servo on/off, r = record on/off, q = quit')
    while rclpy.ok() and (node.image is None or node.K is None):
        print('Waiting for camera image and camera_info ...')
        time.sleep(0.5)
    T_tool_cam = node.wait_for_transform(TWIST_FRAME, node.camera_frame)
    if T_tool_cam is None:
        print('No hand-eye transform on TF; is the static_transform_publisher running?')
        return

    samples = run_servo_stage(node, T_tool_cam, recorder)
    recorder.set_stage('tag measured')
    if samples and PLACE_TARGET == 'stand':
        node.stand_T = measure_tag_depth(node, STAND_TAG_ID)     # arm still at the good view
        print(f'Holder tag {STAND_TAG_ID}: ' + ('measured' if node.stand_T is not None
                                                 else f'not in view here; will look from view {STAND_LOOK_VIEW}'))
    recorder.start_live(node)                                 # --record: keep recording after servoing
    if not samples:
        print('No tag pose measured; no grasp.')
        return

    T_cam_tag, spread_mm = average_pose(samples)
    T_base_tag = node.lookup(BASE_FRAME, node.camera_frame) @ T_cam_tag
    # With the lid correction the first pre-grasp only has to show the lid to the camera: look straight
    # down from above the tag estimate. A small tag's lid normal is often tens of degrees off, and
    # following it puts the pre-grasp at a strange tilted pose.
    approach = 'vertical' if LID_CORRECTION else APPROACH
    pre, grasp, pull_out = grasp_poses(T_base_tag, node.lookup(BASE_FRAME, EE_FRAME)[:3, :3], approach)
    node.last_tag_xyz = T_base_tag[:3, 3]                      # where to look for the lid up close
    tilt = np.degrees(np.arccos(np.clip(T_base_tag[2, 2], -1.0, 1.0)))
    np.set_printoptions(precision=4, suppress=True)
    print(f'\nTag pose from {len(samples)} frames (max deviation {spread_mm:.1f} mm):\n{T_base_tag}')
    print(f'Lid normal {tilt:.0f} deg from vertical (measured); first pre-grasp approach: {approach}')
    print(f'Grasp midpoint (base_link): {grasp_midpoint(T_base_tag, approach)}')
    print('end_effector_link targets [x y z qx qy qz qw]:')
    for name, pose in (('pre-grasp', pre), ('grasp', grasp), ('pull-out', pull_out)):
        print(f'  {name:10s} {np.round(pose, 4).tolist()}')

    if DO_GRASP:
        cv2.destroyAllWindows()
        run_grasp(node, pre, grasp, pull_out, grasp_midpoint(T_base_tag, approach), recorder)


def main():
    global AUTO, RECORD_ALL
    parser = argparse.ArgumentParser(description='AprilTag servo + lid-corrected grasp.')
    parser.add_argument('--auto', action='store_true',
                        help='no human in the loop: start servoing by itself, answer every prompt yes')
    parser.add_argument('--record', action='store_true',
                        help='record the camera for the whole run into one video (no pop-up windows)')
    args = parser.parse_args()
    AUTO, RECORD_ALL = args.auto, args.record

    # No rclpy Ctrl+C handler: ROS must stay up so the stop command can still be sent on Ctrl+C.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TagServoNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    recorder = Recorder()
    try:
        run(node, recorder)
    except KeyboardInterrupt:
        print('Interrupted.')
    finally:
        recorder.stop()                      # saves the video whatever happened
        cv2.destroyAllWindows()
        executor.shutdown()                  # stop spinning before tearing ROS down, or the process aborts
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
