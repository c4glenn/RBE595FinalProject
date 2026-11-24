"""ROS I/O for the real task: RealSense colour + aligned depth + camera_info, TF, the gripper action,
and the finger joint. Spun in a background thread (HW7 style), so the task code can block."""
import threading
import time
import warnings

import cv2
import numpy as np
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from helpers.geometry import transform_to_matrix
from helpers.robot_motion import wait_for


class TaskIO(Node):
    def __init__(self, cfg):
        super().__init__('spice_task')
        self.cfg = cfg
        t = cfg['topics']
        self.base_frame, self.ee_link = cfg['base_frame'], cfg['ee_link']
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self.image = self.depth = self.K = self.D = None
        self.camera_frame = None
        self.image_count = self.depth_count = 0
        self.finger = None
        self.last_grip = None                                         # last GripperCommand result
        self.joints = {}                                             # name -> position (rad)
        self.create_subscription(Image, t['image'], self._on_image, qos_profile_sensor_data)
        self.create_subscription(Image, t['depth'], self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, t['info'], self._on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, t['joint_states'], self._on_joints, 10)
        self.gripper_client = ActionClient(self, GripperCommand, cfg['gripper']['action'])
        # For RViz (config/spice.rviz): the last annotated picture and the 3D markers, latched so a
        # late RViz still gets them.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.debug_pub = self.create_publisher(Image, '/spice/debug_image', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/spice/markers', latched)
        self.marker_groups = {}
        self.mount_pub = self.create_publisher(String, '/spice/hand_eye_mount', latched)
        self.show_camera()

    # ------------------------------------------------------------------ callbacks
    def _on_image(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img.copy()
        with self.lock:
            self.image, self.camera_frame = img, msg.header.frame_id
            self.image_count += 1

    def _on_depth(self, msg):
        if msg.encoding == '32FC1':
            d = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).astype(np.float32)
        else:                                                        # 16UC1, mm
            d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) * 0.001
        d[d <= 0] = np.nan
        with self.lock:
            self.depth = d
            self.depth_count += 1

    def _on_info(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.array(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def _on_joints(self, msg):
        self.joints = {**self.joints, **dict(zip(msg.name, (float(p) for p in msg.position)))}
        name = self.cfg['gripper']['finger_joint']
        if name in msg.name:
            i = msg.name.index(name)
            if i < len(msg.position):
                self.finger = float(msg.position[i])

    # ------------------------------------------------------------------ TF
    def lookup(self, target, source):
        return transform_to_matrix(self.tf_buffer.lookup_transform(target, source, Time()).transform)

    def ee_pose(self):
        """4x4 base_link <- end_effector_link."""
        return self.lookup(self.base_frame, self.ee_link)

    def camera_in_ee(self):
        """Camera optical centre in end_effector_link (hand-eye TF + the RealSense's own TF)."""
        return self.lookup(self.ee_link, self.camera_frame)[:3, 3]

    # ------------------------------------------------------------------ camera mount (hand-eye)
    def mount_name(self):
        """Which of hand_eye_mounts the hand-eye in use is ('outside', 'inside'), or None."""
        he = self.cfg['hand_eye']
        for name, m in (self.cfg.get('hand_eye_mounts') or {}).items():
            if np.allclose(m['translation'], he['translation']) and np.allclose(m['rotation'], he['rotation']):
                return name
        return None

    def set_mount(self, name, timeout=5.0):
        """Switch the hand-eye TF to mount `name` (helpers/hand_eye_tf.py republishes it) and wait
        until TF has it. True if it took."""
        he = self.cfg['hand_eye_mounts'][name]
        self.mount_pub.publish(String(data=name))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                t = self.lookup(he['parent'], he['child'])[:3, 3]
                if np.allclose(t, he['translation'], atol=1e-4):
                    self.cfg['hand_eye'] = he
                    time.sleep(0.2)
                    return True
            except Exception:                                        # noqa: BLE001
                pass
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ camera
    def wait_ready(self, timeout=30.0):
        """Camera image, depth, intrinsics and the base -> camera TF all available."""
        end = time.monotonic() + timeout
        why = 'camera'
        while time.monotonic() < end:
            if self.image is None or self.K is None:
                why = f"no colour image / camera_info on {self.cfg['topics']['image']}"
            elif self.depth is None:
                why = f"no depth on {self.cfg['topics']['depth']} (align_depth enabled?)"
            else:
                try:
                    self.lookup(self.base_frame, self.camera_frame)
                    return True, ''
                except Exception as e:                               # noqa: BLE001
                    why = f'no TF {self.base_frame} -> {self.camera_frame} ({e}); hand-eye TF running?'
            time.sleep(0.1)
        return False, why

    def frames(self, n, timeout=5.0):
        """n fresh (image, depth, T_base_cam) triples from a still arm."""
        out, last, end = [], self.image_count, time.monotonic() + timeout
        while len(out) < n and time.monotonic() < end:
            with self.lock:
                fresh = self.image_count != last and self.depth is not None
                if fresh:
                    last = self.image_count
                    image, depth, frame = self.image.copy(), self.depth.copy(), self.camera_frame
            if fresh:
                out.append((image, depth, self.lookup(self.base_frame, frame)))
            else:
                time.sleep(0.01)
        return out

    def snapshot(self, n_depth=5, timeout=5.0):
        """(image, median depth over n_depth frames, T_base_cam) from a still arm, or None."""
        frames = self.frames(n_depth, timeout)
        if not frames:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')                         # all-NaN pixels stay NaN
            depth = np.nanmedian(np.stack([f[1] for f in frames]), axis=0)
        return frames[-1][0], depth, frames[-1][2]

    # ------------------------------------------------------------------ gripper
    def set_gripper(self, position, timeout=15.0):
        """True if the gripper reached the goal or stalled on something."""
        g = self.cfg['gripper']
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"gripper action {g['action']} not available")
            return False
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = float(position), float(g['max_effort'])
        handle = wait_for(self.gripper_client.send_goal_async(goal), timeout=5.0)
        if handle is None or not handle.accepted:
            self.get_logger().error('gripper goal rejected')
            return False
        res = wait_for(handle.get_result_async(), timeout=timeout)
        if res is None:
            self.get_logger().error('gripper gave no result')
            return False
        r = res.result
        self.get_logger().info(f'gripper: asked {position:.2f}, at {r.position:.3f}, '
                               f'reached={r.reached_goal}, stalled={r.stalled}')
        self.last_grip = {'position': float(r.position), 'reached': bool(r.reached_goal), 'stalled': bool(r.stalled)}
        return bool(r.reached_goal or r.stalled)


    # ------------------------------------------------------------------ RViz
    def publish_picture(self, bgr):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.camera_frame or ''
        msg.height, msg.width = bgr.shape[:2]
        msg.encoding, msg.step = 'bgr8', bgr.shape[1] * 3
        msg.data = np.ascontiguousarray(bgr).tobytes()
        self.debug_pub.publish(msg)

    def show_camera(self):
        """A D405-sized box (42 x 42 x 23 mm) and its viewing direction, fixed to the camera frame
        of the hand-eye TF: where the code thinks the camera is. It should sit on the real camera
        in RViz; if it does not, hand_eye in config/real.yaml is not this robot's mount."""
        frame = self.cfg['hand_eye']['child']
        self.show_markers('camera (hand-eye)', [
            {'type': 'box', 'frame': frame, 'center': (-0.0115, 0.0, 0.0), 'size': (0.023, 0.042, 0.042),
             'rgba': (0.9, 0.9, 0.9, 0.8)},
            {'type': 'arrow', 'frame': frame, 'start': (0.0, 0.0, 0.0), 'end': (0.08, 0.0, 0.0),
             'rgba': (1.0, 0.2, 0.2, 1.0)},
            {'type': 'text', 'frame': frame, 'pos': (0.0, 0.0, 0.04), 'text': 'camera (hand-eye)',
             'rgba': (1, 1, 1, 1)}])

    def show_markers(self, group, shapes):
        """Replace marker group `group` (a namespace) with `shapes`: dicts with
          type 'cylinder' (center, radius, height), 'box' (center, size, yaw), 'arrow' (start, end),
          'text' (pos, text); and rgba. Coordinates are in base_link, or in shape['frame'] (the
          marker then moves with that frame). An empty list removes the group."""
        self.marker_groups[group] = shapes
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        stamp = self.get_clock().now().to_msg()
        for ns, items in self.marker_groups.items():
            for k, sh in enumerate(items):
                m = Marker()
                m.header.frame_id, m.header.stamp = sh.get('frame', self.base_frame), stamp
                m.frame_locked = 'frame' in sh
                m.ns, m.id, m.action = ns, k, Marker.ADD
                m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in sh.get('rgba', (1, 1, 0, 1)))
                m.pose.orientation.w = 1.0
                t = sh['type']
                if t == 'cylinder':
                    m.type = Marker.CYLINDER
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(c) for c in sh['center'])
                    m.scale.x = m.scale.y = 2 * float(sh['radius'])
                    m.scale.z = float(sh.get('height', 0.002))
                elif t == 'box':
                    m.type = Marker.CUBE
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(c) for c in sh['center'])
                    yaw = float(sh.get('yaw', 0.0))
                    m.pose.orientation.z, m.pose.orientation.w = np.sin(yaw / 2), np.cos(yaw / 2)
                    m.scale.x, m.scale.y, m.scale.z = (float(c) for c in sh['size'])
                elif t == 'arrow':
                    m.type = Marker.ARROW
                    m.points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in (sh['start'], sh['end'])]
                    m.scale.x, m.scale.y, m.scale.z = 0.006, 0.012, 0.02
                else:
                    m.type = Marker.TEXT_VIEW_FACING
                    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(c) for c in sh['pos'])
                    m.text = sh['text']
                    m.scale.z = 0.02
                arr.markers.append(m)
        self.marker_pub.publish(arr)


def spin_in_background(node):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor, thread
