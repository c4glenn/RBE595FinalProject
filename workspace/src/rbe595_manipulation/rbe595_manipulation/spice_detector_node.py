"""ROS node: finds the spice bottles and the bowl with the wrist camera, using colour
thresholding + Hough circles (rbe595_manipulation/vision/circle_detector.py), and numbers the
bottles so one can be picked by number.

It takes one snapshot of the workspace: moves the arm to the observe pose
(config/spice_detection.yaml), grabs a wrist camera image, looks up where the camera was on
TF, and runs the detector on that image. The result is kept, and republished every second,
until the next snapshot; so the bottle numbers stay put while the arm (and its camera) is off
picking. A snapshot is taken at startup (snapshot_on_start) and on request:
  ros2 service call /spice_detector/snapshot std_srvs/srv/Trigger
Only take one while no pick is running: it moves the arm.

Subscribes:
  <camera.image_topic>        sensor_msgs/Image
  <camera.info_topic>         sensor_msgs/CameraInfo
Publishes (all poses in the world frame, at the cap top / rim top centre), every second:
  ~/annotated_image           sensor_msgs/Image        snapshot with numbered detections
  ~/bottles                   geometry_msgs/PoseArray  poses[i] is bottle number i
  ~/bowl                      geometry_msgs/PoseStamped
  ~/selected_bottle           geometry_msgs/PoseStamped  bottle number `selected_bottle`
Services:
  ~/snapshot                  std_srvs/Trigger         move to the observe pose, detect again
Parameters:
  config             path to the YAML config (default: the installed spice_detection.yaml)
  selected_bottle    bottle number to pick, -1 for none; change it at runtime with
                     ros2 param set /spice_detector selected_bottle 3
  world_frame        frame_id for the published poses (default "world")
  snapshot_on_start  take a snapshot as soon as MoveIt and the planning scene are up (true)
  save_dir           where each snapshot's image, annotated image and camera pose are saved
                     (default ~/.ros/spice_snapshot; empty: not saved)
  show_window        also show the annotated image in an OpenCV window (default false)
"""

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene
from rcl_interfaces.msg import SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from rbe595_manipulation._orientation import _matrix_to_rpy
from rbe595_manipulation._wait import wait_for_future
from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.poses import ARM_BASE_WORLD_XYZ
from rbe595_manipulation.vision.circle_detector import (
    CameraModel,
    SpiceCircleDetector,
    format_table,
    load_config,
)

# The pick/place server adds these to the planning scene at startup; the arm is not moved
# before they are there.
SCENE_OBJECT = "table_top"


def _pose(xyz):
    p = Pose()
    p.position.x, p.position.y, p.position.z = xyz
    p.orientation.w = 1.0
    return p


def _quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def look_at_optical(position, target, up):
    """Optical-frame axes (columns: x right, y down, z forward) of a camera at `position`
    looking at `target`, with world direction `up` towards the top of the image."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    down = -np.asarray(up, float)
    y = down - np.dot(down, z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


class SpiceDetectorNode(Node):
    def __init__(self):
        super().__init__("spice_detector")
        self.declare_parameter("config", "")
        self.declare_parameter("selected_bottle", -1)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("snapshot_on_start", True)
        self.declare_parameter("save_dir", os.path.expanduser("~/.ros/spice_snapshot"))
        self.declare_parameter("show_window", False)

        self._cfg = load_config(self.get_parameter("config").value or None)
        self._cam_cfg = self._cfg["camera"]
        self._selected = self.get_parameter("selected_bottle").value
        self._frame = self.get_parameter("world_frame").value
        self._save_dir = self.get_parameter("save_dir").value
        self._show = self.get_parameter("show_window").value
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._busy = threading.Lock()  # one snapshot at a time
        self._image = None  # newest wrist camera Image
        self._info = None  # newest CameraInfo
        # Last snapshot: (image, detector, result, stamp)
        self._snapshot = None

        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._arm = ArmMoveClient(self)
        self._get_scene = self.create_client(GetPlanningScene, "/get_planning_scene")

        self._pub_image = self.create_publisher(Image, "~/annotated_image", 1)
        self._pub_bottles = self.create_publisher(PoseArray, "~/bottles", 10)
        self._pub_bowl = self.create_publisher(PoseStamped, "~/bowl", 10)
        self._pub_selected = self.create_publisher(PoseStamped, "~/selected_bottle", 10)

        sensors = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            CameraInfo, self._cam_cfg["info_topic"], self._on_info, 1, callback_group=sensors
        )
        self.create_subscription(
            Image, self._cam_cfg["image_topic"], self._on_image, qos_profile_sensor_data,
            callback_group=sensors,
        )
        # Snapshots block (arm moves) while the other groups keep the node going.
        slow = ReentrantCallbackGroup()
        self.create_service(Trigger, "~/snapshot", self._on_snapshot, callback_group=slow)
        if self.get_parameter("snapshot_on_start").value:
            self._start_timer = self.create_timer(1.0, self._initial_snapshot, callback_group=slow)
        self.create_timer(1.0, self._publish, callback_group=MutuallyExclusiveCallbackGroup())
        self.add_on_set_parameters_callback(self._on_params)

    # --- inputs ---

    def _on_params(self, params):
        for p in params:
            if p.name == "selected_bottle":
                self._selected = p.value
                self.get_logger().info(f"selected bottle {p.value}")
        return SetParametersResult(successful=True)

    def _on_info(self, msg):
        self._info = msg

    def _on_image(self, msg):
        self._image = msg

    # --- snapshot ---

    def _initial_snapshot(self):
        self._start_timer.cancel()
        ok, message = self.snapshot()
        (self.get_logger().info if ok else self.get_logger().error)(message)

    def _on_snapshot(self, request, response):
        response.success, response.message = self.snapshot()
        return response

    def snapshot(self):
        """Move to the observe pose, grab an image there and detect. Returns (ok, message)."""
        if not self._busy.acquire(blocking=False):
            return False, "a snapshot is already being taken"
        try:
            return self._snapshot_locked()
        finally:
            self._busy.release()

    def _snapshot_locked(self):
        log = self.get_logger()
        if not self._wait_for_planning_scene():
            return False, "planning scene not ready; is pick_place_spice.launch.py running?"
        pose = self._observe_ee_pose()
        if pose is None:
            return False, "camera mount not on TF; is the sim (robot_state_publisher) running?"
        xyz, rpy = pose
        log.info("moving to the observe pose for a snapshot")
        if not self._arm.move_to(xyz, rpy):
            return False, "could not move the arm to the observe pose"
        time.sleep(self._cfg["observe"]["settle_s"])

        # An image taken after the arm stopped, and the camera pose at that time.
        stopped = self.get_clock().now()
        deadline = time.monotonic() + 10.0
        while True:
            msg, info = self._image, self._info
            if msg is not None and info is not None and Time.from_msg(msg.header.stamp) >= stopped:
                break
            if time.monotonic() > deadline:
                return False, f"no image on {self._cam_cfg['image_topic']}"
            time.sleep(0.05)
        camera = self._camera_at(msg.header.stamp, info)
        if camera is None:
            return False, "no camera pose on TF for the snapshot"

        image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        detector = SpiceCircleDetector(self._cfg, camera)
        result = detector.detect(image)
        with self._lock:
            self._snapshot = (image, detector, result, msg.header.stamp)
        self._save(image, detector, result, camera)
        self._publish()
        log.info(f"snapshot detections:\n{format_table(result)}")
        return True, f"{len(result.bottles)} bottles, bowl {'found' if result.bowl else 'not found'}"

    def _wait_for_planning_scene(self, timeout_s=120.0):
        """Wait until the pick/place server has put the table etc. in the planning scene."""
        deadline = time.monotonic() + timeout_s
        next_log = 0.0
        while time.monotonic() < deadline:
            if self._get_scene.wait_for_service(timeout_sec=1.0):
                request = GetPlanningScene.Request()
                request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
                future = self._get_scene.call_async(request)
                if wait_for_future(future, 5.0) and future.result() is not None:
                    names = [o.id for o in future.result().scene.world.collision_objects]
                    if SCENE_OBJECT in names:
                        return True
            if time.monotonic() > next_log:
                self.get_logger().info("waiting for MoveIt and the pick/place server's planning scene")
                next_log = time.monotonic() + 10.0
            time.sleep(1.0)
        return False

    def _lookup(self, target, source, stamp=None, timeout_s=2.0):
        """(rotation, translation) of `source` in `target`, or None."""
        try:
            tf = self._tf.lookup_transform(
                target, source, Time.from_msg(stamp) if stamp else Time(),
                timeout=Duration(seconds=timeout_s),
            )
        except Exception as exc:  # noqa: BLE001 -- any TF failure: caller reports it
            self.get_logger().warn(f"TF {target} -> {source}: {exc}")
            return None
        t = tf.transform.translation
        return _quat_to_matrix(tf.transform.rotation), np.array([t.x, t.y, t.z])

    def _observe_ee_pose(self):
        """End-effector (world xyz, rpy) that puts the camera at the observe pose."""
        obs = self._cfg["observe"]
        mount = self._lookup(self._cam_cfg["end_effector_frame"], self._cam_cfg["optical_frame"], timeout_s=10.0)
        if mount is None:
            return None
        r_ee_opt, p_ee_opt = mount
        r_world_opt = look_at_optical(obs["camera_position"], obs["look_at"], obs["image_up"])
        r_world_ee = r_world_opt @ r_ee_opt.T
        p_ee = np.asarray(obs["camera_position"], float) - r_world_ee @ p_ee_opt
        return tuple(float(v) for v in p_ee), _matrix_to_rpy(r_world_ee.tolist())

    def _camera_at(self, stamp, info):
        pose = self._lookup(self._cam_cfg["base_frame"], self._cam_cfg["optical_frame"], stamp)
        if pose is None:
            return None
        r, p = pose
        return CameraModel(info.k, r, p + np.asarray(ARM_BASE_WORLD_XYZ))

    def _save(self, image, detector, result, camera):
        if not self._save_dir:
            return
        d = Path(self._save_dir)
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "snapshot.png"), image)
        cv2.imwrite(str(d / "snapshot_annotated.png"), detector.annotate(image, result, self._selected))
        with open(d / "snapshot_camera.yaml", "w") as f:
            yaml.safe_dump(camera.to_dict(), f)

    # --- output ---

    def _publish(self):
        with self._lock:
            snap = self._snapshot
        if snap is None:
            return
        image, detector, result, _ = snap
        # Stamped now: the poses are world-fixed, and consumers judge freshness by arrival.
        stamp = self.get_clock().now().to_msg()

        bottles = PoseArray()
        bottles.header.frame_id = self._frame
        bottles.header.stamp = stamp
        bottles.poses = [_pose(b.world_xyz) for b in result.bottles]
        self._pub_bottles.publish(bottles)
        if result.bowl:
            self._pub_bowl.publish(self._stamped(result.bowl.world_xyz, stamp))

        selected = next((b for b in result.bottles if b.number == self._selected), None)
        if selected:
            self._pub_selected.publish(self._stamped(selected.world_xyz, stamp))
        elif self._selected >= 0:
            self.get_logger().warn(
                f"selected bottle {self._selected} not found "
                f"({len(result.bottles)} bottles in the snapshot)",
                throttle_duration_sec=5.0,
            )

        annotated = detector.annotate(image, result, self._selected)
        out = self._bridge.cv2_to_imgmsg(annotated, "bgr8")
        out.header.stamp = stamp
        out.header.frame_id = self._cam_cfg["optical_frame"]
        self._pub_image.publish(out)
        if self._show:
            cv2.imshow("spice_detector", annotated)
            cv2.waitKey(1)

    def _stamped(self, xyz, stamp):
        msg = PoseStamped()
        msg.header.frame_id = self._frame
        msg.header.stamp = stamp
        msg.pose = _pose(xyz)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SpiceDetectorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
