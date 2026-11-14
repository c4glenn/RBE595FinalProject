"""Finds tag36h11 AprilTags in the wrist camera's images and returns where each one is, in
rbe595_sim's world frame (see poses.py).

Detection is OpenCV's aruco module with its built-in tag36h11 dictionary; each tag's pose comes
from its four corners (solvePnP, IPPE_SQUARE) and the camera intrinsics on camera_info, and is
taken into the world through /tf (the camera's optical frame on the arm, at the moment the
image was taken). Works the same on the real D405 (realsense2_camera) as on the sim camera, as
long as /tf has the camera mounted on the arm.

Every image it looks at is republished with the tags it found drawn on (debug_image_topic).
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from rbe595_manipulation.poses import ARM_BASE_WORLD_XYZ


@dataclass
class TagPose:
    """A tag's centre (world frame, m) and orientation: columns are the tag's x (right on the
    tag image), y (up on the tag image) and z (out of its face) axes in the world frame."""

    position: np.ndarray
    rotation: np.ndarray


def _quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _image_to_bgr(msg: Image) -> np.ndarray:
    channels = {"rgb8": 3, "bgr8": 3, "mono8": 1, "rgba8": 4, "bgra8": 4}[msg.encoding]
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = img[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    conversion = {
        "rgb8": cv2.COLOR_RGB2BGR,
        "rgba8": cv2.COLOR_RGBA2BGR,
        "bgra8": cv2.COLOR_BGRA2BGR,
        "mono8": cv2.COLOR_GRAY2BGR,
    }.get(msg.encoding)
    return cv2.cvtColor(img, conversion) if conversion is not None else img.copy()


class TagLocator:
    """Blocking like ArmMoveClient: call detect() only from a callback the node's executor is
    not blocked on; the image, camera_info and /tf subscriptions are served by the executor's
    other threads."""

    def __init__(
        self,
        node: Node,
        tag_sizes: Dict[int, float],
        image_topic: str,
        camera_info_topic: str,
        camera_frame: str,
        debug_image_topic: str = "/tag_detections/image",
    ):
        self._node = node
        self._tag_sizes = tag_sizes
        self._camera_frame = camera_frame
        self._lock = threading.Lock()
        self._image: Optional[Image] = None
        self._info: Optional[CameraInfo] = None
        group = MutuallyExclusiveCallbackGroup()
        node.create_subscription(
            Image, image_topic, self._on_image, qos_profile_sensor_data, callback_group=group
        )
        node.create_subscription(
            CameraInfo, camera_info_topic, self._on_info, qos_profile_sensor_data,
            callback_group=group,
        )
        self._debug = node.create_publisher(Image, debug_image_topic, 1)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node)

        self._dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._params = cv2.aruco.DetectorParameters_create()
        # Sub-pixel corners: a cap tag is only ~40 px across from the survey pose.
        self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        # ...which leaves ~6 px per tag cell; reading the bits from a finer unwarped grid than
        # the default 4 px per cell keeps such a tag from failing to decode.
        self._params.perspectiveRemovePixelPerCell = 8

    def _on_image(self, msg: Image):
        with self._lock:
            self._image = msg

    def _on_info(self, msg: CameraInfo):
        with self._lock:
            self._info = msg

    def _next_image(self, after: Time, timeout_s: float) -> Optional[Image]:
        """The first image taken after `after` (so not one from before the arm stopped)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                image = self._image
            if image is not None and Time.from_msg(image.header.stamp) > after:
                return image
            time.sleep(0.02)
        return None

    def detect(self, frames: int = 3, timeout_s: float = 10.0) -> Dict[int, TagPose]:
        """Every known tag (tag_sizes) seen in the next `frames` images, by ID. Call with the
        arm standing still; each tag's position is the median over the frames it was seen in,
        which steadies the real camera's noise."""
        positions: Dict[int, list] = {}
        rotations: Dict[int, np.ndarray] = {}
        after = self._node.get_clock().now()
        for _ in range(frames):
            image = self._next_image(after, timeout_s)
            if image is None:
                self._node.get_logger().error("no image from the wrist camera")
                break
            after = Time.from_msg(image.header.stamp)
            for tag_id, pose in self._detect_once(image).items():
                positions.setdefault(tag_id, []).append(pose.position)
                rotations[tag_id] = pose.rotation
        return {
            tag_id: TagPose(np.median(np.array(p), axis=0), rotations[tag_id])
            for tag_id, p in positions.items()
        }

    def _detect_once(self, image: Image) -> Dict[int, TagPose]:
        with self._lock:
            info = self._info
        if info is None:
            self._node.get_logger().error("no camera_info from the wrist camera")
            return {}
        try:
            # The arm is standing still: the latest transform is the one the image was taken at.
            tf = self._tf_buffer.lookup_transform("base_link", self._camera_frame, Time())
        except TransformException as exc:
            self._node.get_logger().error(f"no transform to the camera frame: {exc}")
            return {}
        t = tf.transform.translation
        cam_origin = np.array([t.x, t.y, t.z]) + np.array(ARM_BASE_WORLD_XYZ)
        cam_rotation = _quaternion_matrix(tf.transform.rotation)

        bgr = _image_to_bgr(image)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary, parameters=self._params)
        camera_matrix = np.array(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.array(info.d, dtype=np.float64) if len(info.d) else np.zeros(5)

        found: Dict[int, TagPose] = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            for tag_corners, tag_id in zip(corners, ids.flatten()):
                size = self._tag_sizes.get(int(tag_id))
                if size is None:
                    continue
                h = size / 2
                # aruco's corner order: top-left, top-right, bottom-right, bottom-left
                object_points = np.array(
                    [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64
                )
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, tag_corners.reshape(4, 2).astype(np.float64),
                    camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if not ok:
                    continue
                cv2.drawFrameAxes(bgr, camera_matrix, distortion, rvec, tvec, size / 2)
                tag_rotation_cam, _ = cv2.Rodrigues(rvec)
                found[int(tag_id)] = TagPose(
                    cam_origin + cam_rotation @ tvec.flatten(), cam_rotation @ tag_rotation_cam
                )

        debug = Image()
        debug.header = image.header
        debug.height, debug.width = bgr.shape[:2]
        debug.encoding = "bgr8"
        debug.step = debug.width * 3
        debug.data = bgr.tobytes()
        self._debug.publish(debug)
        return found
