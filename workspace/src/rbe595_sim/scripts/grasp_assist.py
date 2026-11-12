#!/usr/bin/env python3
"""Simulation-only grasp assist for the gen3_lite_2f gripper.

Gazebo's contact physics cannot hold a jar between the gen3_lite_2f fingers reliably: the
fingers are coupled by a mimic constraint, and under bullet-featherstone a held jar slowly pries
them open and slips out. So, like most Gazebo pick-and-place setups, this node backs the
physical grasp with a fixed joint:

- every spice bottle has a DetachableJoint system on the robot (added by sim.launch.py), parent
  end_effector_link, child the bottle;
- when the fingers have closed onto a bottle (finger joint past ATTACH_AT, bottle between the
  finger pads) and have lifted it LIFT_TO_ATTACH off where it stood -- clear of a rack pocket,
  where a rigidly held bottle would bind -- that bottle is attached; when the fingers open
  again (below DETACH_AT) it is let go.

The fingers still close on the bottle physically and stop at its width, so a grasp that misses
closes all the way and attaches nothing, exactly as on the real gripper. Nothing here is needed
on the real robot.
"""

import math
import threading

import rclpy
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.stringmsg_pb2 import StringMsg
from gz.transport13 import Node as GzNode
from rclpy.node import Node
from sensor_msgs.msg import JointState

FINGER_JOINT = "right_finger_bottom_joint"
# Finger joint values: ~0.35 open, ~0.56-0.60 closed on a 48 mm jar, 0.75+ closed on nothing.
ATTACH_AT = 0.50
ATTACH_MAX = 0.70  # past this the fingers closed on nothing
DETACH_AT = 0.45
# Grasp point: kortex_description tool_frame, 0.13 m out along end_effector_link z.
GRASP_POINT_REACH = 0.13
# A bottle counts as between the fingers if the grasp point is within this distance of its
# axis and within its height (plus margin).
MAX_RADIAL = 0.035
HEIGHT_MARGIN = 0.02
# Rack pockets are 20 mm deep.
LIFT_TO_ATTACH = 0.022


def topic(bottle, leaf):
    return f"/rbe595_sim/grasp_assist/{bottle}/{leaf}"


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # t = 2 * cross(q_vec, v)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_conj(q):
    return (q[0], -q[1], -q[2], -q[3])


class GraspAssist(Node):
    def __init__(self):
        super().__init__("grasp_assist")
        self.robot = self.declare_parameter("robot_name", "gen3_lite").value
        self.bottles = list(self.declare_parameter("bottles", ["spice_bottle_0"]).value)
        self.bottle_height = self.declare_parameter("bottle_height", 0.0813).value
        world = self.declare_parameter("world", "table_world").value

        self._lock = threading.Lock()
        self._poses = {}  # name -> (position, quaternion wxyz); robot links are model-relative
        self._state = {}  # bottle -> "attached" / "detached", from the DetachableJoint
        self._held = None
        self._gripped = None  # (bottle, z it stood at) once the fingers closed on it

        self._gz = GzNode()
        self._attach = {b: self._gz.advertise(topic(b, "attach"), Empty) for b in self.bottles}
        self._detach = {b: self._gz.advertise(topic(b, "detach"), Empty) for b in self.bottles}
        self._gz.subscribe(Pose_V, f"/world/{world}/pose/info", self._on_poses)
        for b in self.bottles:
            self._gz.subscribe(StringMsg, topic(b, "state"), self._state_cb(b))

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        # The DetachableJoint system attaches every bottle when it starts; let them all go.
        self.create_timer(0.5, self._release_unheld)

    def _state_cb(self, bottle):
        def cb(msg):
            with self._lock:
                self._state[bottle] = msg.data
        return cb

    def _on_poses(self, msg):
        poses = {}
        for p in msg.pose:
            poses[p.name] = (
                (p.position.x, p.position.y, p.position.z),
                (p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z),
            )
        with self._lock:
            self._poses.update(poses)

    def _release_unheld(self):
        with self._lock:
            states = dict(self._state)
            held = self._held
        for b in self.bottles:
            if b != held and states.get(b) != "detached":
                self._detach[b].publish(Empty())

    def _grasp_point_world(self):
        with self._lock:
            robot = self._poses.get(self.robot)
            ee = self._poses.get("end_effector_link")
        if robot is None or ee is None:
            return None
        # end_effector_link is reported relative to the robot model
        ee_pos = tuple(r + d for r, d in zip(robot[0], quat_rotate(robot[1], ee[0])))
        ee_rot = quat_mul(robot[1], ee[1])
        reach = quat_rotate(ee_rot, (0.0, 0.0, GRASP_POINT_REACH))
        return tuple(p + r for p, r in zip(ee_pos, reach))

    def _bottle_between_fingers(self):
        grasp = self._grasp_point_world()
        if grasp is None:
            return None
        best, best_radial = None, MAX_RADIAL
        for b in self.bottles:
            with self._lock:
                pose = self._poses.get(b)
            if pose is None:
                continue
            # grasp point in the bottle's frame (origin at the centre of its base)
            local = quat_rotate(quat_conj(pose[1]), tuple(g - p for g, p in zip(grasp, pose[0])))
            radial = math.hypot(local[0], local[1])
            within_height = -HEIGHT_MARGIN <= local[2] <= self.bottle_height + HEIGHT_MARGIN
            if within_height and radial < best_radial:
                best, best_radial = b, radial
        return best

    def _on_joint_states(self, msg):
        if FINGER_JOINT not in msg.name:
            return
        finger = msg.position[msg.name.index(FINGER_JOINT)]
        with self._lock:
            held, gripped = self._held, self._gripped
        if finger < DETACH_AT:
            if held is not None:
                self._detach[held].publish(Empty())
                self.get_logger().info(f"fingers opened: {held} released")
            with self._lock:
                self._held = self._gripped = None
            return
        if held is not None:
            return
        if gripped is None:
            if ATTACH_AT <= finger < ATTACH_MAX:
                bottle = self._bottle_between_fingers()
                if bottle is not None:
                    with self._lock:
                        self._gripped = (bottle, self._poses[bottle][0][2])
                    self.get_logger().info(f"fingers closed on {bottle}")
            return
        bottle, z0 = gripped
        with self._lock:
            z = self._poses[bottle][0][2]
        if z >= z0 + LIFT_TO_ATTACH:
            self._attach[bottle].publish(Empty())
            with self._lock:
                self._held = bottle
            self.get_logger().info(f"{bottle} lifted: attached")


def main():
    rclpy.init()
    node = GraspAssist()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
