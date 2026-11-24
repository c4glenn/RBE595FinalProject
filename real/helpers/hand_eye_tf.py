#!/usr/bin/env python3
"""Publishes the hand-eye TF (end_effector_link -> camera_link) from config/real.yaml, like a
static_transform_publisher, but it can be switched to the other camera mount while running:

  ros2 topic pub --once /spice/hand_eye_mount std_msgs/String "data: inside"

The task does that by itself when the rack's tags show the camera is on the other side of the
gripper (see spice_task/steps.py, check_mount).

  python3 hand_eye_tf.py [path/to/real.yaml]
"""
import os
import sys

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

HERE = os.path.dirname(os.path.abspath(__file__))
MOUNT_TOPIC = '/spice/hand_eye_mount'


def mount_qos():
    return QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class HandEyeTF(Node):
    def __init__(self, cfg):
        super().__init__('hand_eye_tf')
        self.mounts = cfg.get('hand_eye_mounts') or {}
        # Our own /tf_static publisher (same QoS as tf2's static broadcaster): Jazzy's Python
        # StaticTransformBroadcaster never updates a frame it has sent once, so it cannot switch.
        self.tf_pub = self.create_publisher(TFMessage, '/tf_static', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        self.publish(cfg['hand_eye'], 'from the config')
        self.create_subscription(String, MOUNT_TOPIC, self.on_mount, mount_qos())

    def on_mount(self, msg):
        he = self.mounts.get(msg.data)
        if he is None:
            self.get_logger().error(f'unknown camera mount {msg.data!r} (have: {sorted(self.mounts)})')
            return
        self.publish(he, msg.data)

    def publish(self, he, what):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = he['parent'], he['child']
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(v) for v in he['translation'])
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
         t.transform.rotation.w) = (float(v) for v in he['rotation'])
        self.tf_pub.publish(TFMessage(transforms=[t]))
        self.get_logger().info(f'hand-eye {he["parent"]} -> {he["child"]} ({what}): '
                               f'translation {list(he["translation"])}')


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'config', 'real.yaml')
    with open(path) as f:
        cfg = yaml.safe_load(f)
    rclpy.init()
    node = HandEyeTF(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
