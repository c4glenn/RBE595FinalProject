#!/usr/bin/env python3
#python3 go_home.py [path/to/real.yaml]   (copied from HW7)
import os
import sys
import threading
import time

import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from joint_bounds import JointWatcher, unwind
from retreat import retreat_and_home
from robot_motion import ArmMover, parse_pose

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'config', 'real.yaml')
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('home') is None:
        print(f'No home pose set in {config_path}; not moving.')
        return 0
    try:
        home = parse_pose(cfg['home'], 'home')
    except ValueError as e:
        print(e)
        return 1

    rclpy.init()
    node = Node('go_home')
    watcher = JointWatcher(node)
    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    mover = ArmMover(node, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                     motion=cfg.get('motion', 'cartesian'), cartesian_fallback=cfg.get('cartesian_fallback', False),
                     planning=cfg.get('planning'))
    ok = False
    try:
        if mover.wait_until_ready(timeout=120.0) and unwind(node, watcher, log=node.get_logger().info):
            mover.set_scene(cfg)

            def ee_pose():
                t = tf_buffer.lookup_transform(cfg['base_frame'], cfg['ee_link'], Time(),
                                               timeout=Duration(seconds=5.0)).transform
                return [t.translation.x, t.translation.y, t.translation.z,
                        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
            try:
                # back out of a pocket / from round the jar first (the arm may be where a run stopped)
                ok = retreat_and_home(mover, cfg, ee_pose, log=node.get_logger().info,
                                      finger=watcher.positions.get(cfg['gripper']['finger_joint']))
            except Exception as e:                                   # noqa: BLE001  (no TF yet)
                node.get_logger().warn(f'safe retreat not possible ({e}); plain move home')
                for attempt in range(1, 4):
                    node.get_logger().info(f'Moving home (attempt {attempt}) ...')
                    if mover.move_to(home):
                        ok = True
                        break
                    time.sleep(3.0)
    finally:
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()
    print('Home reached.' if ok else 'Could not reach home (see errors above).')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())