import math
import sys
import threading

import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from rbe595_manipulation import poses
from rbe595_manipulation.arm_client import ARM_BASE_WORLD_XYZ, ArmMoveClient
from rbe595_manipulation.shake import shake

from rbe595_manipulation._orientation import rpy_to_quaternion
from rbe595_manipulation.gripper_client_actual import GripperClient

TWIST_DEG = 120.0    # flip the sign to twist the other way
WIGGLE_M = 0.01     # 1 cm each side of center


def quat_to_rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def current_ee_pose_world(tf_buffer):
    """Current EE pose, with xyz converted into the 'world' frame that
    ArmMoveClient.move_to() expects (it subtracts ARM_BASE_WORLD_XYZ again)."""
    t = tf_buffer.lookup_transform(
        "base_link", "end_effector_link", Time(), timeout=Duration(seconds=5.0)
    )
    tr, q = t.transform.translation, t.transform.rotation
    world_xyz = (
        tr.x + ARM_BASE_WORLD_XYZ[0],
        tr.y + ARM_BASE_WORLD_XYZ[1],
        tr.z + ARM_BASE_WORLD_XYZ[2],
    )
    return world_xyz, quat_to_rpy(q.x, q.y, q.z, q.w)


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def twisted_rpy(rpy, deg):
    """Same pose rotated `deg` about the tool's own z (approach) axis."""
    q = rpy_to_quaternion(*rpy)
    half = math.radians(deg) / 2.0
    # post-multiplying rotates about the tool's local axis, not the world's
    r = quat_mul((q.x, q.y, q.z, q.w), (0.0, 0.0, math.sin(half), math.cos(half)))
    return quat_to_rpy(*r)


def run_twist(arm, gripper, log, xyz, rpy):
    try:
        # gripper.close()
        if not gripper.move_to(0.7):
            log.error("gripper close failed")
            return False
        twisted = twisted_rpy(rpy, TWIST_DEG)
        if not arm.move_to(xyz, twisted):
            log.error("twist move failed")
            return False
        # for dy in (-WIGGLE_M, +WIGGLE_M):  # left, then right, in base_link y
        #     if not arm.move_to((xyz[0], xyz[1] + dy, xyz[2]), twisted):
        #         log.error("wiggle move failed")
        #         return False
        for _ in range(2):
            for dy in (-WIGGLE_M, +WIGGLE_M):  # left, then right
                if not arm.move_to((xyz[0], xyz[1] + dy, xyz[2]), twisted):
                    log.error("wiggle move failed")
                    return False

        # if not arm.move_to((xyz[0], xyz[1] + WIGGLE_M, xyz[2]), twisted):
        #     log.error("wiggle move failed")
        #     return False
        return True
    finally:
        # always attempt to reset, even after a failure
        arm.move_to(xyz, rpy)
        #gripper.open()
        #gripper.move_to(0.0)

def open_gripper(arm, gripper, log, xyz, rpy):
    # try:
    #     # gripper.close()
    #     if not gripper.move_to(0.6):
    #         log.error("gripper close failed")
    #         return False
    #     twisted = twisted_rpy(rpy, TWIST_DEG)
    #     if not arm.move_to(xyz, twisted):
    #         log.error("twist move failed")
    #         return False
    #     for dy in (-WIGGLE_M, +WIGGLE_M):  # left, then right, in base_link y
    #         if not arm.move_to((xyz[0], xyz[1] + dy, xyz[2]), twisted):
    #             log.error("wiggle move failed")
    #             return False
    #     return True
    # finally:
    #     # always attempt to reset, even after a failure
    arm.move_to(xyz, rpy)
    #gripper.open()
    gripper.move_to(0.0)

def main(args=None):
    rclpy.init(args=args)
    node = Node("test_step5_shake")
    log = node.get_logger()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    arm = ArmMoveClient(node)
    gripper = GripperClient(node)
    mode = sys.argv[1] if len(sys.argv) > 1 else "twist"

    #     arm = ArmMoveClient(node)
    # gripper = GripperClient(node)

    try:
        if mode == "here":
            xyz, rpy = current_ee_pose_world(tf_buffer)
            log.info(f"current EE (world frame) xyz={xyz} rpy={rpy}")

            input("Step A: no-op move to current pose (planner sanity check). Enter...")
            if not arm.move_to(xyz, rpy):
                log.error("even a no-op move failed -> MoveIt config/planning scene issue")
                return

            input("Step B: shake around the CURRENT pose. Area clear, e-stop in reach. Enter...")
            log.info(f"shake result: {shake(arm, xyz, rpy)}")

        elif mode == "bowl":
            input("Enter to move to bowl-over pose...")
            if not arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_APPROACH_ORIENTATION_RPY):
                log.error("approach move failed")
                return
            if not arm.move_to(poses.BOWL_OVER_EE_XYZ, poses.BOWL_OVER_ORIENTATION_RPY):
                log.error("over-orientation move failed")
                return
            input("At bowl-over pose. Enter to shake...")
            log.info(f"shake result: {shake(arm, poses.BOWL_OVER_EE_XYZ, poses.BOWL_OVER_ORIENTATION_RPY)}")
        
        elif mode == "twist":
            xyz, rpy = current_ee_pose_world(tf_buffer)
            log.info(f"start EE (world frame) xyz={xyz} rpy={rpy}")
            input("Close gripper, twist 45 deg, wiggle, reset, open. Area clear, e-stop in reach. Enter...")
            log.info(f"twist result: {run_twist(arm, gripper, log, xyz, rpy)}")
            input("Shake Done, enter to open. Enter...")
            open_gripper(arm, gripper, log, xyz, rpy)

        else:
            log.error(f"unknown mode {mode!r}; use 'here' or 'bowl'")
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()