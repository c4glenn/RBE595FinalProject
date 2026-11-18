import math
import sys
import threading
import numpy as np
import time

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

TWIST_DEG = -150.0    # flip the sign to twist the other way
WIGGLE_M = 0.01     # 1 cm each side of center
ITERATIONS = 1

from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Adjust to your controller's action name
TRAJ_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
LAST_JOINT = "joint_6"

class WristRotator:
    def __init__(self, node):
        self.node = node
        self.latest = None
        node.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.client = ActionClient(node, FollowJointTrajectory, TRAJ_ACTION)

    def _cb(self, msg):
        self.latest = msg

    def rotate_last_joint(self, offset_deg, duration_s=4.0):
        import time
        t0 = time.time()
        while self.latest is None and time.time() - t0 < 5.0:
            time.sleep(0.05)
        if self.latest is None:
            self.node.get_logger().error("no /joint_states")
            return False
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("trajectory action server not available")
            return False

        pos = dict(zip(self.latest.name, self.latest.position))
        arm_joints = [n for n in self.latest.name if n.startswith("joint_")]  # excludes gripper joints
        if LAST_JOINT not in pos:
            self.node.get_logger().error(f"{LAST_JOINT} not in joint_states: {list(pos)}")
            return False

        target = [pos[n] for n in arm_joints]
        target[arm_joints.index(LAST_JOINT)] += math.radians(offset_deg)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = arm_joints
        pt = JointTrajectoryPoint()
        pt.positions = target
        pt.velocities = [0.0] * len(arm_joints)
        pt.time_from_start = Duration(seconds=duration_s).to_msg()
        goal.trajectory.points = [pt]

        send_future = self.client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.02)
        handle = send_future.result()
        if not handle.accepted:
            self.node.get_logger().error("goal rejected (check joint limits)")
            return False
        res_future = handle.get_result_async()
        while not res_future.done():
            time.sleep(0.02)
        return res_future.result().result.error_code == 0


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])

def check_level_forward(tf_buffer, log, tol_deg=10.0,
                        forward=(1.0, 0.0, 0.0), finger_axis=1):
    """True if the tool approach axis is horizontal and points along `forward`,
    and the finger-opening axis (tool y by default; use 0 for x) is horizontal."""
    t = tf_buffer.lookup_transform(
        "base_link", "end_effector_link", Time(), timeout=Duration(seconds=5.0))
    q = t.transform.rotation
    R = quat_to_rot(q.x, q.y, q.z, q.w)

    approach = R[:, 2]                 # tool z expressed in base_link
    fwd = np.array(forward) / np.linalg.norm(forward)
    tol = math.radians(tol_deg)

    pitch_err = math.asin(max(-1.0, min(1.0, approach[2])))            # 0 = level
    heading_err = math.acos(max(-1.0, min(1.0, float(approach @ fwd))))  # 0 = forward
    roll_err = math.asin(max(-1.0, min(1.0, R[2, finger_axis])))       # 0 = fingers horizontal

    log.info(f"pitch_err={math.degrees(pitch_err):.1f} "
             f"heading_err={math.degrees(heading_err):.1f} "
             f"roll_err={math.degrees(roll_err):.1f} deg")
    return max(abs(pitch_err), abs(heading_err), abs(roll_err)) < tol

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


def run_twist(arm, gripper, log, xyz, rpy, iter):
    try:
        # gripper.close()
        if not gripper.move_to(0.7):
            log.error("gripper close failed")
            return False
        twisted = twisted_rpy(rpy, TWIST_DEG)
        if not arm.move_to(xyz, twisted):
            log.error("twist move failed")
            return False

        for _ in range(iter):
            for dy in (-WIGGLE_M, +WIGGLE_M):  # left, then right
                if not arm.move_to((xyz[0], xyz[1] + dy, xyz[2]), twisted):
                    log.error("wiggle move failed")
                    return False

        return True
    finally:
        # always attempt to reset, even after a failure
        arm.move_to(xyz, rpy)
        #gripper.open()
        #gripper.move_to(0.0)

# def run_wrist_twist(arm, gripper, wrist, tf_buffer, log, iters):
#     rolled = False
#     center = None
#     try:
#         if not gripper.move_to(0.7):
#             log.error("gripper close failed")
#             return False

#         if not wrist.rotate_last_joint(TWIST_DEG):
#             log.error("wrist roll failed")
#             return False
#         rolled = True

#         # pose AFTER the roll, so wiggles hold the real rolled orientation
#         xyz, rpy = current_ee_pose_world(tf_buffer)
#         center = (xyz, rpy)

#         for _ in range(iters):
                    
#             if not arm.move_to((xyz[0], xyz[1] + WIGGLE_M, xyz[2]), rpy):
#                 log.error("wiggle move failed")
#                 return False
#             arm.move_to(*center)
            
#         return True
#     finally:
#         # # always recenter, then roll back, even after a failure
#         # if center is not None:
#         #     arm.move_to(*center)
#         if rolled:
#             wrist.rotate_last_joint(-TWIST_DEG)



def run_wrist_twist(arm, gripper, wrist, tf_buffer, log, iters, close_gripper=True):
    # pose BEFORE anything moves: this is the known-good level/forward pose
    start_xyz, start_rpy = current_ee_pose_world(tf_buffer)
    rolled = False
    center = None

    try:
        if close_gripper:
            if not gripper.move_to(0.7):
                log.error("gripper close failed")
                return False
            
        if not gripper.move_to(0.7):
            log.error("gripper close failed")
            return False

        if not wrist.rotate_last_joint(TWIST_DEG):
            log.error("wrist roll failed")
            return False
        rolled = True

        # pose AFTER the roll, so wiggles hold the real rolled orientation
        xyz, rpy = current_ee_pose_world(tf_buffer)
        center = (xyz, rpy)

        for _ in range(iters):
            if not arm.move_to((xyz[0], xyz[1] + WIGGLE_M, xyz[2]), rpy):
                log.error("wiggle move failed")
                return False
            arm.move_to(*center)
            print("Iteration: ", _)

        return True
    finally:
        # roll back by dead reckoning, then correct the residual error with MoveIt
        if rolled:
            wrist.rotate_last_joint(-TWIST_DEG)
            time.sleep(0.5)  # let the joint settle before reading/planning
        if not arm.move_to(start_xyz, start_rpy):
            log.error("stabilize move failed; check orientation manually")
        else:
            log.info("stabilized to start pose")

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

def run_experiment(arm, gripper, wrist, tf_buffer, log):
    input("Enter to close gripper...")
    if not gripper.move_to(0.7):
        log.error("gripper close failed")
        return False

    runs = 0
    while True:
        cmd = input(f"[{runs} runs] Enter = run wrist twist, 'q' = finish: ").strip().lower()
        if cmd == "q":
            break

        # make sure the previous run left the gripper level before going again
        if not check_level_forward(tf_buffer, log):
            log.error("not level/forward after last run; stopping")
            break

        ok = run_wrist_twist(arm, gripper, wrist, tf_buffer, log, ITERATIONS,
                             close_gripper=False)
        runs += 1
        log.info(f"run {runs} result: {ok}")
    return True

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
    wrist = WristRotator(node)

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

        elif mode == "open":
            xyz, rpy = current_ee_pose_world(tf_buffer)
            open_gripper(arm, gripper, log, xyz, rpy)

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
            log.info(f"twist result: {run_twist(arm, gripper, log, xyz, rpy, ITERATIONS)}")
            input("Shake Done, enter to open. Enter...")
            open_gripper(arm, gripper, log, xyz, rpy)

        elif mode == "test":
            if not check_level_forward(tf_buffer, log):
                log.error("gripper not level/forward; aborting")
                return
            log.info(f"experiment result: {run_experiment(arm, gripper, wrist, tf_buffer, log)}")
            input("Done. Enter to open gripper...")
            xyz, rpy = current_ee_pose_world(tf_buffer)
            open_gripper(arm, gripper, log, xyz, rpy)
            
        elif mode == "wrist":
            if not check_level_forward(tf_buffer, log):
                log.error("gripper not level/forward; aborting")
                return
            input("Close gripper, roll, wiggle, roll back. Area clear, e-stop in reach. Enter...")
            log.info(f"wrist twist result: {run_wrist_twist(arm, gripper, wrist, tf_buffer, log, ITERATIONS)}")
            input("Done. Enter to open gripper...")
            gripper.move_to(0.0)
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