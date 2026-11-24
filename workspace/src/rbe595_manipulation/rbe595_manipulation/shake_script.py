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

from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


# --- SHAKE CONTROLS ---
TWIST_DEG = -150.0 
ITERATIONS = 3 #number of wrist twists (abotu 2-3g per twist from experiments)
BOWL_Z_POS = [0.0, 0.0, 1.0] # position of the end effecto over the bowl (will be updated by bowl detection service)
FORWARD_RPY = (0.0, -math.pi / 2.0, math.pi)    # should point gripper straight forward with camera up
# ----------------------

# Joint controlling action name
TRAJ_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
LAST_JOINT = "joint_6"

JOINT_LIMITS_DEG = {"joint_6": (-160.0, 160.0)}   


# this class manually controls the wrist joint for a safe and reliable rotation. 
# do not exceed joint position limits or it may freeze the robot and require a restart
class WristRotator:
    def __init__(self, node):
        self.node = node
        self.latest = None
        node.create_subscription(JointState, "/joint_states", self._cb, 10)
        self.client = ActionClient(node, FollowJointTrajectory, TRAJ_ACTION)

    def _cb(self, msg):
        self.latest = msg
    

    def _min_jerk_traj(self, joint_names, start, idx, delta_rad, duration_s, dt=0.05):
        """Single smooth trajectory: only joint `idx` moves, others hold still."""
        n = max(2, int(math.ceil(duration_s / dt)))
        pts = []
        for k in range(1, n + 1):
            tau = k / n
            s    = 10*tau**3 - 15*tau**4 + 6*tau**5            # position fraction
            sdot = (30*tau**2 - 60*tau**3 + 30*tau**4) / duration_s
            pos = list(start)
            vel = [0.0] * len(joint_names)
            pos[idx] = start[idx] + delta_rad * s
            vel[idx] = delta_rad * sdot
            pt = JointTrajectoryPoint()
            pt.positions = pos
            pt.velocities = vel
            pt.time_from_start = Duration(seconds=tau * duration_s).to_msg()
            pts.append(pt)
        return pts

    def rotate_last_joint(self, offset_deg, duration_s=2.5):
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
        arm_joints = [n for n in self.latest.name if n.startswith("joint_")]
        idx = arm_joints.index(LAST_JOINT)
        start = [pos[n] for n in arm_joints]

        # clamp to joint limits
        lo, hi = JOINT_LIMITS_DEG[LAST_JOINT]
        start_deg = math.degrees(start[idx])
        goal_deg = max(lo, min(hi, start_deg + offset_deg))
        delta = math.radians(goal_deg - start_deg)
        if abs(goal_deg - (start_deg + offset_deg)) > 1e-6:
            self.node.get_logger().warn(f"clamped {start_deg + offset_deg:.1f} -> {goal_deg:.1f} deg")

        peak_v = 1.875 * abs(delta) / duration_s      # rad/s
        peak_a = 5.77 * abs(delta) / duration_s**2    # rad/s^2
        self.node.get_logger().info(
            f"{LAST_JOINT}: {start_deg:.1f} -> {goal_deg:.1f} deg in {duration_s:.2f}s "
            f"(peak {peak_v:.2f} rad/s, {peak_a:.1f} rad/s^2)")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = arm_joints
        goal.trajectory.points = self._min_jerk_traj(arm_joints, start, idx, delta, duration_s)

        send_future = self.client.send_goal_async(goal)
        t0 = time.time()
        while not send_future.done():
            if time.time() - t0 > 5.0:
                self.node.get_logger().error("timed out sending goal")
                return False
            time.sleep(0.01)
        handle = send_future.result()
        if not handle.accepted:
            self.node.get_logger().error("goal rejected")
            return False

        res_future = handle.get_result_async()
        t0 = time.time()
        while not res_future.done():
            if time.time() - t0 > duration_s + 5.0:
                self.node.get_logger().error("result timeout, cancelling")
                handle.cancel_goal_async()
                return False
            time.sleep(0.01)
        res = res_future.result().result
        if res.error_code != 0:
            self.node.get_logger().error(f"controller error {res.error_code}: {res.error_string}")
            return False
        return True

 

def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])



def check_level_forward(tf_buffer, log, tol_deg=15.0,
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




def move_to_forward(arm, tf_buffer, log, xyz, roll_deg=0.0, check=True):
    """Move EE to `xyz` (world frame) with the gripper level and pointing straight forward.

    roll_deg rotates the gripper about its own approach axis, in case your
    fingers need to open vertically instead (e.g. 90).
    """
    rpy = FORWARD_RPY if roll_deg == 0.0 else twisted_rpy(FORWARD_RPY, roll_deg)

    if not arm.move_to(tuple(xyz), rpy):
        log.error(f"move_to_forward failed for xyz={xyz}")
        return False

    if check:
        # finger_axis=1 -> tool y horizontal; if you pass roll_deg=90, use finger_axis=0
        finger_axis = 1 if roll_deg % 180 == 0 else 0
        if not check_level_forward(tf_buffer, log, finger_axis=finger_axis):
            log.warn("arrived, but orientation is outside tolerance")
            return False
    return True



# main wrist pipeline code
def run_wrist_twist_pipeline(arm, gripper, wrist, tf_buffer, log, iters, close_gripper=True):
    # pose BEFORE anything moves: this is the known-good level/forward pose
    start_xyz, start_rpy = current_ee_pose_world(tf_buffer)
    rolled = False
    center = None
    
    try:
        for _ in range(iters):

            if not wrist.rotate_last_joint(TWIST_DEG):
                log.error("wrist roll failed")
                return False
            rolled = True

            wrist.rotate_last_joint(-TWIST_DEG)
            time.sleep(0.5)  # let the joint settle before reading/planning
        return True
 
    finally:
        # correct the residual error with MoveIt when done
        if not arm.move_to(start_xyz, start_rpy):
            log.error("stabilize move failed; check orientation manually")
        else:
            log.info("stabilized to start pose")

def open_gripper(arm, gripper, log, xyz, rpy):
  
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
    mode = sys.argv[1] if len(sys.argv) > 1 else "shake"

    #     arm = ArmMoveClient(node)
    # gripper = GripperClient(node)
    wrist = WristRotator(node)

    try:

        # goto mode, you can test if the robot goes to the bowl position without shake
        if mode == "goto":
            # xyz = [0.0, 0.0, 1.0]
            input(f"Move to {BOWL_Z_POS} gripper forward. Area clear, e-stop in reach. Enter...")
            log.info(f"goto result: {move_to_forward(arm, tf_buffer, log, BOWL_Z_POS)}")


        # wrist pipeline mode goes to the xyz position which will above be the bowl, 
        # and does n twists controlled by the ITERATION param
        elif mode == "shake":

            # xyz = [0.0, 0.0, 1.0]
            input(f"Move to {BOWL_Z_POS} gripper forward. Area clear, e-stop in reach. Enter...")
            log.info(f"goto result: {move_to_forward(arm, tf_buffer, log, BOWL_Z_POS)}")

            time.sleep(1.0)
            
            if not check_level_forward(tf_buffer, log):
                log.error("gripper not level/forward; aborting")
                return
            
            log.info(f"wrist twist result: {run_wrist_twist_pipeline(arm, gripper, wrist, tf_buffer, log, ITERATIONS)}")
            log.info("SHAKE DONE")
          

        
        else:
            log.error(f"unknown mode {mode!r}; use 'goto' or 'shake'")
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()