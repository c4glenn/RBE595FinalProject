import math
import time
from typing import Sequence, Tuple

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from rbe595_manipulation._orientation import rpy_to_quaternion
from rbe595_manipulation._wait import wait_for_future
from rbe595_manipulation.poses import ARM_BASE_WORLD_XYZ, DEFAULT_VELOCITY_SCALING

PLANNING_GROUP = "arm"  # kinova_gen3_lite_moveit_config gen3_lite.srdf
END_EFFECTOR_LINK = "end_effector_link"
# Tight enough that a bottle being set down is not pushed into the table / slot floor.
POSITION_TOLERANCE_M = 0.003
# The grasp point is GRIPPER_FINGERTIP_REACH (0.13 m) out, so each 0.01 rad of tilt moves it
# 1.3 mm sideways; a bottle in a rack pocket has only ~1.4 mm of play before it tips and jams.
ORIENTATION_TOLERANCE_RAD = 0.01
# The Gen3 lite can reach most poses either facing the target or folded back over its own base
# (joint_1 turned ~180 deg, joint_2 flipped), and the IK sampler picks either at random. Keep
# joint_1 within this much of the target's bearing from the base so the arm always faces it.
BASE_JOINT = "joint_1"
BASE_JOINT_TOLERANCE_RAD = 1.0
# Likewise the elbow: a pose reached with the arm nearly straight can come out with joint_3 just
# past zero, and from there a straight-line move that brings the hand in closer has to fold the
# elbow the wrong way and stalls at the straight-arm singularity. Keep it on the positive
# (elbow-up) branch, the one the home pose uses, for every pose goal.
ELBOW_JOINT = "joint_3"
ELBOW_RANGE_RAD = (0.0, 2.6)  # URDF limit is +/-2.61


def world_to_base_link(position_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(p - b for p, b in zip(position_xyz, ARM_BASE_WORLD_XYZ))


# Found by testing: issuing the next move immediately after "trajectory execution SUCCEEDED"
# can catch the arm mid-settle (momentary overshoot past a joint limit as it decelerates),
# which move_group's next CheckStartStateBounds then rejects outright. A short pause here
# lets /joint_states publish the settled state before the next goal is sent.
SETTLE_TIME_S = 1.0

# Straight-line moves: interpolation step, and the least fraction of the line that has to be
# achievable (a straight approach is all-or-nothing -- a partial one stops short of the grasp).
CARTESIAN_MAX_STEP_M = 0.005
CARTESIAN_MIN_FRACTION = 0.999


class ArmMoveClient:
    """Plans and executes a single Cartesian end-effector pose goal via MoveIt's MoveGroup
    action, in one call (planning_options.plan_only=False).

    Uses a raw ActionClient against moveit_msgs/action/MoveGroup rather than moveit_py /
    MoveGroupInterface, since moveit_py isn't installed in this workspace's image.

    Blocking: safe to call only from a callback running under a ReentrantCallbackGroup on a
    node spun by a MultiThreadedExecutor (see pick_place_spice_action_server.py's main()).
    """

    def __init__(self, node: Node, action_name: str = "/move_action"):
        self._node = node
        self._client = ActionClient(node, MoveGroup, action_name)
        self._cartesian = node.create_client(GetCartesianPath, "/compute_cartesian_path")
        self._execute = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")

    def move_linear(
        self,
        position_xyz: Tuple[float, float, float],
        rpy: Sequence[float],
        velocity_scaling: float = DEFAULT_VELOCITY_SCALING,
        timeout_s: float = 60.0,
    ) -> bool:
        """Move the end effector in a straight line from where it is to (position_xyz, rpy)
        (world frame), keeping clear of the planning scene -- for the short approach, grasp,
        place and retreat legs, where a free-space plan could swing the fingers into the
        bottle."""
        request = GetCartesianPath.Request()
        request.header.frame_id = "base_link"
        request.start_state.is_diff = True
        request.group_name = PLANNING_GROUP
        request.link_name = END_EFFECTOR_LINK
        pose = Pose()
        x, y, z = world_to_base_link(position_xyz)
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
        request.waypoints = [pose]
        request.max_step = CARTESIAN_MAX_STEP_M
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = velocity_scaling
        request.max_acceleration_scaling_factor = velocity_scaling

        if not self._cartesian.wait_for_service(timeout_sec=timeout_s):
            self._node.get_logger().error("compute_cartesian_path service not available")
            return False
        future = self._cartesian.call_async(request)
        if not wait_for_future(future, timeout_s) or future.result() is None:
            self._node.get_logger().error("compute_cartesian_path call timed out")
            return False
        response = future.result()
        if response.fraction < CARTESIAN_MIN_FRACTION:
            self._node.get_logger().error(
                f"straight-line move only {response.fraction:.0%} achievable "
                f"(error_code={response.error_code.val})"
            )
            return False

        if not self._execute.wait_for_server(timeout_sec=timeout_s):
            self._node.get_logger().error("execute_trajectory action server not available")
            return False
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = response.solution
        send_future = self._execute.send_goal_async(goal)
        if not wait_for_future(send_future, timeout_s):
            self._node.get_logger().error("execute_trajectory goal send timed out")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("execute_trajectory goal rejected")
            return False
        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, timeout_s) or result_future.result() is None:
            self._node.get_logger().error("execute_trajectory result timed out")
            return False
        error_code = result_future.result().result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(f"execute_trajectory failed: error_code={error_code}")
            return False
        time.sleep(SETTLE_TIME_S)
        return True

    def move_to(
        self,
        position_xyz: Tuple[float, float, float],
        rpy: Sequence[float],
        velocity_scaling: float = DEFAULT_VELOCITY_SCALING,
        timeout_s: float = 60.0,
    ) -> bool:
        """position_xyz is in rbe595_sim's world frame (see poses.py); converted to base_link
        (MoveIt's actual planning frame) before building the goal -- see ARM_BASE_WORLD_XYZ.

        Retried once on CONTROL_FAILED: letting go of (or setting down) a bottle can nudge the
        arm after the previous move finished, so the new plan's start point no longer matches
        the measured joint state."""
        error_code = self._move_once(position_xyz, rpy, velocity_scaling, timeout_s)
        if error_code == MoveItErrorCodes.CONTROL_FAILED:
            self._node.get_logger().warn("move_group CONTROL_FAILED, retrying once after settling")
            time.sleep(SETTLE_TIME_S)
            error_code = self._move_once(position_xyz, rpy, velocity_scaling, timeout_s)
        if error_code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(f"move_group failed: error_code={error_code}")
            return False
        time.sleep(SETTLE_TIME_S)
        return True

    def _move_once(self, position_xyz, rpy, velocity_scaling, timeout_s) -> int:
        """One plan-and-execute request; returns the MoveIt error code (FAILURE if no result)."""
        goal = MoveGroup.Goal()
        goal.request.group_name = PLANNING_GROUP
        goal.request.num_planning_attempts = 10
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = velocity_scaling
        goal.request.max_acceleration_scaling_factor = velocity_scaling
        goal.request.goal_constraints = [
            self._build_constraints(world_to_base_link(position_xyz), rpy)
        ]
        goal.planning_options.plan_only = False

        if not self._client.wait_for_server(timeout_sec=timeout_s):
            self._node.get_logger().error("move_action action server not available")
            return MoveItErrorCodes.FAILURE

        send_future = self._client.send_goal_async(goal)
        if not wait_for_future(send_future, timeout_s):
            self._node.get_logger().error("move_group goal send timed out")
            return MoveItErrorCodes.FAILURE
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("move_group goal rejected")
            return MoveItErrorCodes.FAILURE

        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, timeout_s):
            self._node.get_logger().error("move_group result timed out")
            return MoveItErrorCodes.FAILURE
        result = result_future.result()
        if result is None:
            return MoveItErrorCodes.FAILURE
        return result.result.error_code.val

    @staticmethod
    def _build_constraints(
        position_xyz: Tuple[float, float, float], rpy: Sequence[float]
    ) -> Constraints:
        """position_xyz must already be in base_link frame -- see move_to()."""
        constraints = Constraints()

        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = "base_link"
        position_constraint.link_name = END_EFFECTOR_LINK
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [POSITION_TOLERANCE_M]

        target_pose = Pose()
        target_pose.position.x = position_xyz[0]
        target_pose.position.y = position_xyz[1]
        target_pose.position.z = position_xyz[2]
        target_pose.orientation.w = 1.0

        region = BoundingVolume()
        region.primitives = [sphere]
        region.primitive_poses = [target_pose]
        position_constraint.constraint_region = region
        position_constraint.weight = 1.0
        constraints.position_constraints = [position_constraint]

        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = "base_link"
        orientation_constraint.link_name = END_EFFECTOR_LINK
        orientation_constraint.orientation = rpy_to_quaternion(rpy[0], rpy[1], rpy[2])
        orientation_constraint.absolute_x_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE_RAD
        orientation_constraint.weight = 1.0
        constraints.orientation_constraints = [orientation_constraint]

        base_constraint = JointConstraint()
        base_constraint.joint_name = BASE_JOINT
        base_constraint.position = math.atan2(position_xyz[1], position_xyz[0])
        base_constraint.tolerance_above = BASE_JOINT_TOLERANCE_RAD
        base_constraint.tolerance_below = BASE_JOINT_TOLERANCE_RAD
        base_constraint.weight = 1.0
        elbow_constraint = JointConstraint()
        elbow_constraint.joint_name = ELBOW_JOINT
        low, high = ELBOW_RANGE_RAD
        elbow_constraint.position = (low + high) / 2
        elbow_constraint.tolerance_above = (high - low) / 2
        elbow_constraint.tolerance_below = (high - low) / 2
        elbow_constraint.weight = 1.0
        constraints.joint_constraints = [base_constraint, elbow_constraint]

        return constraints
