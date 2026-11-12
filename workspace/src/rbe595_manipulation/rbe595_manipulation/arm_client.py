import time
from typing import Sequence, Tuple

from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    MoveItErrorCodes,
    OrientationConstraint,
    PositionConstraint,
)
from rclpy.action import ActionClient
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from rbe595_manipulation._orientation import rpy_to_quaternion
from rbe595_manipulation._wait import wait_for_future
from rbe595_manipulation.poses import DEFAULT_VELOCITY_SCALING

PLANNING_GROUP = "arm"
END_EFFECTOR_LINK = "end_effector_link"
POSITION_TOLERANCE_M = 0.01
ORIENTATION_TOLERANCE_RAD = 0.05

# The arm's base_link position in rbe595_sim's world frame (sim.launch.py's x/y/z launch args
# default to exactly this). MUST match whatever sim.launch.py actually spawned the arm at.
ARM_BASE_WORLD_XYZ = (-0.4, 0.0, 0.75)


def world_to_base_link(position_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(p - b for p, b in zip(position_xyz, ARM_BASE_WORLD_XYZ))


# Found by testing: issuing the next move immediately after "trajectory execution SUCCEEDED"
# can catch the arm mid-settle (momentary overshoot past a joint limit as it decelerates),
# which move_group's next CheckStartStateBounds then rejects outright. A short pause here
# lets /joint_states publish the settled state before the next goal is sent.
SETTLE_TIME_S = 1.0


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

    def move_to(
        self,
        position_xyz: Tuple[float, float, float],
        rpy: Sequence[float],
        velocity_scaling: float = DEFAULT_VELOCITY_SCALING,
        timeout_s: float = 20.0,
    ) -> bool:
        """position_xyz is in rbe595_sim's world frame (see poses.py); converted to base_link
        (MoveIt's actual planning frame) before building the goal -- see ARM_BASE_WORLD_XYZ."""
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
            return False

        send_future = self._client.send_goal_async(goal)
        if not wait_for_future(send_future, timeout_s):
            self._node.get_logger().error("move_group goal send timed out")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("move_group goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, timeout_s):
            self._node.get_logger().error("move_group result timed out")
            return False
        result = result_future.result()
        if result is None:
            return False

        error_code = result.result.error_code.val
        if error_code != MoveItErrorCodes.SUCCESS:
            self._node.get_logger().error(f"move_group failed: error_code={error_code}")
            return False
        time.sleep(SETTLE_TIME_S)
        return True

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

        return constraints
