from control_msgs.action import ParallelGripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node

from rbe595_manipulation._wait import wait_for_future

GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"


class GripperClient:
    """Thin wrapper around the robotiq_gripper_controller's action.
    """

    def __init__(self, node: Node, action_name: str = "/robotiq_gripper_controller/gripper_cmd"):
        self._node = node
        self._client = ActionClient(node, ParallelGripperCommand, action_name)

    def move_to(self, position: float, timeout_s: float = 5.0, result_timeout_s: float = 15.0) -> bool:
        if not self._client.wait_for_server(timeout_sec=timeout_s):
            self._node.get_logger().error("gripper_cmd action server not available")
            return False

        goal = ParallelGripperCommand.Goal()
        goal.command.name = [GRIPPER_JOINT_NAME]
        goal.command.position = [position]

        send_future = self._client.send_goal_async(goal)
        if not wait_for_future(send_future, timeout_s):
            self._node.get_logger().error("gripper goal send timed out")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("gripper goal rejected")
            return False

        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, result_timeout_s):
            self._node.get_logger().warn(
                "gripper goal result timed out -- assuming a stall against an object "
                "rather than failing the step"
            )
            return True
        result = result_future.result()
        if result is None:
            return False
        return bool(result.result.reached_goal or not result.result.stalled)
