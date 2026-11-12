
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node

from rbe595_manipulation._wait import wait_for_future


class GripperClient:
    """Client for the Kinova Gen3 Lite gripper controller."""

    def __init__(
        self,
        node: Node,
        action_name: str = "/gen3_lite_2f_gripper_controller/gripper_cmd",
    ):
        self._node = node
        self._client = ActionClient(node, GripperCommand, action_name)

    def move_to(
        self,
        position: float,
        timeout_s: float = 5.0,
        result_timeout_s: float = 15.0,
    ) -> bool:
        """Move the gripper to a commanded position.

        Position range confirmed on the hardware: approximately 0.0 to 0.8.
        """

        if not self._client.wait_for_server(timeout_sec=timeout_s):
            self._node.get_logger().error(
                "Kinova gripper action server not available"
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = float(position)
        goal.command.max_effort = 20.0

        send_future = self._client.send_goal_async(goal)

        if not wait_for_future(send_future, timeout_s):
            self._node.get_logger().error("Gripper goal send timed out")
            return False

        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().error("Gripper goal rejected")
            return False

        result_future = goal_handle.get_result_async()

        if not wait_for_future(result_future, result_timeout_s):
            self._node.get_logger().error("Gripper result timed out")
            return False

        result = result_future.result()

        if result is None:
            self._node.get_logger().error("Gripper returned no result")
            return False

        gripper_result = result.result

        self._node.get_logger().info(
            f"Gripper result: position={gripper_result.position:.3f}, "
            f"effort={gripper_result.effort:.2f}, "
            f"reached_goal={gripper_result.reached_goal}, "
            f"stalled={gripper_result.stalled}"
        )

        return bool(gripper_result.reached_goal or gripper_result.stalled)
