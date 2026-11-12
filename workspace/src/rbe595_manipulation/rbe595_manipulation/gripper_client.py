import time

from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

from rbe595_manipulation._wait import wait_for_future

# gen3_lite_2f: one actuated joint (right_finger_bottom_joint), the other finger joints mimic it.
# Its controller is a position_controllers/GripperActionController, so the max_effort below is
# only what the controller reports a stall against, not a force limit (URDF effort limit: 10).
GRIPPER_JOINT = "right_finger_bottom_joint"
GRIPPER_MAX_EFFORT = 10.0
# Let the fingers, and anything they just let go of or closed on, come to rest before the arm's
# next move is planned from the measured joint state.
SETTLE_TIME_S = 1.0


class GripperClient:
    """Thin wrapper around the gen3_lite_2f_gripper_controller's GripperCommand action."""

    def __init__(
        self, node: Node, action_name: str = "/gen3_lite_2f_gripper_controller/gripper_cmd"
    ):
        self._node = node
        self._client = ActionClient(node, GripperCommand, action_name)
        self._finger_position = None
        node.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

    def _on_joint_states(self, msg: JointState) -> None:
        if GRIPPER_JOINT in msg.name:
            self._finger_position = msg.position[msg.name.index(GRIPPER_JOINT)]

    def grasp(self, closed: float, holding_max: float) -> bool:
        """Close to `closed` and check something is actually held: fingers closing on a bottle
        stop at its width, short of `holding_max`; past it they closed on nothing."""
        if not self.move_to(closed):
            return False
        if self._finger_position is None:
            self._node.get_logger().error("no /joint_states for the gripper")
            return False
        if self._finger_position >= holding_max:
            self._node.get_logger().error(
                f"grasp missed: fingers closed to {self._finger_position:.3f} "
                f"(holding stops below {holding_max:.2f})"
            )
            return False
        return True

    def move_to(
        self, position: float, timeout_s: float = 5.0, result_timeout_s: float = 15.0
    ) -> bool:
        if not self._client.wait_for_server(timeout_sec=timeout_s):
            self._node.get_logger().error("gripper_cmd action server not available")
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = GRIPPER_MAX_EFFORT

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
        time.sleep(SETTLE_TIME_S)
        # Closing on a bottle stops short of the commanded position: the controller reports
        # that as stalled (allow_stalling: true), which is the expected outcome of a grasp.
        return bool(result.result.reached_goal or result.result.stalled)
