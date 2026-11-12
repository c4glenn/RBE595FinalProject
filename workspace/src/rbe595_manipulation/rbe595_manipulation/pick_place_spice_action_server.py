import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.gripper_client import GripperClient
from rbe595_manipulation.scene_setup import BottleScene, add_known_obstacles
from rbe595_manipulation.steps import (
    step1_go_to_rack,
    step2_grasp_top,
    step3_place_staging,
    step4_regrasp_side,
    step5_shake_over_bowl,
    step6_return_to_staging,
    step7_regrasp_top,
    step8_return_to_rack,
)
from rbe595_manipulation_interfaces.action import PickPlaceSpice

STEPS = [
    step1_go_to_rack,
    step2_grasp_top,
    step3_place_staging,
    step4_regrasp_side,
    step5_shake_over_bowl,
    step6_return_to_staging,
    step7_regrasp_top,
    step8_return_to_rack,
]


class PickPlaceSpiceActionServer(Node):
    def __init__(self):
        super().__init__("pick_place_spice_action_server")
        if not add_known_obstacles(self):
            self.get_logger().warn(
                "failed to register table/rack collision objects -- planning will be "
                "blind to them"
            )
        self._ctx = StepContext(
            arm=ArmMoveClient(self),
            gripper=GripperClient(self),
            bottle=BottleScene(self),
            logger=self.get_logger(),
        )
        self._server = ActionServer(
            self,
            PickPlaceSpice,
            "pick_place_spice",
            execute_callback=self._execute,
            callback_group=ReentrantCallbackGroup(),
        )

    def _execute(self, goal_handle):
        self.get_logger().info(
            f"v1 ignores spice_id={goal_handle.request.spice_id!r}; "
            "always targets the hardcoded rack slot"
        )
        total = len(STEPS)
        for index, step in enumerate(STEPS, start=1):
            feedback = PickPlaceSpice.Feedback()
            feedback.step_name = step.NAME
            feedback.step_index = index
            feedback.total_steps = total
            goal_handle.publish_feedback(feedback)
            self.get_logger().info(f"starting step {step.NAME}")

            try:
                ok = step.run(self._ctx)
            except Exception as exc:  # noqa: BLE001 -- any step failure aborts the same way
                self.get_logger().error(f"step {step.NAME} raised: {exc}")
                ok = False

            if not ok:
                goal_handle.abort()
                result = PickPlaceSpice.Result()
                result.success = False
                result.message = f"failed at step {step.NAME}"
                return result

        goal_handle.succeed()
        result = PickPlaceSpice.Result()
        result.success = True
        result.message = "ok"
        return result


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceSpiceActionServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
