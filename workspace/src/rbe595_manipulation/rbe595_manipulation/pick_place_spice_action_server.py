import re

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rbe595_manipulation.arm_client import ArmMoveClient
from rbe595_manipulation.context import StepContext
from rbe595_manipulation.gripper_client import GripperClient
from rbe595_manipulation.poses import CAP_TAG_IDS, TAG_SIZES
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
from rbe595_manipulation.tag_locator import TagLocator
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


def parse_spice_id(spice_id: str):
    """The cap tag ID a goal's spice_id names: "3", "tag3" or "tag_3"; None if it names none."""
    match = re.fullmatch(r"(?:tag_?)?(\d+)", spice_id.strip().lower())
    if match is None or int(match.group(1)) not in CAP_TAG_IDS:
        return None
    return int(match.group(1))


class PickPlaceSpiceActionServer(Node):
    def __init__(self):
        super().__init__("pick_place_spice_action_server")
        if not add_known_obstacles(self):
            self.get_logger().warn(
                "failed to register table/rack collision objects -- planning will be "
                "blind to them"
            )
        # Step 5 (shaking over the bowl) is developed separately; leave it out until it is ready.
        shake = self.declare_parameter("shake", False).value
        # Wrist camera (D405): the sim's bridged topics by default; on the real arm,
        # realsense2_camera's colour topics and optical frame.
        image_topic = self.declare_parameter("image_topic", "/wrist_camera/image").value
        camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/wrist_camera/camera_info"
        ).value
        camera_frame = self.declare_parameter("camera_frame", "camera_color_optical_frame").value
        self._ctx = StepContext(
            arm=ArmMoveClient(self),
            gripper=GripperClient(self),
            bottle=BottleScene(self),
            tags=TagLocator(self, TAG_SIZES, image_topic, camera_info_topic, camera_frame),
            logger=self.get_logger(),
            shake=shake,
        )
        self._server = ActionServer(
            self,
            PickPlaceSpice,
            "pick_place_spice",
            execute_callback=self._execute,
            callback_group=ReentrantCallbackGroup(),
        )

    def _execute(self, goal_handle):
        spice_id = goal_handle.request.spice_id
        tag_id = parse_spice_id(spice_id)
        if tag_id is None:
            self.get_logger().error(
                f"spice_id {spice_id!r} is not a cap tag ID ({min(CAP_TAG_IDS)}-{max(CAP_TAG_IDS)})"
            )
            goal_handle.abort()
            result = PickPlaceSpice.Result()
            result.success = False
            result.message = f"unknown spice_id {spice_id!r}"
            return result
        self.get_logger().info(f"fetching the jar with cap tag {tag_id}")
        self._ctx.target_tag = tag_id
        self._ctx.rack_jar_top = None
        self._ctx.stand_seat = None
        self._ctx.staging_jar_top = None
        total = len(STEPS)
        for index, step in enumerate(STEPS, start=1):
            if step is step5_shake_over_bowl and not self._ctx.shake:
                self.get_logger().info(f"skipping step {step.NAME} (shake:=false)")
                continue
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
    # The goal blocks one thread; the others serve MoveIt's replies, the camera and /tf.
    executor = MultiThreadedExecutor(num_threads=6)
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
