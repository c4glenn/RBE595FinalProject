"""Registers the static obstacles rbe595_sim spawns (table, rack) with MoveIt's planning
scene, in base_link frame"""

import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from rbe595_manipulation.arm_client import world_to_base_link

TABLE_TOP_CENTER_WORLD = (0.0, 0.0, 0.725)
TABLE_TOP_SIZE = (1.2, 1.5, 0.05)

RACK_CENTER_WORLD = (0.1, 0.0, 0.775)  # (0.75 spawn z + 0.80 wall-top z) / 2
RACK_SIZE = (0.34, 0.44, 0.05)  # half-extents from random_bowl_position()'s own formula, x2


def _box(frame_id: str, object_id: str, center_world, size) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(size)
    obj.primitives = [primitive]
    pose = obj.pose
    cx, cy, cz = world_to_base_link(center_world)
    pose.position.x, pose.position.y, pose.position.z = cx, cy, cz
    pose.orientation.w = 1.0
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return obj


def add_known_obstacles(node: Node, timeout_s: float = 10.0) -> bool:
    """Call once at startup, before the node's executor starts spinning -- this uses
    rclpy.spin_until_future_complete() (safe only because nothing else is spinning the node
    yet; see arm_client.py/gripper_client.py for why calls made *during* spin() must not do
    that)."""
    client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
    if not client.wait_for_service(timeout_sec=timeout_s):
        node.get_logger().error("/apply_planning_scene service not available")
        return False

    scene = PlanningScene()
    scene.is_diff = True
    scene.world.collision_objects = [
        _box("base_link", "table_top", TABLE_TOP_CENTER_WORLD, TABLE_TOP_SIZE),
        _box("base_link", "spice_rack", RACK_CENTER_WORLD, RACK_SIZE),
    ]

    request = ApplyPlanningScene.Request()
    request.scene = scene
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)

    result = future.result()
    if result is None or not result.success:
        node.get_logger().error("apply_planning_scene call failed")
        return False

    node.get_logger().info("added table_top and spice_rack collision objects to planning scene")
    return True
