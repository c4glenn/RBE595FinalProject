"""Registers the static obstacles rbe595_sim spawns (table, rack, the bottles left in the rack,
bowl) with MoveIt's planning scene, in base_link frame"""

import math

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive

from rbe595_manipulation import poses
from rbe595_manipulation._wait import wait_for_future
from rbe595_manipulation.arm_client import world_to_base_link

TABLE_TOP_CENTER_WORLD = (0.0, 0.0, 0.725)
TABLE_TOP_SIZE = (1.2, 1.5, 0.05)

RACK_CENTER_WORLD = (0.1, 0.0, 0.762)  # (0.75 spawn z + 0.774 rack-top z) / 2
RACK_SIZE = (0.231, 0.209, 0.024)  # len_x, len_y, rack_height (config/spice_rack.yaml)

# Slot centres (sim.launch.py spice_rack_slots(), rack at x=0.1, y=0.0): two columns along x,
# three rows along y. Every slot but the one being picked from holds a bottle.
RACK_SLOTS_XY = [(x, y) for x in (0.068, 0.164) for y in (-0.067, 0.0, 0.067)]
BOTTLE_RADIUS = 0.0483 / 2
BOWL_RADIUS = 0.075  # sim.launch.py BOWL_RADIUS / BOWL_HEIGHT
BOWL_HEIGHT = 0.065


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


def _cylinder(frame_id: str, object_id: str, center_world, radius, height) -> CollisionObject:
    obj = _box(frame_id, object_id, center_world, (0.0, 0.0, 0.0))
    obj.primitives[0].type = SolidPrimitive.CYLINDER
    obj.primitives[0].dimensions = [height, radius]
    return obj


def _rack_bottles(frame_id: str):
    target = poses.RACK_SLOT_XY
    return [
        _cylinder(
            frame_id,
            f"rack_bottle_{i}",
            (x, y, poses.BOTTLE_BASE_Z + poses.BOTTLE_HEIGHT / 2),
            BOTTLE_RADIUS,
            poses.BOTTLE_HEIGHT,
        )
        for i, (x, y) in enumerate(RACK_SLOTS_XY)
        if abs(x - target[0]) > 1e-3 or abs(y - target[1]) > 1e-3
    ]


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
        *_rack_bottles("base_link"),
        _cylinder(
            "base_link",
            "bowl",
            (poses.BOWL_XY[0], poses.BOWL_XY[1], poses.TABLE_TOP_Z + BOWL_HEIGHT / 2),
            BOWL_RADIUS,
            BOWL_HEIGHT,
        ),
    ]

    request = ApplyPlanningScene.Request()
    request.scene = scene
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)

    result = future.result()
    if result is None or not result.success:
        node.get_logger().error("apply_planning_scene call failed")
        return False

    node.get_logger().info(
        f"added {len(scene.world.collision_objects)} collision objects to the planning scene "
        "(table, rack, bottles left in the rack, bowl)"
    )
    return True


BOTTLE_ID = "bottle"  # the bottle being moved: held, or standing at the staging spot
STANDING_PADDING = 0.02  # [m]

# The gen3_lite_2f links allowed to touch a held bottle.
GRIPPER_LINKS = [
    "end_effector_link",
    "gripper_base_link",
    "left_finger_prox_link",
    "left_finger_dist_link",
    "right_finger_prox_link",
    "right_finger_dist_link",
]


class BottleScene:
    """Keeps MoveIt's planning scene aware of the bottle being moved, so free-space moves plan
    around it:

    - while held, it is attached to end_effector_link (at the top or side grasp), so the arm
      does not swing it through the table, the rack or the bowl;
    - while it stands at the staging spot, it is an obstacle there, so moves round it (e.g. from
      the top-grasp pose to the side-grasp approach) do not sweep through it.

    The straight-line legs into and out of the rack pocket and down onto / off the table run with
    the bottle out of the scene, since they end touching the rack or table.

    Blocking like ArmMoveClient: call only from a callback the node's executor is not blocked on.
    """

    def __init__(self, node: Node):
        self._node = node
        self._client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def hold(self, grasp: str, timeout_s: float = 10.0) -> bool:
        """Attach the bottle to the gripper, held by the top ("top") or by the side ("side")."""
        attached = AttachedCollisionObject()
        attached.link_name = "end_effector_link"
        attached.touch_links = GRIPPER_LINKS
        obj = attached.object
        obj.header.frame_id = "end_effector_link"
        obj.id = BOTTLE_ID
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.CYLINDER
        primitive.dimensions = [poses.BOTTLE_HEIGHT, BOTTLE_RADIUS]
        obj.primitives = [primitive]
        pose = Pose()
        reach = poses.GRIPPER_FINGERTIP_REACH
        if grasp == "top":
            # bottle axis along the approach (end-effector z), its top TOP_GRASP_BELOW_TOP
            # above the grasp point
            pose.position.z = reach + poses.BOTTLE_HEIGHT / 2 - poses.TOP_GRASP_BELOW_TOP
            pose.orientation.w = 1.0
        else:
            # bottle axis along end-effector x (pointing to its base), its top
            # SIDE_GRASP_BELOW_TOP from the grasp point
            pose.position.x = poses.BOTTLE_HEIGHT / 2 - poses.SIDE_GRASP_BELOW_TOP
            pose.position.z = reach
            pose.orientation.y = math.sin(math.pi / 4)  # cylinder z onto end-effector x
            pose.orientation.w = math.cos(math.pi / 4)
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        return self._apply(scene, timeout_s)

    def let_go(self, timeout_s: float = 10.0) -> bool:
        """Drop the held bottle from the scene (just before setting it down)."""
        attached = AttachedCollisionObject()
        attached.link_name = "end_effector_link"
        attached.object.id = BOTTLE_ID
        attached.object.operation = CollisionObject.REMOVE
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        # Detaching puts the object back into the world where it is (in the gripper); remove
        # that copy too.
        scene.world.collision_objects = [self._remove_from_world()]
        return self._apply(scene, timeout_s)

    def standing_at_staging(self, timeout_s: float = 10.0) -> bool:
        """The bottle now stands at the staging spot. Padded by STANDING_PADDING, since the
        free-space moves round it are only collision-checked at discrete points and the
        straight-line reach in to grasp it runs with it out of the scene anyway."""
        scene = PlanningScene()
        scene.is_diff = True
        height = poses.BOTTLE_HEIGHT + STANDING_PADDING
        scene.world.collision_objects = [
            _cylinder(
                "base_link",
                BOTTLE_ID,
                (poses.STAGING_XY[0], poses.STAGING_XY[1], poses.TABLE_TOP_Z + height / 2),
                BOTTLE_RADIUS + STANDING_PADDING,
                height,
            )
        ]
        return self._apply(scene, timeout_s)

    def clear(self, timeout_s: float = 10.0) -> bool:
        """Take the standing bottle out of the scene (just before reaching in to grasp it)."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [self._remove_from_world()]
        return self._apply(scene, timeout_s)

    @staticmethod
    def _remove_from_world() -> CollisionObject:
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.id = BOTTLE_ID
        obj.operation = CollisionObject.REMOVE
        return obj

    def _apply(self, scene: PlanningScene, timeout_s: float) -> bool:
        if not self._client.wait_for_service(timeout_sec=timeout_s):
            self._node.get_logger().error("/apply_planning_scene service not available")
            return False
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        if not wait_for_future(future, timeout_s) or future.result() is None:
            self._node.get_logger().error("apply_planning_scene call timed out")
            return False
        return future.result().success
