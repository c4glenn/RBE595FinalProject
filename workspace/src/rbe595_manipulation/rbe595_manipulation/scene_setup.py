"""Registers the static obstacles rbe595_sim spawns (table, rack, regrasp stand, bowl) and the wrist
camera with MoveIt's planning scene, in base_link frame; the jars in the rack are added where their
cap tags are seen (BottleScene.set_rack_jars()), the rack and the stand are moved to where their
datum tags are seen (BottleScene.set_rack(), set_stand())."""

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

# Where the rack is set up (rbe595_sim sim.launch.py), until its datum tags are seen
# (BottleScene.set_rack()): centred 0.4 m in front of the arm base, (0.75 spawn z + 0.774
# rack-top z) / 2 high.
RACK_CENTER_WORLD = (0.0, 0.0, 0.762)

BOTTLE_RADIUS = 0.0483 / 2
BOWL_RADIUS = 0.075  # sim.launch.py BOWL_RADIUS / BOWL_HEIGHT
BOWL_HEIGHT = 0.065


def _box(frame_id: str, object_id: str, center_world, size, yaw: float = 0.0) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = frame_id
    obj.id = object_id
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(size)
    obj.primitives = [primitive]
    # The object's pose places it; its primitive's pose is relative to that, so identity.
    cx, cy, cz = world_to_base_link(center_world)
    obj.pose.position.x, obj.pose.position.y, obj.pose.position.z = cx, cy, cz
    obj.pose.orientation.z = math.sin(yaw / 2)
    obj.pose.orientation.w = math.cos(yaw / 2)
    identity = Pose()
    identity.orientation.w = 1.0
    obj.primitive_poses = [identity]
    obj.operation = CollisionObject.ADD
    return obj


def _cylinder(frame_id: str, object_id: str, center_world, radius, height) -> CollisionObject:
    obj = _box(frame_id, object_id, center_world, (0.0, 0.0, 0.0))
    obj.primitives[0].type = SolidPrimitive.CYLINDER
    obj.primitives[0].dimensions = [height, radius]
    return obj


# The D405 body (42 x 42 x 23 mm) behind its lens (poses.CAMERA_IN_EE_XYZ), in
# end_effector_link: 23 mm deep along the approach (z), 42 mm across in x and y.
CAMERA_ID = "wrist_camera"
CAMERA_BOX_SIZE = (0.042, 0.042, 0.023)
CAMERA_BOX_CENTER_EE = (
    poses.CAMERA_IN_EE_XYZ[0],
    poses.CAMERA_IN_EE_XYZ[1],
    poses.CAMERA_IN_EE_XYZ[2] - CAMERA_BOX_SIZE[2] / 2,
)


def _wrist_camera() -> AttachedCollisionObject:
    """The wrist camera, fixed to the hand, so planning keeps it clear of the table, the rack and
    the jars (the arm's own description does not have it)."""
    attached = AttachedCollisionObject()
    attached.link_name = "end_effector_link"
    attached.touch_links = GRIPPER_LINKS + ["upper_wrist_link"]  # its bracket is on the wrist
    obj = attached.object
    obj.header.frame_id = "end_effector_link"
    obj.id = CAMERA_ID
    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(CAMERA_BOX_SIZE)
    obj.primitives = [primitive]
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = CAMERA_BOX_CENTER_EE
    pose.orientation.w = 1.0
    obj.primitive_poses = [pose]
    obj.operation = CollisionObject.ADD
    return attached


STAND_PLATE_ID = "regrasp_stand_plate"
STAND_COLUMN_ID = "regrasp_stand_column"


def _stand(seat_world, yaw: float):
    """The regrasp stand's base plate and column (up to its top: the pocket is left solid, the
    jar standing in it is its own object), for a stand whose pocket floor centre is seat_world."""
    sx, sy, _ = seat_world
    plate_t = poses.STAND_PLATE_SIZE[2]
    table_z = seat_world[2] - poses.STAND_SEAT_ABOVE_TABLE
    plate = _box(
        "base_link",
        STAND_PLATE_ID,
        (
            sx + poses.STAND_PLATE_CENTER_X * math.cos(yaw),
            sy + poses.STAND_PLATE_CENTER_X * math.sin(yaw),
            table_z + plate_t / 2,
        ),
        poses.STAND_PLATE_SIZE,
        yaw,
    )
    column_h = poses.STAND_TOP_ABOVE_TABLE - plate_t
    column = _cylinder(
        "base_link",
        STAND_COLUMN_ID,
        (sx, sy, table_z + plate_t + column_h / 2),
        poses.STAND_COLUMN_RADIUS,
        column_h,
    )
    return [plate, column]


STAND_SEAT_NOMINAL = (
    poses.STAGING_XY[0],
    poses.STAGING_XY[1],
    poses.TABLE_TOP_Z + poses.STAND_SEAT_ABOVE_TABLE,
)


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
        _box("base_link", RACK_ID, RACK_CENTER_WORLD, poses.RACK_SIZE),
        _cylinder(
            "base_link",
            "bowl",
            (poses.BOWL_XY[0], poses.BOWL_XY[1], poses.TABLE_TOP_Z + BOWL_HEIGHT / 2),
            BOWL_RADIUS,
            BOWL_HEIGHT,
        ),
        *_stand(STAND_SEAT_NOMINAL, poses.STAND_YAW_NOMINAL),
    ]
    scene.robot_state.is_diff = True
    scene.robot_state.attached_collision_objects = [_wrist_camera()]

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
        "(table, rack, bowl, regrasp stand) and the wrist camera"
    )
    return True


RACK_ID = "spice_rack"
BOTTLE_ID = "bottle"  # the bottle being moved: held, or standing in the regrasp stand
STANDING_PADDING = 0.02  # [m] round the standing bottle's side
# Above its cap only this much: at the stand the hand waits over the cap with the open fingertips
# ~30 mm above it (poses.STAND_APPROACH_CLEARANCE), and must not start inside the padding.
STANDING_TOP_PADDING = 0.005  # [m]

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
    - while it stands in the regrasp stand, it is an obstacle there, so moves round it (e.g. from
      the top-grasp pose to the side-grasp approach) do not sweep through it.

    The straight-line legs into and out of the rack and stand pockets run with the bottle out of
    the scene, since they end touching the rack or stand.

    Blocking like ArmMoveClient: call only from a callback the node's executor is not blocked on.
    """

    def __init__(self, node: Node):
        self._node = node
        self._client = node.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._rack_jar_ids = []

    def set_rack(self, center_world, yaw: float, timeout_s: float = 10.0) -> bool:
        """Move the rack's collision box to where its datum tags put it."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            _box("base_link", RACK_ID, center_world, poses.RACK_SIZE, yaw)
        ]
        return self._apply(scene, timeout_s)

    def set_rack_jars(self, jar_tops, timeout_s: float = 10.0) -> bool:
        """The jars standing in the rack, other than the one being moved: {tag ID: centre of the
        jar's top (world frame)}, as their cap tags were seen. Replaces the previous set."""
        scene = PlanningScene()
        scene.is_diff = True
        removed = []
        for object_id in self._rack_jar_ids:
            obj = CollisionObject()
            obj.header.frame_id = "base_link"
            obj.id = object_id
            obj.operation = CollisionObject.REMOVE
            removed.append(obj)
        added = [
            _cylinder(
                "base_link",
                f"rack_jar_tag_{tag_id}",
                (x, y, z - poses.BOTTLE_HEIGHT / 2),
                BOTTLE_RADIUS,
                poses.BOTTLE_HEIGHT,
            )
            for tag_id, (x, y, z) in sorted(jar_tops.items())
        ]
        scene.world.collision_objects = removed + added
        self._rack_jar_ids = [obj.id for obj in added]
        return self._apply(scene, timeout_s)

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

    def set_stand(self, seat_world, yaw: float, timeout_s: float = 10.0) -> bool:
        """Move the regrasp stand to where its datum tag puts it."""
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = _stand(seat_world, yaw)
        return self._apply(scene, timeout_s)

    def standing_at(self, top_world, timeout_s: float = 10.0) -> bool:
        """The bottle now stands with the centre of its top at top_world (as its cap tag was seen).
        Padded by STANDING_PADDING round its side (STANDING_TOP_PADDING above it), since the
        free-space moves round it are only collision-checked at discrete points and the
        straight-line reach in to grasp it runs with it out of the scene anyway."""
        scene = PlanningScene()
        scene.is_diff = True
        height = poses.BOTTLE_HEIGHT + STANDING_TOP_PADDING
        scene.world.collision_objects = [
            _cylinder(
                "base_link",
                BOTTLE_ID,
                (top_world[0], top_world[1], top_world[2] + STANDING_TOP_PADDING - height / 2),
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
