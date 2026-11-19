"""Picks the bottle a PickPlaceSpice goal targets from the spice detector's output
(spice_detector_node.py), retargets the rack poses at it, and swaps MoveIt's rack-bottle
obstacles for the bottles actually seen.

Which bottle:
  - goal spice_id is a number n -> bottle n as numbered by the detector (see its image);
  - spice_id empty (or not a number) -> the detector's selected bottle
    (ros2 param set /spice_detector selected_bottle n);
  - no detector running / nothing selected -> the hardcoded slot in poses.py, as before.

Grasp heights still assume the bottle stands in the rack.
"""

import math
import time

from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from geometry_msgs.msg import PoseArray, PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node

from rbe595_manipulation import poses
from rbe595_manipulation._wait import wait_for_future
from rbe595_manipulation.scene_setup import BOTTLE_RADIUS, RACK_SLOTS_XY, _cylinder

BOTTLES_TOPIC = "/spice_detector/bottles"
SELECTED_TOPIC = "/spice_detector/selected_bottle"
MAX_AGE_S = 3.0  # detections older than this (wall clock) are ignored
SAME_BOTTLE_M = 0.02  # detections closer than this are the same bottle
# Prefixes of the rack-bottle obstacles: scene_setup.py's startup ones, and ours.
OBSTACLE_PREFIXES = ("rack_bottle_", "detected_bottle_")

# Open gen3_lite_2f fingers, seen from above (rbe595_sim config/gen3_lite_2f_finger_collision.yaml):
# ~16 mm thick pads ~85 mm apart at GRIPPER_OPEN, so each finger's centre is ~50 mm from the
# bottle axis; each is ~24 mm wide across the closing direction.
FINGER_CENTRE_M = 0.050
FINGER_HALF_WIDTH_M = 0.012


def finger_clearance(target_xy, others_xy, closing_angle):
    """Smallest gap [m] between either open finger (a segment across the closing direction)
    and the side of any other bottle."""
    ux, uy = math.cos(closing_angle), math.sin(closing_angle)
    vx, vy = -uy, ux
    gap = math.inf
    for side in (1.0, -1.0):
        cx = target_xy[0] + side * FINGER_CENTRE_M * ux
        cy = target_xy[1] + side * FINGER_CENTRE_M * uy
        for nx, ny in others_xy:
            dx, dy = nx - cx, ny - cy
            along = dx * vx + dy * vy
            along_clamped = max(-FINGER_HALF_WIDTH_M, min(FINGER_HALF_WIDTH_M, along))
            dist = math.hypot(dx * ux + dy * uy, along - along_clamped)
            gap = min(gap, dist - BOTTLE_RADIUS)
    return gap


def best_closing_angle(target_xy, others_xy):
    """The finger closing direction (world, rad, in (-90, 90] deg) that leaves the most room
    between the open fingers and the neighbouring bottles: in a rack grid, a diagonal one,
    between the neighbours, rather than straight at the nearest. None if no neighbours."""
    if not others_xy:
        return None
    # 1 deg steps; on a tie keep the one nearest closing along x (the original grasp).
    angles = sorted(range(-89, 91), key=abs)
    best = max(angles, key=lambda d: round(finger_clearance(target_xy, others_xy, math.radians(d)), 4))
    return math.radians(best)


class DetectedTarget:
    """Blocking like BottleScene: call only from a callback the executor is not blocked on."""

    def __init__(self, node: Node):
        self._node = node
        self._log = node.get_logger()
        self._bottles = None  # (receipt time, PoseArray)
        self._selected = None  # (receipt time, PoseStamped)
        group = MutuallyExclusiveCallbackGroup()
        node.create_subscription(
            PoseArray, BOTTLES_TOPIC, self._on_bottles, 1, callback_group=group
        )
        node.create_subscription(
            PoseStamped, SELECTED_TOPIC, self._on_selected, 1, callback_group=group
        )
        self._get_scene = node.create_client(GetPlanningScene, "/get_planning_scene")
        self._apply_scene = node.create_client(ApplyPlanningScene, "/apply_planning_scene")

    def _on_bottles(self, msg):
        self._bottles = (time.monotonic(), msg)

    def _on_selected(self, msg):
        self._selected = (time.monotonic(), msg)

    @staticmethod
    def _fresh(entry):
        return entry is not None and time.monotonic() - entry[0] < MAX_AGE_S

    def apply(self, spice_id: str):
        """Retarget poses.RACK_* and the planning scene for this goal.
        Returns (ok, message); not ok means the goal should be aborted."""
        spice_id = spice_id.strip()
        bottles = [
            (p.position.x, p.position.y, p.position.z) for p in self._bottles[1].poses
        ] if self._fresh(self._bottles) else None

        if spice_id.lstrip("-").isdigit():
            n = int(spice_id)
            if bottles is None:
                return False, f"bottle {n} requested, but no detections on {BOTTLES_TOPIC}"
            if not 0 <= n < len(bottles):
                return False, f"bottle {n} requested, but only {len(bottles)} detected"
            target = bottles[n]
            how = f"bottle {n} (goal spice_id)"
        elif self._fresh(self._selected) and bottles is not None:
            p = self._selected[1].pose.position
            target = (p.x, p.y, p.z)
            how = "the spice detector's selected bottle"
        else:
            self._log.warn(
                "no bottle number in the goal and no selection from the spice detector; "
                f"using the hardcoded rack slot {poses.RACK_SLOT_XY}"
            )
            return True, "hardcoded slot"

        others = [b for b in bottles if math.dist(b[:2], target[:2]) > SAME_BOTTLE_M]
        slot = min(RACK_SLOTS_XY, key=lambda s: math.dist(s, target[:2]))
        if math.dist(slot, target[:2]) <= SAME_BOTTLE_M:
            # In a rack pocket: the detection says which one, the rack geometry says exactly
            # where. A jar has only ~1.4 mm of play in its pocket, less than the detection's
            # error, and is put back where it was picked from.
            target = (slot[0], slot[1], target[2])
        else:
            self._log.warn(
                f"target ({target[0]:.3f}, {target[1]:.3f}) is "
                f"{math.dist(slot, target[:2]) * 1000:.0f} mm from the nearest rack slot; "
                "grasp heights assume it stands in the rack"
            )
        others_xy = [o[:2] for o in others]
        angle = best_closing_angle(target[:2], others_xy)
        poses.set_rack_slot(target[:2], angle)
        if angle is not None:
            grasp = (
                f"fingers closing at {math.degrees(angle):.0f} deg, "
                f"{finger_clearance(target[:2], others_xy, angle) * 1000:.0f} mm from the "
                f"nearest neighbour (closing along x: "
                f"{finger_clearance(target[:2], others_xy, 0.0) * 1000:.0f} mm)"
            )
        else:
            grasp = "no neighbours, default grasp"
        self._log.info(
            f"targeting {how} at ({target[0]:.3f}, {target[1]:.3f}); {grasp}; "
            f"{len(others)} other bottles as obstacles"
        )
        if not self._replace_obstacles(others):
            return False, "could not update the planning scene with the detected bottles"
        return True, how

    def _replace_obstacles(self, others, timeout_s: float = 10.0) -> bool:
        if not self._get_scene.wait_for_service(timeout_sec=timeout_s):
            self._log.error("/get_planning_scene service not available")
            return False
        request = GetPlanningScene.Request()
        request.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        future = self._get_scene.call_async(request)
        if not wait_for_future(future, timeout_s) or future.result() is None:
            self._log.error("get_planning_scene call timed out")
            return False

        scene = PlanningScene()
        scene.is_diff = True
        for obj in future.result().scene.world.collision_objects:
            if obj.id.startswith(OBSTACLE_PREFIXES):
                remove = CollisionObject()
                remove.header.frame_id = "base_link"
                remove.id = obj.id
                remove.operation = CollisionObject.REMOVE
                scene.world.collision_objects.append(remove)
        for i, (x, y, top_z) in enumerate(others):
            scene.world.collision_objects.append(
                _cylinder(
                    "base_link",
                    f"detected_bottle_{i}",
                    (x, y, top_z - poses.BOTTLE_HEIGHT / 2),
                    BOTTLE_RADIUS,
                    poses.BOTTLE_HEIGHT,
                )
            )

        if not self._apply_scene.wait_for_service(timeout_sec=timeout_s):
            self._log.error("/apply_planning_scene service not available")
            return False
        apply = ApplyPlanningScene.Request()
        apply.scene = scene
        future = self._apply_scene.call_async(apply)
        if not wait_for_future(future, timeout_s) or future.result() is None:
            self._log.error("apply_planning_scene call timed out")
            return False
        return future.result().success
