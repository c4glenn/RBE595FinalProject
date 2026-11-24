"""What MoveIt should not hit besides the arm itself: the table, and the D405 on the wrist.

Without these MoveIt plans in an empty world: a free move (RRTConnect) or an IK answer can put an
elbow or the wrist through the table, and RViz shows that plan as fine. Sent once with the
apply_planning_scene service (move_group's default capability); RViz's MotionPlanning display then
shows the table (and the camera box on the wrist) as green scene geometry.

  config/real.yaml  scene:
    table_top_z   table surface in base_link (rack top z - rack height; spice vision prints the rack top)
    padding       keep the arm this far above the table
    camera        attach a D405-sized box at the hand-eye pose (false: leave it out)
    bowl          [x, y, radius, height] in base_link, or null (it has no tag, so it is not measured)
  + the rack (box, at its measured tags) and the regrasp stand (plate + column, at its measured
    pocket) once step 1 has seen them, like the sim's scene_setup.py. Their tops are `sink` lower
    than the real ones, so the fingertips may come close to them around the jar.
"""
import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneWorld
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

TABLE_ID, CAMERA_ID, RACK_ID, BOWL_ID = 'table', 'd405_camera', 'spice_rack', 'bowl'
STAND_PLATE_ID, STAND_COLUMN_ID = 'regrasp_stand_plate', 'regrasp_stand_column'
# The printed stand (rbe595_sim/config/regrasp_stand.yaml, from its CAD): 136 x 100 x 6 mm plate
# centred 18 mm from the column towards the tag; 32 mm-radius column whose top is 20 mm above the
# pocket floor (the jar's seat), which is 41 mm above the table.
STAND_PLATE = (0.136, 0.100, 0.006)
STAND_PLATE_CENTRE_X = 0.018
STAND_COLUMN_RADIUS = 0.032
STAND_TOP_ABOVE_SEAT = 0.020
STAND_SEAT_ABOVE_TABLE = 0.041
SINK = 0.01
# Links the camera box sits on / next to. Allowed to touch it, so the start state is never
# "in collision" because of the box itself (Gen3 lite + its 2-finger gripper).
CAMERA_TOUCH_LINKS = ['upper_wrist_link', 'lower_wrist_link', 'end_effector_link', 'dummy_link', 'tool_frame',
                      'gripper_base_link', 'right_finger_prox_link', 'right_finger_dist_link',
                      'left_finger_prox_link', 'left_finger_dist_link']
try:
    from helpers.geometry import CAMERA_BOX, CAMERA_BOX_CENTRE
except ImportError:                                    # run from helpers/ (go_home.py)
    from geometry import CAMERA_BOX, CAMERA_BOX_CENTRE


def quat_rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def box(dims, xyz, quat=(0.0, 0.0, 0.0, 1.0)):
    prim = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(d) for d in dims])
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in quat)
    return prim, pose


def table_object(base_frame, top_z, padding=0.01, size=(1.8, 2.0), thickness=0.10, x_from=-0.30):
    """A big box whose top is padding above the table surface (below base_link's z = 0)."""
    co = CollisionObject(id=TABLE_ID, operation=CollisionObject.ADD)
    co.header.frame_id = base_frame
    top = top_z + padding
    prim, pose = box((size[0], size[1], thickness), (x_from + size[0] / 2, 0.0, top - thickness / 2))
    co.primitives, co.primitive_poses = [prim], [pose]
    return co


def yaw_quat(yaw):
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def rack_object(base_frame, rack, size, sink=SINK):
    """rack: {'centre': top centre xyz, 'x_axis': its +x, ...} as step 1 measures it."""
    centre, x = np.asarray(rack['centre'], float), np.asarray(rack['x_axis'], float)
    h = size[2] - sink
    co = CollisionObject(id=RACK_ID, operation=CollisionObject.ADD)
    co.header.frame_id = base_frame
    prim, pose = box((size[0], size[1], h), (centre[0], centre[1], centre[2] - sink - h / 2),
                     yaw_quat(np.arctan2(x[1], x[0])))
    co.primitives, co.primitive_poses = [prim], [pose]
    return co


def stand_objects(base_frame, seat, tag_xy, sink=SINK):
    """Plate (towards the tag) and column (round the pocket) of the stand whose pocket floor is at seat."""
    seat = np.asarray(seat, float)
    d = np.asarray(tag_xy, float)[:2] - seat[:2]
    d = d / max(np.linalg.norm(d), 1e-9)
    table_z = seat[2] - STAND_SEAT_ABOVE_TABLE
    plate = CollisionObject(id=STAND_PLATE_ID, operation=CollisionObject.ADD)
    plate.header.frame_id = base_frame
    c = seat[:2] + STAND_PLATE_CENTRE_X * d
    prim, pose = box(STAND_PLATE, (c[0], c[1], table_z + STAND_PLATE[2] / 2), yaw_quat(np.arctan2(d[1], d[0])))
    plate.primitives, plate.primitive_poses = [prim], [pose]
    column = CollisionObject(id=STAND_COLUMN_ID, operation=CollisionObject.ADD)
    column.header.frame_id = base_frame
    bottom, top = table_z + STAND_PLATE[2], seat[2] + STAND_TOP_ABOVE_SEAT - sink
    cyl = SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[float(top - bottom), STAND_COLUMN_RADIUS])
    _, pose = box((0, 0, 0), (seat[0], seat[1], (top + bottom) / 2))
    column.primitives, column.primitive_poses = [cyl], [pose]
    return [plate, column]


def bowl_object(base_frame, bowl, table_z):
    """bowl: [x, y, radius, height] standing on table_z, or [x, y, radius, height, rim_z] (measured
    rim: the cylinder goes from there down by height, wherever the table really is under it)."""
    x, y, r, h = (float(v) for v in bowl[:4])
    top = float(bowl[4]) if len(bowl) > 4 else table_z + h
    co = CollisionObject(id=BOWL_ID, operation=CollisionObject.ADD)
    co.header.frame_id = base_frame
    _, pose = box((0, 0, 0), (x, y, top - h / 2))
    co.primitives = [SolidPrimitive(type=SolidPrimitive.CYLINDER, dimensions=[h, r])]
    co.primitive_poses = [pose]
    return co


def camera_object(ee_link, hand_eye):
    """The D405 as a box attached to the end effector at the hand-eye pose."""
    q = [float(v) for v in hand_eye['rotation']]
    centre = np.asarray(hand_eye['translation'], float) + quat_rotate(q, CAMERA_BOX_CENTRE)
    co = CollisionObject(id=CAMERA_ID, operation=CollisionObject.ADD)
    co.header.frame_id = ee_link
    prim, pose = box(CAMERA_BOX, centre, q)
    co.primitives, co.primitive_poses = [prim], [pose]
    return AttachedCollisionObject(link_name=ee_link, object=co, touch_links=list(CAMERA_TOUCH_LINKS))


def scene_diff(cfg, table_top_z=None, rack=None, stand=None, bowl=None):
    """PlanningScene diff: the table, the camera (unless scene.camera is false), the bowl (if
    scene.bowl), the rack (rack: as in the task state) and the stand (stand: (seat, tag_xy)).
    bowl: (x, y, radius, height, rim z) as step 1 measured it; else scene.bowl, if set."""
    sc = cfg.get('scene') or {}
    z = sc.get('table_top_z') if table_top_z is None else table_top_z
    base = cfg['base_frame']
    ps = PlanningScene(is_diff=True)
    ps.robot_state.is_diff = True
    objs = []
    if z is not None:
        objs.append(table_object(base, float(z), float(sc.get('padding', 0.01))))
        if bowl or sc.get('bowl'):
            objs.append(bowl_object(base, bowl or sc['bowl'], float(z)))
    if rack:
        objs.append(rack_object(base, rack, cfg['rack']['size']))
    if stand:
        objs.extend(stand_objects(base, *stand))
    ps.world = PlanningSceneWorld(collision_objects=objs)
    if sc.get('camera', True) and cfg.get('hand_eye'):
        ps.robot_state.attached_collision_objects = [camera_object(cfg['ee_link'], cfg['hand_eye'])]
    return ps


def apply_scene(node, cfg, table_top_z=None, timeout=10.0, log=None, rack=None, stand=None, bowl=None):
    """Send the scene to move_group. True if it took (False: no service / refused)."""
    import time
    log = log or node.get_logger().info
    client = node.create_client(ApplyPlanningScene, 'apply_planning_scene')
    if not client.wait_for_service(timeout_sec=timeout):
        log('apply_planning_scene not available: MoveIt does not know the table (plans may go through it)')
        return False
    fut = client.call_async(ApplyPlanningScene.Request(scene=scene_diff(cfg, table_top_z, rack, stand, bowl)))
    end = time.monotonic() + timeout
    while not fut.done() and time.monotonic() < end:
        time.sleep(0.02)
    ok = fut.done() and fut.result() is not None and fut.result().success
    z = (cfg.get('scene') or {}).get('table_top_z') if table_top_z is None else table_top_z
    what = ['camera box on the wrist'] + (['rack'] if rack else []) + (['stand'] if stand else []) + (
        ['bowl'] if bowl or (cfg.get('scene') or {}).get('bowl') else [])
    log(f'MoveIt scene: table top at z {z:+.3f} (+ padding), ' + ', '.join(what)
        if ok else 'MoveIt did not take the scene (table / camera / rack / stand): plans may hit them')
    return ok
