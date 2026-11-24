# A pose is [x, y, z, qx, qy, qz, qw]: position (m) + orientatin in the base frame.
# motion = 'cartesian'  
# motion = 'rrtconnect' 
import math
import time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, OrientationConstraint, PositionConstraint
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive

ERROR_HINTS = {-31: ' (no IK solution: pose unreachable)', -10: ' (start state invalid, e.g. joint_3 limit)',
               -1: ' (planning failed)', -4: ' (execution failed)',
               99999: ' (no valid goal state: the pose is out of reach, or no IK solution fits the joint guards)'}
SUCCESS, CONTROL_FAILED = 1, -4

# Joint guards for planned (RRTConnect) moves, from the sim pipeline. The Gen3 Lite reaches most
# poses either facing the target or folded back over its base, elbow up or down, and the IK
# sampler picks one at random; these keep every planned move on the same branch.
#   joint_1: within BASE_TOL of the target's bearing from the base (the arm faces the target).
#            Skipped for targets within BASE_MIN_REACH of the base axis, where the bearing is noise.
#   joint_3: elbow-up (positive) branch, the one the home pose uses.
#   joint_6: stays WRIST_MARGIN inside its limit, so the next move (or twist servoing) has room.
BASE_TOL = 1.0                 # rad
BASE_MIN_REACH = 0.10          # m
ELBOW_RANGE = (0.0, 2.6)       # rad, URDF limit +-2.61
WRIST_LIMIT, WRIST_MARGIN = 2.6, 0.2   # rad

# Sending the next goal right after "SUCCEEDED" can catch the arm still settling (a small
# overshoot past a limit), and MoveIt then rejects the start state. Wait this long after a move.
SETTLE = 1.0                   # s


def parse_pose(entry, what='pose'):
    if isinstance(entry, dict) and 'pose' in entry:
        entry = entry['pose']
    if isinstance(entry, (list, tuple)) and len(entry) == 7:
        return [float(v) for v in entry]
    raise ValueError(f'{what}: expected a pose [x, y, z, qx, qy, qz, qw] (7 numbers), got {entry!r}')


def list_to_pose(p):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p[:3])
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in p[3:])
    return pose


def max_joint_jump(trajectory):
    points = trajectory.joint_trajectory.points
    jump = 0.0
    for a, b in zip(points, points[1:]):
        jump = max(jump, max(abs(q1 - q0) for q0, q1 in zip(a.positions, b.positions)))
    return jump


def wait_for(future, timeout):
    start = time.time()
    while not future.done():
        if time.time() - start > timeout:
            return None
        time.sleep(0.01)
    return future.result()


class ArmMover:
    def __init__(self, node, group='arm', ee_link='end_effector_link', base_frame='base_link', speed=0.1,
                 motion='cartesian', cartesian_fallback=False, max_step=0.005, max_jump=0.5, joint_guards=True,
                 callback_group=None):
        self.node, self.group, self.ee_link, self.base_frame, self.speed = node, group, ee_link, base_frame, speed
        self.motion, self.cartesian_fallback = motion, cartesian_fallback
        self.max_step = max_step          # m between interpolated points on a straight-line path
        self.max_jump = max_jump          # rad, largest joint step allowed between path points
        self.joint_guards = joint_guards  # joint_1 / joint_3 / joint_6 constraints on planned moves
        self.move_client = ActionClient(node, MoveGroup, 'move_action', callback_group=callback_group)
        self.exec_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=callback_group)
        self.cart_client = node.create_client(GetCartesianPath, 'compute_cartesian_path',
                                              callback_group=callback_group)
        self.log = node.get_logger()

    def wait_until_ready(self, timeout=120.0):
        deadline = time.time() + timeout
        ok = (self.move_client.wait_for_server(timeout_sec=timeout)
              and self.exec_client.wait_for_server(timeout_sec=max(1.0, deadline - time.time()))
              and self.cart_client.wait_for_service(timeout_sec=max(1.0, deadline - time.time())))
        if not ok:
            self.log.error('MoveIt services not available: is the MoveIt launch running?')
        return ok

    def move_to(self, pose):
        if self.motion == 'cartesian':
            if self.move_cartesian(pose):
                return True
            if not self.cartesian_fallback:
                return False
            self.log.warn('Straight line not possible; falling back to RRTConnect')
        return self.move_rrtconnect(pose)

    def move_cartesian(self, pose):
        return self._run(self._cartesian_once, pose)

    def move_rrtconnect(self, pose):
        return self._run(self._rrtconnect_once, pose)

    # Each *_once returns the MoveIt error code, or None for a failure it already reported.
    def _cartesian_once(self, pose):
        if not self.wait_until_ready(timeout=10.0):
            return None
        req = GetCartesianPath.Request()
        req.header.frame_id, req.group_name, req.link_name = self.base_frame, self.group, self.ee_link
        req.start_state.is_diff = True
        req.waypoints = [list_to_pose(pose)]
        req.max_step = self.max_step
        req.avoid_collisions = True
        req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = self.speed
        res = wait_for(self.cart_client.call_async(req), timeout=15.0)
        if res is None:
            self.log.error('compute_cartesian_path did not answer')
            return None
        if res.fraction < 0.999:
            self.log.error(f'Straight line only {res.fraction:.0%} feasible (joint limit, singularity or '
                           'collision on the way); not moving')
            return None
        jump = max_joint_jump(res.solution)
        if jump > self.max_jump:
            self.log.error(f'Straight line needs a sudden joint jump of {jump:.2f} rad (wrist/elbow flip); '
                           'not moving')
            return None
        return self._send(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution))

    def _rrtconnect_once(self, pose):
        target = list_to_pose(pose)
        position = PositionConstraint()
        position.header.frame_id, position.link_name, position.weight = self.base_frame, self.ee_link, 1.0
        position.constraint_region.primitives.append(
            SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.005]))
        position.constraint_region.primitive_poses.append(target)
        orientation = OrientationConstraint()
        orientation.header.frame_id, orientation.link_name, orientation.weight = self.base_frame, self.ee_link, 1.0
        orientation.orientation = target.orientation
        orientation.absolute_x_axis_tolerance = 0.02
        orientation.absolute_y_axis_tolerance = 0.02
        orientation.absolute_z_axis_tolerance = 0.02

        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.log.error('move_action not available: is the MoveIt launch running?')
            return None
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name, r.pipeline_id, r.planner_id = self.group, 'ompl', 'RRTConnectkConfigDefault'
        r.allowed_planning_time, r.num_planning_attempts = 5.0, 5
        r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = self.speed
        r.start_state.is_diff = True
        r.goal_constraints.append(Constraints(position_constraints=[position],
                                              orientation_constraints=[orientation],
                                              joint_constraints=self._joint_guards(pose)))
        goal.planning_options.plan_only = False
        return self._send(self.move_client, goal)

    def _joint_guards(self, pose):
        """Joint constraints that keep a planned move on the arm's usual branch (see BASE_TOL etc.).
        pose must be in base_link (the joint_1 bearing is measured from the base axis)."""
        if not self.joint_guards:
            return []

        def window(name, low, high):
            return JointConstraint(joint_name=name, position=(low + high) / 2, tolerance_above=(high - low) / 2,
                                   tolerance_below=(high - low) / 2, weight=1.0)

        guards = [window('joint_3', *ELBOW_RANGE),
                  window('joint_6', -WRIST_LIMIT + WRIST_MARGIN, WRIST_LIMIT - WRIST_MARGIN)]
        x, y = pose[0], pose[1]
        if self.base_frame == 'base_link' and math.hypot(x, y) > BASE_MIN_REACH:
            bearing = math.atan2(y, x)
            guards.append(window('joint_1', bearing - BASE_TOL, bearing + BASE_TOL))
        return guards

    def _run(self, once, pose):
        """One attempt, retried once on CONTROL_FAILED: the arm can get nudged after the previous move
        (letting go of an object, settling), so the plan's start no longer matches the joint state."""
        code = once(pose)
        if code == CONTROL_FAILED:
            self.log.warn('Execution failed (CONTROL_FAILED); retrying once after settling')
            time.sleep(SETTLE)
            code = once(pose)
        if code == SUCCESS:
            time.sleep(SETTLE)
            return True
        if code is not None:
            hint = ERROR_HINTS.get(code, '')
            if code in (-1, -31, 99999) and self.joint_guards and once == self._rrtconnect_once:
                hint += '; joint guards are on (ArmMover(joint_guards=False) plans without them)'
            self.log.error(f'Motion failed, MoveIt error code {code}{hint}')
        return False

    def _send(self, client, goal):
        handle = wait_for(client.send_goal_async(goal), timeout=10.0)
        if handle is None or not handle.accepted:
            self.log.error('MoveIt rejected the goal')
            return None
        response = wait_for(handle.get_result_async(), timeout=120.0)
        if response is None:
            self.log.error('MoveIt gave no result within 120 s')
            return None
        return response.result.error_code.val
