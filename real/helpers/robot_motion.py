# A pose is [x, y, z, qx, qy, qz, qw]: position (m) + orientatin in the base frame.
# motion = 'cartesian'  
# motion = 'rrtconnect' 
import functools
import math
import time

from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, OrientationConstraint, PositionConstraint
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive

ERROR_HINTS = {-31: ' (no IK solution: pose unreachable)', -10: ' (start state in collision)',
               -26: ' (start state invalid: a joint outside its limits)', -1: ' (planning failed)',
               -4: ' (execution failed)', 99999: ' (no plan found)'}
ARM_JOINTS = [f'joint_{i}' for i in range(1, 7)]
# Free moves keep the arm in its normal shape (as the sim's arm client did): the base turned
# towards the target, the elbow bent the usual way. Without it IK / RRTConnect may pick any of the
# arm's solutions for a hand pose -- base turned away, elbow dropped -- and swing it through the table.
DEFAULT_GUARD = {'base_joint': 'joint_1', 'base_tolerance': 1.0, 'elbow_joint': 'joint_3', 'elbow_range': [0.0, 2.6]}


def parse_pose(entry, what='pose'):
    if isinstance(entry, dict) and 'pose' in entry:
        entry = entry['pose']
    if isinstance(entry, (list, tuple)) and len(entry) == 7:
        return [float(v) for v in entry]
    raise ValueError(f'{what}: expected a pose [x, y, z, qx, qy, qz, qw] (7 numbers), got {entry!r}')


def parse_joints(entry, n, what='joints'):
    if isinstance(entry, (list, tuple)) and len(entry) == n:
        return [float(v) for v in entry]
    raise ValueError(f'{what}: expected {n} joint angles (rad), got {entry!r}')


class RetryMove(Exception):
    """A move was paused from the keyboard (helpers/key_pause.py) and the user carried on."""


def again_after_pause(move):
    """Plan the move again from where the arm stopped, if it was paused part way."""
    @functools.wraps(move)
    def wrapper(self, *args, **kwargs):
        while True:
            try:
                return move(self, *args, **kwargs)
            except RetryMove:
                self.log.info(f'{move.__name__}: planning it again from where the arm stopped')
    return wrapper


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
                 motion='cartesian', cartesian_fallback=False, max_step=0.005, max_jump=0.5, callback_group=None,
                 planning=None):
        """planning: config/real.yaml `planning:` (guard: joint limits for free moves, null = none)."""
        self.node, self.group, self.ee_link, self.base_frame, self.speed = node, group, ee_link, base_frame, speed
        planning = planning or {}
        self.guard = planning.get('guard', DEFAULT_GUARD)
        self.use_ptp = planning.get('ptp', True)
        self.motion, self.cartesian_fallback = motion, cartesian_fallback
        self.max_step = max_step          # m between interpolated points on a straight-line path
        self.max_jump = max_jump       
        self.move_client = ActionClient(node, MoveGroup, 'move_action', callback_group=callback_group)
        self.exec_client = ActionClient(node, ExecuteTrajectory, 'execute_trajectory', callback_group=callback_group)
        self.cart_client = node.create_client(GetCartesianPath, 'compute_cartesian_path',
                                              callback_group=callback_group)
        self.ik_client = node.create_client(GetPositionIK, 'compute_ik', callback_group=callback_group)
        self.log = node.get_logger()
        self.last_fraction = None
        self.last_planner = None
        self.pause = None                  # helpers/key_pause.KeyPause: any key stops the move

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
            self.log.warn('Straight line not possible; falling back to a planned move')
        return self.move_free(pose)

    # ------------------------------------------------------------ free moves
    def guard_constraints(self, pose):
        """JointConstraints that keep a free move's goal in the arm's normal shape (self.guard)."""
        g = self.guard
        if not g:
            return []
        out = []
        if g.get('base_joint') and math.hypot(pose[0], pose[1]) > 0.05:
            out.append(JointConstraint(joint_name=g['base_joint'], position=math.atan2(pose[1], pose[0]),
                                       tolerance_above=float(g['base_tolerance']),
                                       tolerance_below=float(g['base_tolerance']), weight=1.0))
        if g.get('elbow_joint'):
            lo, hi = (float(v) for v in g['elbow_range'])
            out.append(JointConstraint(joint_name=g['elbow_joint'], position=(lo + hi) / 2,
                                       tolerance_above=(hi - lo) / 2, tolerance_below=(hi - lo) / 2, weight=1.0))
        return out

    def ik(self, pose, guard=True, timeout=0.5):
        """Arm joint angles {name: rad} that put end_effector_link at `pose`, collision-free (MoveIt's
        planning scene: table, camera, ...) and in the guard's shape, seeded from the current joints
        (so the nearest solution); None if there is none. Raises RuntimeError without the IK service."""
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError('compute_ik service not available')
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = self.group, self.ee_link, True
        r.robot_state.is_diff = True
        r.pose_stamped.header.frame_id = self.base_frame
        r.pose_stamped.pose = list_to_pose(pose)
        r.timeout = Duration(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        if guard:
            r.constraints = Constraints(joint_constraints=self.guard_constraints(pose))
        res = wait_for(self.ik_client.call_async(req), timeout=5.0 + timeout)
        if res is None or res.error_code.val != 1:
            return None
        js = res.solution.joint_state
        return {n: q for n, q in zip(js.name, js.position) if n in ARM_JOINTS}

    @again_after_pause
    def move_ptp(self, pose):
        """Pilz PTP: IK for the pose (guarded, collision-free, nearest the current joints), then every
        joint in a straight line to it at once -- short, the same every time, no random detours.
        False if there is no such IK solution or Pilz is not loaded / finds the path in collision."""
        try:
            q = self.ik(pose)
        except RuntimeError as e:
            self.log.info(f'PTP: {e}')
            return False
        if q is None or any(j not in q for j in ARM_JOINTS):
            self.log.info('PTP: no collision-free IK solution in the arm\'s normal shape')
            return False
        return self.move_to_joints(ARM_JOINTS, [q[j] for j in ARM_JOINTS], pipeline='pilz_industrial_motion_planner',
                                   planner='PTP')

    def move_free(self, pose):
        """A planned move to `pose`: Pilz PTP if it can, else OMPL RRTConnect (both kept off the table
        etc. by the planning scene, and in the arm's normal shape by the guard)."""
        if self.use_ptp and self.move_ptp(pose):
            self.last_planner = 'PTP'
            return True
        self.last_planner = 'RRTConnect'
        return self.move_rrtconnect(pose)

    def set_scene(self, cfg, table_top_z=None, rack=None, stand=None, bowl=None):
        """Table, wrist camera (+ rack, stand) into MoveIt's planning scene (helpers/scene.py)."""
        try:
            from helpers.scene import apply_scene
        except ImportError:                                    # run from helpers/ (go_home.py)
            from scene import apply_scene
        return apply_scene(self.node, cfg, table_top_z, log=self.log.info, rack=rack, stand=stand, bowl=bowl)

    @again_after_pause
    def move_cartesian(self, pose, quiet=False):
        """quiet: the caller falls back to a planned move, so a partial line is not an error."""
        if not self.wait_until_ready(timeout=10.0):
            return False
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
            return False
        self.last_fraction = res.fraction
        if res.fraction < 0.999:
            if not quiet:
                self.log.error(f'Straight line only {res.fraction:.0%} feasible (joint limit, singularity or '
                               'collision on the way); not moving')
            return False
        jump = max_joint_jump(res.solution)
        if jump > self.max_jump:
            self.log.error(f'Straight line needs a sudden joint jump of {jump:.2f} rad (wrist/elbow flip); '
                           'not moving')
            return False
        return self._check(self._send(self.exec_client, ExecuteTrajectory.Goal(trajectory=res.solution)))

    def reachable(self, pose):
        """Can the arm put end_effector_link at `pose` (joint limits, no collision)? MoveIt's IK,
        no motion. True if it cannot tell (no IK service), so callers just try the move."""
        if not self.ik_client.wait_for_service(timeout_sec=2.0):
            return True
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = self.group, self.ee_link, True
        r.robot_state.is_diff = True
        r.pose_stamped.header.frame_id = self.base_frame
        r.pose_stamped.pose = list_to_pose(pose)
        r.timeout = Duration(sec=0, nanosec=500_000_000)
        r.constraints = Constraints(joint_constraints=self.guard_constraints(pose))
        res = wait_for(self.ik_client.call_async(req), timeout=5.0)
        return res is None or res.error_code.val == 1

    @again_after_pause
    def move_rrtconnect(self, pose):
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
            return False
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name, r.pipeline_id, r.planner_id = self.group, 'ompl', 'RRTConnectkConfigDefault'
        r.allowed_planning_time, r.num_planning_attempts = 5.0, 5
        r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = self.speed
        r.start_state.is_diff = True
        r.goal_constraints.append(Constraints(position_constraints=[position],
                                              orientation_constraints=[orientation],
                                              joint_constraints=self.guard_constraints(pose)))
        goal.planning_options.plan_only = False
        return self._check(self._send(self.move_client, goal))

    @again_after_pause
    def move_to_joints(self, joint_names, positions, tolerance=0.001, pipeline='ompl',
                       planner='RRTConnectkConfigDefault'):
        # Plan in joint space (OMPL) to the given joint angles (rad) and execute.
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.log.error('move_action not available: is the MoveIt launch running?')
            return False
        constraints = Constraints()
        for name, position in zip(joint_names, positions):
            jc = JointConstraint(joint_name=name, position=float(position), weight=1.0)
            jc.tolerance_above = jc.tolerance_below = tolerance
            constraints.joint_constraints.append(jc)
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name, r.pipeline_id, r.planner_id = self.group, pipeline, planner
        r.allowed_planning_time, r.num_planning_attempts = 5.0, 5
        r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = self.speed
        r.start_state.is_diff = True
        r.goal_constraints.append(constraints)
        goal.planning_options.plan_only = False
        return self._check(self._send(self.move_client, goal))

    def _send(self, client, goal, timeout=120.0):
        """Send a motion goal and wait for it. A key press (self.pause) or Ctrl-C while it moves
        cancels it, so the arm stops where it is; after a pause the move is planned again
        (RetryMove -> again_after_pause), Ctrl-C / q goes up as KeyboardInterrupt."""
        if self.pause is not None:
            self.pause.hold()                       # paused between moves: do not start this one
        handle = wait_for(client.send_goal_async(goal), timeout=10.0)
        if handle is None or not handle.accepted:
            self.log.error('MoveIt rejected the goal')
            return None
        done = handle.get_result_async()
        end = time.time() + timeout
        paused = False
        try:
            while not done.done():
                if self.pause is not None and self.pause.requested():
                    paused = True
                    break
                if time.time() > end:
                    return None
                time.sleep(0.01)
        except KeyboardInterrupt:
            self.stop_move(handle, done)
            raise
        if paused:
            self.stop_move(handle, done)
            self.pause.hold()
            raise RetryMove()
        response = done.result()
        return None if response is None else response.result

    def stop_move(self, handle, done):
        """Cancel the goal (the controller stops the arm) and wait until it has."""
        self.log.warning('stopping the arm: cancelling the move')
        wait_for(handle.cancel_goal_async(), timeout=5.0)
        wait_for(done, timeout=10.0)

    def _check(self, result):
        code = None if result is None else result.error_code.val
        if code != 1:
            self.log.error(f'Motion failed, MoveIt error code {code}{ERROR_HINTS.get(code, "")}')
        return code == 1
