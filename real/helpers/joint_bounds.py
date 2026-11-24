"""Bring the arm back inside MoveIt's joint limits before planning.

MoveIt refuses to plan from a start state outside the URDF limits (error -26, START_STATE_INVALID:
"Joint 'joint_3' from the starting state is outside bounds"). The Kinova firmware allows a bit more
than the URDF -- its own Retract position has joint_3 at ~161 deg, the URDF stops at 149.5 deg --
so after the controller's home/retract button or a power cycle, every MoveIt move fails.

unwind() turns each joint that is a little out of bounds (at most `max_fix`) back to just inside,
slowly, with one trajectory straight to the joint trajectory controller (no MoveIt), keeping the
other joints where they are. Further out than that it does nothing and says how to fix it by hand.
"""
import math
import time
import xml.etree.ElementTree as ET

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_JOINTS = [f'joint_{i}' for i in range(1, 7)]
ACTION = '/joint_trajectory_controller/follow_joint_trajectory'


def urdf_limits(urdf_xml, names=ARM_JOINTS):
    """{joint: (lower, upper)} of the revolute joints in a URDF string."""
    out = {}
    for j in ET.fromstring(urdf_xml).iter('joint'):
        lim = j.find('limit')
        if j.get('name') in names and j.get('type') == 'revolute' and lim is not None:
            out[j.get('name')] = (float(lim.get('lower')), float(lim.get('upper')))
    return out


def out_of_bounds(positions, limits, margin=0.05):
    """{joint: (now, target inside the limits)} for joints outside their limits."""
    fix = {}
    for name, (lo, hi) in limits.items():
        q = positions.get(name)
        if q is None:
            continue
        if q > hi:
            fix[name] = (q, hi - margin)
        elif q < lo:
            fix[name] = (q, lo + margin)
    return fix


class JointWatcher:
    """Latest /joint_states and the URDF limits (from /robot_description) for a node."""

    def __init__(self, node, callback_group=None):
        self.positions, self.limits = {}, {}
        node.create_subscription(JointState, '/joint_states', self._on_joints, 10,
                                 callback_group=callback_group)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(String, '/robot_description', self._on_urdf, latched,
                                 callback_group=callback_group)

    def _on_joints(self, msg):
        self.positions.update(zip(msg.name, msg.position))

    def _on_urdf(self, msg):
        try:
            self.limits = urdf_limits(msg.data)
        except ET.ParseError:
            pass

    def wait(self, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.limits and all(j in self.positions for j in ARM_JOINTS):
                return True
            time.sleep(0.05)
        return False


def unwind(node, watcher, speed_rad_s=0.15, margin=0.05, max_fix=0.6, log=print, confirm=None):
    """True if every arm joint is (now) inside its limits. Moves only joints that are out.
    confirm: called with a question before moving; False = do not move."""
    if not watcher.wait():
        log('joint limits / joint states not available; not checking the start pose')
        return True
    fix = out_of_bounds(watcher.positions, watcher.limits, margin)
    if not fix:
        return True
    lines = [f'{n} at {math.degrees(q):+.1f} deg, MoveIt allows '
             f'{math.degrees(watcher.limits[n][0]):+.1f}..{math.degrees(watcher.limits[n][1]):+.1f}'
             for n, (q, _) in fix.items()]
    too_far = [n for n, (q, t) in fix.items() if abs(q - t) > max_fix]
    if too_far:
        log('Out of MoveIt\'s joint limits, too far to fix by itself: ' + '; '.join(lines) + '.\n'
            '  Move it back by hand: Kinova Web App (http://<robot_ip>) -> Arm control -> Joint, '
            'or the controller, then run again.')
        return False
    log('Out of MoveIt\'s joint limits (Kinova Retract pose?): ' + '; '.join(lines) + '.\n'
        '  Turning ' + ', '.join(f'{n} to {math.degrees(t):+.1f} deg' for n, (q, t) in fix.items())
        + ' first (slowly, other joints stay) ...')
    if confirm is not None and not confirm('unwind those joints now?'):
        return False
    client = ActionClient(node, FollowJointTrajectory, ACTION)
    if not client.wait_for_server(timeout_sec=10.0):
        log(f'{ACTION} not available; cannot unwind')
        return False
    now = [watcher.positions[j] for j in ARM_JOINTS]
    goal_q = [fix[j][1] if j in fix else q for j, q in zip(ARM_JOINTS, now)]
    secs = max(3.0, max(abs(g - q) for g, q in zip(goal_q, now)) / speed_rad_s)
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    goal.trajectory.points = [JointTrajectoryPoint(
        positions=goal_q, velocities=[0.0] * 6,
        time_from_start=Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9)))]
    send = client.send_goal_async(goal)
    end = time.monotonic() + 10.0
    while not send.done() and time.monotonic() < end:
        time.sleep(0.02)
    handle = send.result() if send.done() else None
    if handle is None or not handle.accepted:
        log('the trajectory controller rejected the unwind move')
        return False
    res = handle.get_result_async()
    end = time.monotonic() + secs + 15.0
    while not res.done() and time.monotonic() < end:
        time.sleep(0.05)
    time.sleep(0.5)                                  # fresh joint states
    left = out_of_bounds(watcher.positions, watcher.limits, 0.0)
    if left:
        log('still out of bounds after the unwind move: ' + ', '.join(
            f'{n} {math.degrees(q):+.1f} deg' for n, (q, _) in left.items()) + ' (arm not following?)')
        return False
    log('back inside the joint limits')
    return True
