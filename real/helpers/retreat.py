"""Get the arm home safely from wherever a run stopped (a failed step, Ctrl-C, a restart).

A plain "move home" from a hand that is down in a pocket, or round the jar in the stand, can
drag the fingers through the rack / stand / jar. So first:

  1. open the gripper if it is not holding the jar (fingers off whatever they closed on);
  2. back out: holding the jar -> straight up (out of the pocket it may be in); empty -> straight
     back along the approach (out from round the jar: up after a top grasp, back after a side one);
  3. straight up to a safe height;
  4. then home with a planned move (Pilz PTP, else RRTConnect, round the planning scene).

Steps 2 and 3 are straight lines (MoveIt checks them against the scene); one that is not possible
is skipped and the next one tried. Nothing in here needs the task: go_home.py (bringup) and
run_task.py (after a failed step, `spice home`) both use it.
"""
import numpy as np

try:
    from helpers.geometry import quat_to_matrix
except ImportError:                                    # run from helpers/ (go_home.py)
    from geometry import quat_to_matrix

BACK_OFF = 0.08         # m along the approach (or up, holding the jar)
SAFE_Z = 0.25           # m in base_link: high enough to clear the rack, the stand and a held jar
NEAR_HOME = 0.03        # m: already home, nothing to do


def finger_says_holding(finger, cfg):
    """Best guess from the finger joint alone (no task state): closed on something. Open is ~0.35,
    closed on nothing >= holding_max, on the jar ~0.5-0.6."""
    g = cfg['gripper']
    return finger is not None and (g['open'] + 0.1) <= finger < g['holding_max']


def retreat_and_home(mover, cfg, ee_pose, log=print, held=None, finger=None, open_gripper=None,
                     back_off=BACK_OFF, safe_z=SAFE_Z):
    """Back out, go up, go home. ee_pose(): current [x, y, z, qx, qy, qz, qw] of end_effector_link.
    held: True / False from the task state, None = guess from `finger` (finger joint position).
    open_gripper(): opens it (None: leave the gripper as it is). True if home was reached."""
    home = [float(v) for v in cfg['home']]
    pose = list(ee_pose())
    if np.linalg.norm(np.subtract(pose[:3], home[:3])) < NEAR_HOME:
        log('already at home')
        return True
    if held is None:
        held = finger_says_holding(finger, cfg)
    log(f'going home safely from {np.round(pose[:3], 3).tolist()} '
        f'({"holding the jar: up first" if held else "empty: open, back out along the approach"})')
    if not held and open_gripper is not None:
        open_gripper()
    R = quat_to_matrix(*pose[3:])
    p = np.asarray(pose[:3], float)
    back = p + [0.0, 0.0, back_off] if held else p - back_off * R[:, 2]
    if mover.move_cartesian(list(back) + pose[3:], quiet=True):
        log('backed out')
        p = back
    else:
        log('could not back out in a straight line; trying straight up')
    if p[2] < safe_z:
        up = [p[0], p[1], safe_z] + pose[3:]
        if mover.move_cartesian(up, quiet=True):
            log(f'up to z {safe_z:.2f}')
        else:
            log('could not go straight up; planning home from here')
    for attempt in range(1, 4):
        if mover.move_free(home):
            log(f'home ({getattr(mover, "last_planner", None) or "planned"})')
            return True
        log(f'home: no plan (attempt {attempt})')
    log('could not get home: jog it (spice jog) or use the Kinova controller')
    return False
