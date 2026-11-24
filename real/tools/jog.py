#!/usr/bin/env python3
"""Move the arm by hand from the terminal (MoveIt must be running: spice jog starts it).

  3 +10        joint 3 by +10 degrees (joints 1..6; at most 45 per command)
  x +2         hand 2 cm along base +x, keeping its orientation (x, y, z; at most 10 cm)
  open/close   the gripper
  home         the home pose from config/real.yaml
  unwind       turn joints that are outside MoveIt's limits (Kinova Retract pose) back inside
  pose         print the hand pose (paste it into config/real.yaml)
  save NAME    write the hand pose into config/real.yaml as NAME (survey, stand_look, home)
  q            quit

Every move is planned by MoveIt (so it does not hit itself) at the config's speed.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)

import numpy as np                                             # noqa: E402
import rclpy                                                   # noqa: E402
import yaml                                                    # noqa: E402

from helpers.joint_bounds import JointWatcher, unwind          # noqa: E402
from helpers.robot_motion import ArmMover                      # noqa: E402
from spice_task import task_poses as tp                        # noqa: E402
from spice_task.config_file import set_line                    # noqa: E402
from spice_task.io_node import TaskIO, spin_in_background      # noqa: E402

JOINTS = [f'joint_{i}' for i in range(1, 7)]
SAVABLE = ('survey', 'stand_look', 'home')


def pose_now(io):
    T = io.ee_pose()
    return tp.as_pose(T[:3, 3], T[:3, :3])


def fmt_pose(p):
    return '[' + ', '.join(f'{v:.3f}' for v in p) + ']'


def save_pose(config, name, pose):
    """Replace the `name:` line in the YAML (keeps the file's comments)."""
    return set_line(config, name, fmt_pose(pose))


def ask_yes(question):
    try:
        return input(f'{question} [y/N] ').strip().lower() in ('y', 'yes')
    except EOFError:
        return False


def main():
    config = os.path.join(ROOT, 'config', 'real.yaml')
    with open(config) as f:
        cfg = yaml.safe_load(f)
    rclpy.init()
    io = TaskIO(cfg)
    executor, thread = spin_in_background(io)
    mover = ArmMover(io, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                     planning=cfg.get('planning'))
    watcher = JointWatcher(io)
    print(__doc__)
    try:
        if not mover.wait_until_ready(timeout=30.0):
            raise SystemExit('MoveIt is not running (start it with: spice jog)')
        unwind(io, watcher, confirm=ask_yes)        # MoveIt cannot plan from outside the joint limits
        mover.set_scene(cfg)                        # table + wrist camera
        while True:
            joints = io.joints
            if all(j in joints for j in JOINTS):
                print('joints (deg): ' + '  '.join(f'{k + 1}:{math.degrees(joints[j]):+7.1f}'
                                                    for k, j in enumerate(JOINTS)))
            try:
                print('hand: ' + fmt_pose(pose_now(io)))
            except Exception as e:                               # noqa: BLE001
                print(f'hand pose not available ({e})')
            try:
                words = input('jog> ').split()
            except EOFError:
                break
            if not words:
                continue
            cmd = words[0].lower()
            try:
                if cmd in ('q', 'quit', 'exit'):
                    break
                elif cmd in '123456' and len(cmd) == 1 and len(words) == 2:
                    step = float(words[1])
                    if abs(step) > 45:
                        print('at most 45 degrees per command')
                        continue
                    name = JOINTS[int(cmd) - 1]
                    target = [joints[j] for j in JOINTS]
                    target[int(cmd) - 1] += math.radians(step)
                    print(f'{name} -> {math.degrees(target[int(cmd) - 1]):+.1f} deg ...')
                    print('done' if mover.move_to_joints(JOINTS, target) else 'not possible (see the MoveIt message)')
                elif cmd in ('x', 'y', 'z') and len(words) == 2:
                    step = float(words[1]) / 100.0
                    if abs(step) > 0.10:
                        print('at most 10 cm per command')
                        continue
                    p = pose_now(io)
                    p[('x', 'y', 'z').index(cmd)] += step
                    ok = mover.move_cartesian(p) or mover.move_free(p)
                    print('done' if ok else 'not possible (see the MoveIt message)')
                elif cmd in ('open', 'close'):
                    io.set_gripper(cfg['gripper']['open' if cmd == 'open' else 'closed'])
                elif cmd == 'unwind':
                    unwind(io, watcher, confirm=ask_yes)
                elif cmd == 'home':
                    unwind(io, watcher, confirm=ask_yes)
                    ok = mover.move_cartesian(cfg['home']) or mover.move_free(cfg['home'])
                    print('done' if ok else 'not possible')
                elif cmd == 'pose':
                    print(f'paste into config/real.yaml:  survey: {fmt_pose(pose_now(io))}')
                elif cmd == 'save' and len(words) == 2 and words[1] in SAVABLE:
                    print('written to config/real.yaml:  ' + save_pose(config, words[1], pose_now(io)))
                else:
                    print('? ' + ' | '.join(['3 +10', 'x +2', 'open', 'close', 'home', 'pose',
                                             'save survey', 'q']))
            except (ValueError, KeyError) as e:
                print(f'? {e}')
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        io.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    main()
