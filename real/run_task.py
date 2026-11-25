#!/usr/bin/env python3
"""Spice task on the real Kinova Gen3 Lite: find the salt jar in the rack, top grasp, set it in the
regrasp stand, side regrasp, back into the stand, top regrasp, back into its rack slot.

Needs (each in its own terminal, see README.md):
  ros2 launch real/launch/bringup_robot.launch.py     # Kinova driver + MoveIt + hand-eye TF
  ros2 launch real/launch/bringup_camera.launch.py    # RealSense D405, aligned depth

  python3 real/run_task.py --dry-run                  # perception from where the arm is, no motion
  python3 real/run_task.py --steps 1,2,8              # level 1: pick it out of the rack, put it back
  python3 real/run_task.py                            # all 8 steps, with the shake over the bowl
  python3 real/run_task.py --steps 1-4,6-8            # level 2: without the shake
  python3 real/run_task.py --steps 4 --resume         # carry on from the last run's state
Moves without asking; ANY KEY PAUSES (the arm stops where it is), then Enter = carry on, q = stop
(helpers/key_pause.py). Ctrl-C stops too. --ask: ask [y/N] before every motion instead.
Pictures and state.json go to real/captures/<time>/.
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rclpy                                                   # noqa: E402
import yaml                                                    # noqa: E402

from helpers.joint_bounds import JointWatcher, unwind          # noqa: E402
from helpers.key_pause import KeyPause                         # noqa: E402
from helpers.robot_motion import ArmMover                      # noqa: E402
from spice_task.io_node import TaskIO, spin_in_background      # noqa: E402
from spice_task.steps import StepFailed, Task                  # noqa: E402

DEFAULT_STEPS = '1-8'
CAPTURES = os.path.join(HERE, 'captures')


def parse_steps(text):
    steps = []
    for part in text.split(','):
        if '-' in part:
            a, b = part.split('-')
            steps += list(range(int(a), int(b) + 1))
        elif part.strip():
            steps.append(int(part))
    bad = [s for s in steps if not 1 <= s <= 8]
    if bad:
        raise SystemExit(f'steps must be 1-8, got {bad}')
    return steps


def ask_yes(question):
    try:
        return input(f'{question} [y/N] ').strip().lower() in ('y', 'yes')
    except EOFError:
        return False


def offer_home(task, args):
    """A step failed (or Ctrl-C): bring the arm home safely, so the next run can start from there
    instead of from wherever it stopped. Asked first (--home: without asking)."""
    if args.home:
        pass
    elif not sys.stdin.isatty():
        print('Stopped. The arm stays where it is; spice home brings it back safely.')
        return
    elif not ask_yes('Stopped. Bring the arm home safely now (open if empty, back out, up, home)?'):
        print('The arm stays where it is; spice home brings it back safely later.')
        return
    if task.go_home_safely():
        print('Home. Carry on with:  spice resume <steps>   (the saved state is kept)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(HERE, 'config', 'real.yaml'))
    p.add_argument('--steps', default=DEFAULT_STEPS, help=f'steps to run, e.g. 1,2,8 or 1-4 (default {DEFAULT_STEPS})')
    p.add_argument('--resume', nargs='?', const='latest', default=None,
                   help='start from a saved state.json (default: the newest run in real/captures)')
    p.add_argument('--spice', help='fetch the jar whose paper label reads this (default: labels.spice in '
                   'the config, salt)')
    p.add_argument('--index', type=int, help='ID mode: pick lid N (numbers in the survey picture) instead of '
                   'by its label')
    p.add_argument('--scan', action='store_true', help='no steps: go to the survey pose and list the jars on '
                   'the rack with their labels (a SCAN_RESULT {json} line; the recipe window uses it)')
    p.add_argument('--pour-test', action='store_true', help='step 5: instead of tip + shake, tip slowly until a '
                   'key is pressed when it pours; the angle goes to real/data/pour_tests.csv (spice pourtest)')
    p.add_argument('--ask', action='store_true', help='ask [y/N] before every motion (default: no questions; '
                   'any key pauses the arm, Enter carries on, q stops)')
    p.add_argument('--auto', action='store_true', help=argparse.SUPPRESS)          # the default now
    p.add_argument('--dry-run', action='store_true', help='no motion and no gripper: perception + planned poses')
    p.add_argument('--show', action='store_true', help='pop up each picture as it is taken')
    p.add_argument('--home', action='store_true', help='go home at the end (if the gripper is empty), and '
                   'after a failed step without asking')
    p.add_argument('--go-home', action='store_true', help='no steps: just get the arm home safely from where it is '
                   '(open if empty, back out, up, home); spice home')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    steps = parse_steps(args.steps)
    out_dir = os.path.join(CAPTURES, time.strftime('%Y%m%d_%H%M%S'))

    rclpy.init()
    io = TaskIO(cfg)
    executor, thread = spin_in_background(io)
    mover = ArmMover(io, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                     motion='cartesian', cartesian_fallback=False, planning=cfg.get('planning'))
    joints = JointWatcher(io)
    auto = not args.ask
    task = Task(io, mover, cfg, out_dir, auto=auto, dry_run=args.dry_run, show=args.show, index=args.index,
                config_path=args.config, spice=args.spice, pour_test=args.pour_test)
    pause = None
    if auto and not args.dry_run:
        pause = KeyPause(log=lambda m: print(m, flush=True))
        if pause.start():
            mover.pause = task.pause = pause
            print('AUTOMATIC: no questions before moves. Press ANY KEY to PAUSE (the arm stops where it is); '
                  'then Enter = carry on, q = stop. Ctrl-C stops too. Keep a hand on the e-stop.')
        else:
            pause = None
            print('AUTOMATIC, no keyboard to pause with (not a terminal): keep a hand on the e-stop.')
    ok = False
    try:
        if args.resume:
            path = args.resume
            if path == 'latest':
                runs = sorted(glob.glob(os.path.join(CAPTURES, '*', 'state.json')))
                if not runs:
                    raise SystemExit('--resume: no earlier run in real/captures')
                path = runs[-1]
            task.load_state(path)
        ready, why = io.wait_ready()
        if not ready:
            raise SystemExit(f'Not ready: {why}')
        if not args.dry_run and not mover.wait_until_ready(timeout=30.0):
            raise SystemExit('MoveIt is not running (bringup_robot.launch.py)')
        if not args.dry_run and not unwind(io, joints, confirm=None if auto else ask_yes):
            raise SystemExit('A joint is outside MoveIt\'s limits, so MoveIt cannot plan any move (see above).')
        if args.go_home:
            ok = task.go_home_safely()
        elif args.scan:
            try:
                task.update_scene()
                task.scan()
                ok = True
            except StepFailed as e:
                print(f'\nFAILED scan: {e}')
        else:
            print(f'Steps {steps}; pictures and state in {out_dir}' + (' [DRY RUN: no motion]' if args.dry_run else ''))
            ok = task.run(steps)
            if ok and args.home and not args.dry_run and not task.state.get('held'):
                task.free(cfg['home'], 'home')
            elif not ok and not args.dry_run:
                if pause is not None:
                    pause.close()                      # typed answers from here on
                offer_home(task, args)
    except KeyboardInterrupt:
        print('\nInterrupted.')
        if pause is not None:
            pause.close()
        if not args.dry_run and not args.go_home:
            try:
                offer_home(task, args)
            except KeyboardInterrupt:
                print('\nInterrupted again: leaving the arm where it is (spice home brings it back).')
    finally:
        if pause is not None:
            pause.close()                              # the terminal back to normal
        task.save_state()
        executor.shutdown()
        thread.join(timeout=2.0)
        io.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
