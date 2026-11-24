"""The spice task on the real arm: the tran branch's 8 steps (rbe595_manipulation/steps/), with

  * the salt jar found by its round lid (Hough circle, lid_detector.py) instead of a cap
    AprilTag: the jar has "salt" written on it, no tag;
  * the rack (tags 6, 7) and the regrasp stand (tag 8) still found by their AprilTags; the stand
    is tag only (the jar's place in it comes from the tag and the stand's shape, no Hough);
  * "is the jar in the pocket?" answered by comparing a picture of the pocket with reference
    pictures taken earlier (reference.py): stand empty (step 1), stand with the jar (step 3),
    rack slot with the jar (step 1), rack slot empty (step 2);
  * HW7's ArmMover (MoveIt: straight lines via compute_cartesian_path; free moves via Pilz PTP,
    else RRTConnect, with the sim's base/elbow joint limits; table, camera, rack and stand in the
    planning scene) and the GripperCommand action instead of the sim's arm/gripper clients.

Step                What happens                                                     Found by
1 find_jar          from the survey pose: rack tags, stand tag, lids. Camera over    tags, Hough lid
                    the stand: tag 8 close up + "stand empty" picture. Aim the hand
                    over the lid (look-move-look), "slot with jar" picture
2 grasp_top         straight down, close, check the fingers, straight up; camera    lid; picture compare
                    over the slot: it must now be empty ("slot empty" picture)
3 place_on_stand    carry over the stand's pocket, lower, let go, back up; camera   tag 8; picture compare
                    over the stand: the jar must be there ("stand with jar" picture)
4 regrasp_side      slide back top-down, turn side-on, come in level, close, lift   tag 8
5 shake_over_bowl   up, carry the side-held jar upright over the bowl (step 1), tip it   bowl (step 1)
                    over with joint_6, shake, upright again
6 return_to_stand   lower into the pocket, let go, back out, turn top-down; jar     tag 8; picture compare
                    there?
7 regrasp_top       top grasp in the stand                                           tag 8
8 return_to_rack    back over its slot, lower, let go, back up; jar there?          lid (step 1); picture compare
"""
import contextlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from helpers.geometry import quat_to_matrix
from helpers.retreat import retreat_and_home
from spice_task import reference as ref
from spice_task.bowl import detect_bowl
from spice_task import task_poses as tp
from spice_task.config_file import set_line
from spice_task.lid_detector import LidDetector, draw_circle
from spice_task.tags import draw_tags, measure_tags

ARM_JOINTS = [f'joint_{i}' for i in range(1, 7)]
JOINT6_LIMIT = 2.6                     # rad, URDF (+-149 deg)
STEP_NAMES = {1: 'find_jar', 2: 'grasp_top', 3: 'place_on_stand', 4: 'regrasp_side',
              5: 'shake_over_bowl', 6: 'return_to_stand', 7: 'regrasp_top', 8: 'return_to_rack'}
YELLOW, CYAN, GREEN, RED, MAGENTA = (0, 255, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)
# RViz marker colours (r, g, b, a)
M_LID, M_TARGET, M_STAND, M_RACK, M_TAG, M_GOAL, M_TEXT = (
    (1, 1, 0, 0.9), (1, 0.5, 0, 0.5), (0, 1, 1, 0.6), (0.2, 0.4, 1, 0.25), (0, 1, 0, 0.9), (0, 1, 0, 1),
    (1, 1, 1, 1))


class StepFailed(RuntimeError):
    pass


def fmt(v):
    return '(' + ', '.join(f'{x:+.4f}' for x in v) + ')'


class Task:
    def __init__(self, io, mover, cfg, out_dir, auto=False, dry_run=False, show=False, index=None,
                 config_path=None):
        self.io, self.mover, self.cfg, self.config_path = io, mover, cfg, config_path
        self.out_dir, self.auto, self.dry, self.show, self.index = out_dir, auto, dry_run, show, index
        self.pause = None                  # helpers/key_pause.KeyPause (run_task.py): any key pauses
        self.detector = LidDetector(cfg['lid_detection'], cfg['jar']['lid_diameter'])
        self.state = {}
        self.picture_no = 0
        os.makedirs(out_dir, exist_ok=True)

    # ================================================================== small helpers
    def log(self, msg):
        print(msg, flush=True)

    def ask(self, question):
        if self.auto or self.dry:
            if self.pause is not None:
                self.pause.hold()                   # a key was pressed: wait here before the next thing
            self.log(f'{question} [{"dry run" if self.dry else "auto"}: yes]')
            return True
        return input(f'{question} [y/N] ').strip().lower() == 'y'

    def ask_person(self, question):
        """A question that is never answered automatically (letting go of the jar). No terminal: no."""
        if self.dry:
            return False
        if not sys.stdin.isatty():
            self.log(f'{question} [no terminal: no]')
            return False
        with (self.pause.suspended() if self.pause is not None else contextlib.nullcontext()):
            try:
                return input(f'{question} [y/N] ').strip().lower() in ('y', 'yes')
            except EOFError:
                return False

    def save_state(self):
        def plain(v):
            return v.tolist() if isinstance(v, np.ndarray) else v
        with open(os.path.join(self.out_dir, 'state.json'), 'w') as f:
            json.dump({k: plain(v) for k, v in self.state.items()}, f, indent=2)

    def load_state(self, path):
        with open(path) as f:
            self.state = json.load(f)
        self.log(f'Loaded state from {path}: {sorted(self.state)}')

    def need(self, *keys):
        missing = [k for k in keys if self.state.get(k) is None]
        if missing:
            raise StepFailed(f'needs {", ".join(missing)} from an earlier step (run it first, or --resume)')

    def R_current(self):
        return self.io.ee_pose()[:3, :3]

    def save_picture(self, label, picture):
        self.picture_no += 1
        path = os.path.join(self.out_dir, f'{self.picture_no:02d}_{label}.png')
        cv2.imwrite(path, picture)
        if hasattr(self.io, 'publish_picture'):
            self.io.publish_picture(picture)
        if self.show:
            cv2.imshow('spice task', picture)
            cv2.waitKey(500)
        return path

    def viz(self, group, shapes):
        """3D markers in RViz (/spice/markers), if the IO has them (not in the offline test)."""
        if hasattr(self.io, 'show_markers'):
            self.io.show_markers(group, shapes)

    def viz_goal(self, pose, what):
        """Green arrow: where the gripper goes next (end_effector_link -> finger pads)."""
        R = quat_to_matrix(*pose[3:])
        start = np.array(pose[:3])
        end = start + self.cfg['gripper']['reach'] * R[:, 2]
        self.viz('next move', [{'type': 'arrow', 'start': start, 'end': end, 'rgba': M_GOAL},
                               {'type': 'text', 'pos': start + [0, 0, 0.03], 'text': what, 'rgba': M_GOAL}])

    def viz_jar(self, top, label='salt'):
        h = self.cfg['jar']['height']
        r = self.cfg['jar']['lid_diameter'] / 2
        self.viz('salt jar', [{'type': 'cylinder', 'center': np.asarray(top) - [0, 0, h / 2], 'radius': r,
                               'height': h, 'rgba': M_TARGET},
                              {'type': 'text', 'pos': np.asarray(top) + [0, 0, 0.03], 'text': label, 'rgba': M_TEXT}])

    def viz_stand(self, seat):
        d = self.cfg['rack']['pocket_depth']
        self.viz('stand', [{'type': 'cylinder', 'center': np.asarray(seat) + [0, 0, d / 2],
                            'radius': self.cfg['stand']['pocket_mouth'] / 2, 'height': d, 'rgba': M_STAND},
                           {'type': 'text', 'pos': np.asarray(seat) + [0, 0, d + 0.03], 'text': 'stand pocket',
                            'rgba': M_STAND}])

    # ================================================================== motion
    def line(self, pose, what, slow=False):
        """Straight-line move (Cartesian), asked first. slow: at speed_contact (going onto / into /
        out of something with the jar or round it); else at speed."""
        self.viz_goal(pose, what)
        if not self.ask(f'{what}: straight line to {fmt(pose[:3])}?'):
            raise StepFailed(f'not confirmed: {what}')
        if self.dry:
            return
        fast = getattr(self.mover, 'speed', None)
        if slow and fast is not None:
            self.mover.speed = min(fast, float(self.cfg.get('speed_contact', fast)))
        try:
            ok = self.mover.move_cartesian(pose)
        finally:
            if fast is not None:
                self.mover.speed = fast
        if not ok:
            raise StepFailed(f'{what}: straight line not possible (see the MoveIt message above)')
        self.arrived(pose, what)

    def free(self, pose, what, holding=False):
        """Move anywhere: a straight line if MoveIt can follow one, else a planned (RRTConnect) move."""
        self.viz_goal(pose, what)
        if not self.ask(f'{what}: move to {fmt(pose[:3])}?'):
            raise StepFailed(f'not confirmed: {what}')
        if self.dry:
            return
        if self.mover.move_cartesian(pose, quiet=True):
            self.arrived(pose, what)
            return
        frac = getattr(self.mover, 'last_fraction', None)
        why = f' (only {frac:.0%} of it possible)' if frac is not None else ''
        if holding and not self.ask(f'{what}: no straight line{why}; plan a free move with the jar in the hand?'):
            raise StepFailed(f'not confirmed: free move for {what}')
        self.log(f'{what}: no straight line{why}; planning a joint move instead (Pilz PTP, else RRTConnect) '
                 '-- normal for big turns')
        if not self.mover.move_free(pose):
            raise StepFailed(f'{what}: no collision-free plan (see the MoveIt message above)')
        if getattr(self.mover, 'last_planner', None):
            self.log(f'{what}: moved with {self.mover.last_planner}')
        self.arrived(pose, what)

    def arrived(self, pose, what, tol_m=0.01, tol_deg=5.0, wait=3.0):
        """MoveIt said 'done': is the hand really there (TF from the joint states)? The controller
        reports success even when the arm ignores it (e.g. a second driver took it out of
        low-level servoing), so check before looking or grasping at the wrong place."""
        goal = np.eye(4)
        goal[:3, :3], goal[:3, 3] = quat_to_matrix(*pose[3:]), pose[:3]
        end = time.monotonic() + wait
        while True:
            T = self.io.ee_pose()
            off = float(np.linalg.norm(T[:3, 3] - goal[:3, 3]))
            ang = math.degrees(math.acos(np.clip((np.trace(goal[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)))
            if off <= tol_m and ang <= tol_deg:
                return
            if time.monotonic() > end:
                raise StepFailed(
                    f'{what}: MoveIt says done, but the hand is {1000 * off:.0f} mm / {ang:.0f} deg from the '
                    'goal: the arm did not follow. Driver not in low-level servoing? (robot.log: '
                    'WRONG_SERVOING_MODE; happens when a second bringup was started). Stop every spice '
                    'terminal, then start one again.')
            time.sleep(0.1)

    def gripper(self, position, what):
        g = self.cfg['gripper']
        if self.pause is not None:
            self.pause.hold()
        if self.dry:
            self.log(f'{what}: gripper -> {position:.2f} [dry run]')
            return
        ok = self.io.set_gripper(position)
        if position < g['moved_min'] and not ok:          # opening must work; closing may stall
            raise StepFailed(f'{what}: the gripper did not open')

    def close_on_jar(self, what):
        g = self.cfg['gripper']
        if not self.ask(f'{what}: close the gripper?'):
            raise StepFailed(f'not confirmed: close for {what}')
        self.io.last_grip = None
        self.gripper(g['closed'], what)
        time.sleep(0.5)
        r = None if self.dry else getattr(self.io, 'last_grip', None)
        if r is not None:
            # the gripper's own answer: stopped by the jar (stalled) or got all the way (nothing there)
            if r['reached'] and not r['stalled']:
                raise StepFailed(f'{what}: the fingers closed all the way ({r["position"]:.2f}, nothing stopped '
                                 'them): the jar is not between them')
            self.log(f'{what}: the fingers stopped on the jar at {r["position"]:.3f}')
        self.still_holding(what, after_close=True)

    def still_holding(self, doing, after_close=False):
        """The fingers stop on the jar; if it slipped out they close on nothing."""
        if self.dry:
            return
        g, p = self.cfg['gripper'], self.io.finger
        if p is None:
            self.log(f'WARNING: no {g["finger_joint"]} on joint_states; cannot check the grasp ({doing})')
            return
        if p < g['moved_min'] and after_close:
            raise StepFailed(f'{doing}: the fingers did not move ({p:.2f}); gripper not responding?')
        if p >= g['holding_max']:
            raise StepFailed(f'{doing}: the fingers closed on nothing ({p:.2f} >= {g["holding_max"]})')
        if p < g['open'] + 0.05:
            raise StepFailed(f'{doing}: the gripper is open ({p:.2f}): the jar is not in the hand. Start from '
                             'the step that picks it up (2: from the rack, 4: side grasp in the stand, '
                             '7: top grasp in the stand)')
        self.log(f'fingers at {p:.3f}: holding the jar')

    # ================================================================== perception
    def look_tags(self, ids, label):
        """Measure the given tags from where the arm is (still). {id: TagPose}."""
        t = self.cfg['tags']
        sizes = {int(k): float(v) for k, v in t['sizes'].items() if int(k) in ids}
        frames = self.io.frames(t['measure_frames'])
        if not frames:
            raise StepFailed('no camera frames')
        tags = measure_tags(frames, sizes, self.io.K, self.io.D, use_depth=t['use_depth'], flat=True,
                            same_plane=[t['rack_ids']])
        for tag_id, tag in tags.items():
            self.log(f'tag {tag_id}: at {fmt(tag.position)}, edge {tag.edge_mm:.1f} mm, '
                     f'{tag.frames} frames, spread {tag.spread_mm:.1f} mm')
        for tag_id in ids:
            if tag_id not in tags:
                self.log(f'tag {tag_id}: not seen')
        shapes = []
        for tag_id, tag in tags.items():
            size = sizes[tag_id]
            yaw = math.atan2(tag.T[1, 0], tag.T[0, 0])
            shapes += [{'type': 'box', 'center': tag.position, 'size': (size, size, 0.002), 'yaw': yaw,
                        'rgba': M_TAG},
                       {'type': 'text', 'pos': tag.position + [0, 0, 0.02], 'text': f'tag {tag_id}', 'rgba': M_TAG}]
        self.viz(f'tags {label}', shapes)
        image, _, T_base_cam = frames[-1]
        picture = draw_tags(image.copy(), tags, self.io.K, self.io.D, T_base_cam, sizes)
        self.save_picture(label, picture)
        return tags

    def look_lids(self, top_z, around=None, label='lid', region=None):
        """All lids the camera sees (a still snapshot), near `around` or inside `region`."""
        c = self.cfg['lid_detection']
        snap = self.io.snapshot(c['depth_frames'])
        if snap is None:
            raise StepFailed('no camera frames')
        image, depth, T_base_cam = snap
        result = self.detector.detect(depth, self.io.K, self.io.D, T_base_cam, T_base_ee=self.io.ee_pose(),
                                      top_z=top_z, around=around, region=region, image=image)
        if result.birdseye is not None:                 # the top-down view the Hough ran on
            self.picture_no += 1
            cv2.imwrite(os.path.join(self.out_dir, f'{self.picture_no:02d}_{label}_topdown.png'), result.birdseye)
        picture = image.copy()
        result.lids.sort(key=lambda lid: (round(lid.pixel[1] / 60.0), lid.pixel[0]))   # reading order
        for k, lid in enumerate(result.lids):
            draw_circle(picture, lid.center, lid.radius, T_base_cam, self.io.K, self.io.D, YELLOW,
                        f'lid {k}  z {lid.center[2]:+.3f}')
        if around is not None:
            draw_circle(picture, [around[0], around[1], result.top_z if np.isfinite(result.top_z) else 0.0],
                        0.004, T_base_cam, self.io.K, self.io.D, MAGENTA, 'expected')
        self.save_picture(label, picture)
        self.viz('lids seen', [sh for k, lid in enumerate(result.lids) for sh in (
            {'type': 'cylinder', 'center': lid.center, 'radius': lid.radius, 'height': 0.004, 'rgba': M_LID},
            {'type': 'text', 'pos': lid.center + [0, 0.04, 0.01], 'text': f'lid {k}', 'rgba': M_LID})])
        z = f'{result.top_z:+.4f}' if np.isfinite(result.top_z) else 'n/a'
        self.log(f'lids at height {z} ({result.method or "-"}): ' + (', '.join(
            f'{k}: {fmt(lid.center)} r {1000 * lid.radius:.1f} mm (depth {lid.fill:.2f}, edge {lid.support:.2f})'
            for k, lid in enumerate(result.lids)) or f'none ({result.reason})'))
        return result

    def look_pose(self, top, R, height=None):
        """End-effector pose (orientation R, pointing down) that puts the camera's optical centre
        lid_detection.look_height straight above `top`. The camera is ~6 cm off the gripper axis
        and its 58 deg side is along that offset, so from the grasp approach the lid can be at the
        edge of the image; from here it is in the middle."""
        p_cam = self.io.camera_in_ee()
        h = self.cfg['lid_detection']['look_height'] if height is None else height
        pos = np.asarray(top, float) + [0.0, 0.0, h] - R @ p_cam
        return tp.as_pose(pos, R)

    def look_at_lid(self, guess, R, where, first_free=False):
        """Move the camera over `guess` (lid top), measure the lid there, return its top."""
        pose = self.look_pose(guess, R)
        if first_free:
            self.free(pose, f'camera over the lid {where}')
        else:
            self.line(pose, f'camera over the lid {where}')
        return self.measure_lid(guess, where)

    def measure_lid(self, guess, where):
        """The target lid near `guess` (lid top xyz), measured from where the arm is. In a dry run
        the arm has not moved, so the guess is returned."""
        if self.dry:
            return np.asarray(guess, float)
        result = self.look_lids(guess[2], around=guess[:2], label=f'lid_{where.replace(" ", "_")}')
        if not result.lids:
            raise StepFailed(f'cannot see the lid {where}: {result.reason}')
        lid = min(result.lids, key=lambda lid: math.dist(lid.center[:2], guess[:2]))
        self.viz_jar(lid.center)
        return lid.center

    # ------------------------------------------------------------------ reference pictures
    def rack_pocket(self, lid_top):
        """Centre of the rack slot's mouth (the plane the pictures are compared on)."""
        j, r = self.cfg['jar'], self.cfg['rack']
        return np.asarray(lid_top, float) - [0.0, 0.0, j['height'] - r['pocket_depth']]

    def stand_pocket(self, seat):
        return np.asarray(seat, float) + [0.0, 0.0, self.cfg['rack']['pocket_depth']]

    def pocket_patch(self, centre):
        snap = self.io.snapshot(1)
        if snap is None:
            raise StepFailed('no camera frames')
        image, _, T_base_cam = snap
        radius = self.cfg['jar']['lid_diameter'] / 2 + self.cfg['checks']['compare_margin']
        return ref.patch(image, T_base_cam, self.io.K, self.io.D, centre, radius)

    def take_reference(self, key, centre, patch=None):
        """Store a top-down picture of a pocket (now, or `patch`) as reference `key`."""
        if self.dry:
            return
        if patch is None:
            patch = self.pocket_patch(centre)
        path = os.path.join(self.out_dir, f'ref_{key}.png')
        cv2.imwrite(path, patch)
        self.state.setdefault('refs', {})[key] = os.path.abspath(path)
        self.log(f'reference picture "{key}" saved ({path})')

    def reference(self, key):
        path = (self.state.get('refs') or {}).get(key)
        if path and os.path.exists(path):
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return None

    def check_pocket(self, pocket, centre, expect_jar, where, lid_top=None):
        """Is the jar in (expect_jar) / out of the pocket? With lid_top (where a seated jar's lid
        is): the lid itself (Hough + depth, as for aiming) decides: seen there -> in; seen off
        the axis or low -> in, but not standing straight (stop). The reference pictures
        ('<pocket>_empty', '<pocket>_full') only when no lid is seen (or for "out of"): they change
        with the light, a stand that moved, and the label turned round by a regrasp. Fails the
        step if the jar is not where it should be. Returns the patch taken."""
        if self.dry:
            return None
        ck = self.cfg['checks']
        now = self.pocket_patch(centre)
        empty, full = self.reference(f'{pocket}_empty'), self.reference(f'{pocket}_full')
        there, msg = ref.judge(now, empty, full, ck['compare_changed'], ck['compare_grey_step'])
        pic = ref.side_by_side([empty, full, now], ['empty', 'with jar', 'now'])
        if pic is not None:
            self.save_picture(f'compare_{pocket}_{where.replace(" ", "_")}', pic)
        want = 'in' if expect_jar else 'out of'
        if expect_jar and lid_top is not None:
            lid_top = np.asarray(lid_top, float)
            res = self.look_lids(lid_top[2], around=lid_top[:2], label=f'lid_{pocket}_{where.replace(" ", "_")}')
            if res.lids:
                lid = min(res.lids, key=lambda l: math.dist(l.center[:2], lid_top[:2]))
                off, low = math.dist(lid.center[:2], lid_top[:2]), lid_top[2] - lid.center[2]
                tol = ck.get('seat_tolerance', 0.010)
                if off > 2 * tol or low > tol:
                    raise StepFailed(f'{where}: the jar is not standing straight in the {pocket} (lid '
                                     f'{1000 * off:.0f} mm off the pocket, {1000 * low:.0f} mm low): set it upright '
                                     'by hand, then spice resume the next step')
                self.log(f'{where}: jar in the {pocket} pocket: its lid is there ({1000 * off:.0f} mm off the axis, '
                         f'{1000 * low:+.0f} mm low)' + ('' if there else f'; the pictures disagree ({msg}): '
                                                         'the lid decides'))
                return now
            self.log(f'{where}: no lid seen in the {pocket} pocket ({res.reason}); going by the pictures')
        if there is None:
            self.log(f'{where}: cannot tell from the pictures ({msg})')
            if not self.ask(f'{where}: is the jar {want} the {pocket} pocket? (look at it)'):
                raise StepFailed(f'{where}: jar not {want} the {pocket} pocket (you said)')
        elif there != expect_jar:
            raise StepFailed(f'{where}: the jar is {"still in" if there else "not in"} the {pocket} pocket ({msg})')
        else:
            self.log(f'{where}: jar {want} the {pocket} pocket, as it should be ({msg})')
        return now

    def aim(self, top, poses_for, R, where, first_free=False, at_last_look=None):
        """Look-move-look: put the camera over where the lid is thought to be, measure it there,
        move over the measurement and look again, until the correction is below aim_tolerance.
        at_last_look(top) is called from the last look pose. Then go to the grasp approach above
        it. Returns the measured top."""
        top = self.aim_lid(top, R, where, first_free)
        if at_last_look is not None:
            at_last_look(top)
        self.line(poses_for(top, R, self.cfg)[0], f'right above the lid {where}')
        return top

    def aim_lid(self, top, R, where, first_free=False):
        """Look-move-look on the lid near `top` (no grasp move). Returns the measured lid top."""
        ck = self.cfg['checks']
        correction = math.inf
        for look in range(1, ck['max_aims'] + 1):
            measured = self.look_at_lid(top, R, f'aim{look} {where}', first_free=first_free and look == 1)
            correction = math.dist(measured, top)
            self.log(f'aim {look} {where}: lid top at {fmt(measured)}, {1000 * correction:.1f} mm from the aim')
            top = measured
            if correction < ck['aim_tolerance']:
                break
        else:
            self.log(f'still {1000 * correction:.1f} mm off after {ck["max_aims"]} looks; going on')
        return np.asarray(top, float)

    def jar_in_stand(self, seat, R, where):
        """The jar's lid in the stand measured again, and checked it stands straight in the pocket."""
        top = self.jar_in_pocket(tp.seated_top(seat, self.cfg), R, where, 'stand')
        self.state['stand_lid_top'] = top.tolist()
        return top

    def jar_in_pocket(self, expected, R, where, pocket):
        """Measure the lid of the jar standing in a pocket (look-move-look, Hough) and check it
        stands straight: a jar knocked over / sitting on the rim has its lid off the pocket's axis
        or low. expected: where a seated jar's lid top is. Returns the measured lid top."""
        expected = np.asarray(expected, float)
        top = self.aim_lid(expected, R, where)
        if self.dry:
            return top
        off = math.dist(top[:2], expected[:2])
        low = expected[2] - top[2]
        tol = self.cfg['checks'].get('seat_tolerance', 0.010)
        self.log(f'jar in the {pocket} ({where}): lid {1000 * off:.0f} mm off the pocket axis, '
                 f'{1000 * low:+.0f} mm below where a seated jar\'s lid is')
        if off > tol or low > tol:
            raise StepFailed(f'{where}: the jar is not standing straight in the {pocket} (lid {1000 * off:.0f} mm '
                             f'off the pocket, {1000 * low:.0f} mm low; more than {1000 * tol:.0f} mm): set it '
                             'upright in the pocket by hand, then spice resume this step')
        self.viz_jar(top)
        return top

    def stand_from_tags(self, tags):
        tag = tags.get(self.cfg['tags']['stand_id'])
        if tag is None:
            return None
        seat = tp.stand_seat(tag, self.cfg)
        self.state['stand_tag_xy'] = tag.position[:2].tolist()
        self.viz_stand(seat)
        self.log(f'regrasp stand: pocket floor at {fmt(seat)}')
        return seat

    def draw_stand(self, seat, label):
        """Picture of the predicted pocket rim, to check the tag -> pocket offset by eye."""
        snap = self.io.snapshot(1)
        if snap is None:
            return
        image, _, T_base_cam = snap
        rim = np.asarray(seat) + [0.0, 0.0, self.cfg['rack']['pocket_depth']]
        draw_circle(image, rim, self.cfg['stand']['pocket_mouth'] / 2, T_base_cam, self.io.K, self.io.D, CYAN,
                    'stand pocket (predicted)')
        self.save_picture(label, image)

    # ================================================================== camera side (mount)
    def camera(self):
        """Camera position in end_effector_link, from the hand-eye in use (+x: mount 'outside')."""
        return np.asarray(self.cfg['hand_eye']['translation'], float)

    def survey_pose(self):
        """The survey pose (else home), turned 180 deg about the approach if that puts the camera
        outside (away from the robot), when the arm can reach it that way."""
        pose = self.cfg.get('survey') or self.cfg['home']
        turned, did = tp.turn_camera_out(pose, self.camera())
        if did and (self.dry or self.mover.reachable(turned)):
            self.log('survey: gripper turned 180 deg about its axis so the camera is outside')
            return turned
        return list(pose)

    def pick_top_down(self, closing_xy, at_xy, poses_for, what):
        """Top-down rotation for closing_xy with the camera as far outside as the arm can reach
        (joint_6 turns +-149 deg only): every pose in poses_for(R) must be reachable."""
        options = tp.top_down_options(closing_xy, self.camera(), at_xy, self.R_current())
        for k, R in enumerate(options):
            if self.dry or all(self.mover.reachable(p) for p in poses_for(R)):
                if k:
                    self.log(f'{what}: the camera-outside wrist angle is out of reach; using the other')
                return R
        return options[0]

    def mirrored(self, tags):
        """True / False if the rack (else the stand) tags come out backwards / right; None if
        neither is seen. They always face the arm, so backwards = the camera is mirrored."""
        rack = tp.rack_pose(tags, self.cfg)
        if rack is not None:
            return tp.rack_backwards(rack)
        tag = tags.get(self.cfg['tags']['stand_id'])
        return None if tag is None else tp.stand_backwards(tag, self.cfg)

    def check_mount(self, tags):
        """If the tags come out backwards, the camera is on the other side of the gripper than the
        hand-eye in use says: switch the hand-eye TF to the other mount (no restart) and remember
        it in config/real.yaml. Returns True if it switched (the caller measures again)."""
        if self.dry or not self.mirrored(tags):
            return False
        mounts = self.cfg.get('hand_eye_mounts') or {}
        now = self.io.mount_name() if hasattr(self.io, 'mount_name') else None
        others = [m for m in mounts if m != now]
        if now is None or len(others) != 1:
            raise StepFailed(tp.BACKWARDS_HINT)
        other = others[0]
        self.log(f'the tags come out backwards: the camera is on the other side of the gripper ("{other}", '
                 f'not "{now}"); switching the hand-eye to "{other}"')
        if not self.io.set_mount(other):
            raise StepFailed(f'could not switch the hand-eye TF to "{other}" (restart spice; or set '
                             f'hand_eye: *{other} in config/real.yaml)')
        if self.config_path:
            set_line(self.config_path, 'hand_eye', f'*{other}')
            self.log(f'saved hand_eye: *{other} in {self.config_path} (the next start uses it)')
        self.state['mount'] = other
        self.update_table(None)                       # the camera box moved with the hand-eye
        return True

    # ================================================================== the steps
    def update_table(self, measured_z):
        """The table height measured from the rack (else the config's) into MoveIt's scene."""
        z = (self.cfg.get('scene') or {}).get('table_top_z')
        if measured_z is not None:
            if z is not None and abs(measured_z - z) > 0.01:
                self.log(f'table measured at z {measured_z:+.3f} (rack top - rack height), config says {z:+.3f}: '
                         f'using the measured one (set scene.table_top_z to it)')
            self.state['table_z'] = float(measured_z)
        self.update_scene()

    def update_scene(self):
        """MoveIt's planning scene from what is known: table, camera, rack, stand (helpers/scene.py)."""
        if self.dry or not hasattr(self.mover, 'set_scene'):
            return
        st = self.state
        stand = (st['stand_seat'], st['stand_tag_xy']) if st.get('stand_seat') and st.get('stand_tag_xy') else None
        bowl = None
        if st.get('bowl'):
            c = st['bowl']['centre']
            h = st['bowl'].get('height') or (c[2] - self.table_z())
            bowl = (c[0], c[1], st['bowl']['radius'], h, c[2])
        self.mover.set_scene(self.cfg, st.get('table_z'), rack=st.get('rack'), stand=stand, bowl=bowl)

    def step1_find_jar(self):
        cfg = self.cfg
        self.gripper(cfg['gripper']['open'], 'open the gripper')
        self.free(self.survey_pose(), 'survey pose')
        time.sleep(0.5)
        ids = list(cfg['tags']['rack_ids']) + [cfg['tags']['stand_id']]
        tags = self.look_tags(ids, 'survey_tags')
        if self.check_mount(tags):
            # the camera was on the other side: turn it outside now and look again
            self.free(self.survey_pose(), 'survey pose, camera outside')
            time.sleep(0.5)
            tags = self.look_tags(ids, 'survey_tags_again')
            if self.mirrored(tags):
                raise StepFailed('the tags come out backwards with the camera on either side: tags 6 and 7 '
                                 'swapped (6 must be left of 7 seen from the arm), or stand.pocket_angle_deg')

        rack = tp.rack_pose(tags, cfg)
        top_z = None
        if rack is None:
            self.log('rack tags not both seen: lid height found from the depth, fingers close towards the jar')
            self.state['rack'] = None
        else:
            if tp.rack_backwards(rack):
                raise StepFailed(tp.BACKWARDS_HINT)
            centre, x_axis, rack_top = rack
            top_z = tp.lid_top_in_rack(rack_top, cfg)
            self.log(f'rack: top centre {fmt(centre)}, +x {fmt(x_axis)}; expected lid top z {top_z:+.4f}')
            self.state['rack'] = {'centre': centre.tolist(), 'x_axis': x_axis.tolist(), 'top_z': rack_top}
            self.update_table(rack_top - cfg['rack']['size'][2])
            size = cfg['rack']['size']
            self.viz('rack', [{'type': 'box', 'center': centre - [0, 0, size[2] / 2], 'size': size,
                               'yaw': math.atan2(x_axis[1], x_axis[0]), 'rgba': M_RACK}])

        # only on the rack: the stand's empty cup is round too
        region = tp.rack_region(rack, cfg) if rack is not None else None
        result = self.look_lids(top_z, label='survey_lids', region=region)
        if not result.lids:
            raise StepFailed(f'no salt lid found from the survey pose: {result.reason}')
        if len(result.lids) == 1:
            lid = result.lids[0]
        elif self.index is not None and 0 <= self.index < len(result.lids):
            lid = result.lids[self.index]
        else:
            raise StepFailed(f'{len(result.lids)} lids seen; pick one with --index N (numbers in '
                             f'{self.out_dir}/*survey_lids.png)')
        top = lid.center
        self.viz_jar(top)
        self.log(f'target lid at {fmt(top)}')
        bowl = self.look_bowl('survey_bowl', rack)      # same still picture spot as the lids

        self.find_stand(tags)
        self.find_bowl(bowl, rack)

        closing = rack[1] if rack is not None else top
        R = self.pick_top_down(closing[:2], top, lambda R: [self.look_pose(top, R),
                                                            tp.rack_top_poses(top, R, cfg)[1]], 'rack')
        top = self.aim(top, tp.rack_top_poses, R, 'in the rack', first_free=True,
                       at_last_look=lambda t: self.take_reference('rack_full', self.rack_pocket(t)))
        self.state.update(rack_lid_top=top.tolist(), R_rack=R.tolist())
        self.lid_fresh = True                          # step 2 right after this: no need to look again

    def find_stand(self, survey_tags):
        """The regrasp stand from tag 8: seen from the survey pose (else from stand_look), then
        measured again from right above it, where the "stand empty" picture is taken too."""
        cfg = self.cfg
        stand_id = cfg['tags']['stand_id']
        seat = self.stand_from_tags(survey_tags)
        if seat is None:
            look = cfg.get('stand_look')
            if look is None:
                raise StepFailed(f'regrasp stand (tag {stand_id}) not seen from the survey pose and no '
                                 f'stand_look pose set (tools/check_vision.py shows why)')
            self.free(look, 'look for the stand')
            seat = self.stand_from_tags(self.look_tags([stand_id], 'stand_look_tags'))
            if seat is None:
                raise StepFailed(f'regrasp stand tag {stand_id} not seen from stand_look either')
        seated = tp.seated_top(seat, cfg)
        R = self.pick_top_down(seat[:2], seat, lambda R: [self.look_pose(seated, R),
                                                          tp.stand_top_poses(seated, R, cfg)[1]], 'stand')
        seat = self.track_stand(seat, R, 'close')
        self.state['R_stand'] = R.tolist()
        self.draw_stand(seat, 'stand_close')
        self.take_reference('stand_empty', self.stand_pocket(seat))

    def track_stand(self, seat, R, label, first_free=True):
        """Tag 8 measured again from right above the stand (look-move-look): from far away the tag
        is small and at the edge of the picture, so a small hand-eye error moves it by cm; and the
        stand may have been nudged since the last look (a grasp, a placed jar). Updates the state
        and MoveIt's scene; returns the pocket floor."""
        cfg, chk = self.cfg, self.cfg['checks']
        stand_id = cfg['tags']['stand_id']
        seat = start = np.asarray(seat, float)
        holding = bool(self.state.get('held'))
        for k in range(1, chk['max_aims'] + 1):
            look = self.look_pose(tp.seated_top(seat, cfg), R)
            if k == 1 and first_free:
                self.free(look, f'camera over the regrasp stand ({label})', holding=holding)
            else:
                self.line(look, f'camera over the regrasp stand ({label}), aim {k}')
            if self.dry:
                break
            time.sleep(0.3)
            new = self.stand_from_tags(self.look_tags([stand_id], f'stand_{label.replace(" ", "_")}{k}_tags'))
            if new is None and self.state.get('stand_tag_xy'):
                # the column's rim can hide a corner of the tag next to it: look from over the tag
                tag_xy = self.state['stand_tag_xy']
                over_tag = [tag_xy[0], tag_xy[1], tp.seated_top(seat, cfg)[2]]
                self.log(f'tag {stand_id} not seen from over the pocket; looking from right above the tag')
                self.line(self.look_pose(over_tag, R), f'camera over the stand tag ({label})')
                time.sleep(0.3)
                new = self.stand_from_tags(self.look_tags([stand_id], f'stand_{label.replace(" ", "_")}{k}b_tags'))
            if new is None:
                self.log(f'tag {stand_id} not seen from above the stand; using the estimate so far')
                break
            d = math.dist(new[:2], seat[:2])
            if d > 0.05:
                raise StepFailed(f'stand ({label}) measured {1000 * d:.0f} mm from the estimate before: tag '
                                 f'{stand_id} misread, or the stand was knocked far (see stand_*_tags.png)')
            self.log(f'stand aim {k}: {1000 * d:.1f} mm from the estimate before')
            seat = new
            if d < 2 * chk['aim_tolerance']:
                break
        moved = math.dist(seat[:2], start[:2])
        if moved > 0.002 and not self.dry:
            self.log(f'stand ({label}): {1000 * moved:.0f} mm from where it was thought to be; using the new place')
        self.state['stand_seat'] = seat.tolist()
        self.update_scene()                            # the stand is an obstacle where it is now
        return seat

    def track_rack(self, top, R, label):
        """Rack tags 6, 7 measured again from right above the slot (the camera beside the jar in the
        hand): the slot moves with the rack if it was nudged. Returns (slot lid top, R)."""
        cfg = self.cfg
        rack = self.state.get('rack')
        top = np.asarray(top, float)
        if not rack:
            return top, R
        # camera over the middle of the two tags (front ledge), a bit higher than for a lid, so both
        # are in the picture and the jar in the hand (beside the camera) does not hide one
        old_c, old_x = np.asarray(rack['centre'], float), np.asarray(rack['x_axis'], float)
        mid = old_c + cfg['rack']['tag_x_in_rack'] * old_x
        mid[2] = top[2]
        # wrist turned so the fingers close along the tag line: the camera (and the jar beside it)
        # are then offset across it, and the tags lie along the picture's wide (87 deg) side
        old_y = np.cross(tp.UP, old_x)
        R_look = tp.top_down_R(old_y[:2], R)
        self.free(self.look_pose(mid, R_look, cfg['lid_detection']['look_height'] + 0.06),
                  f'camera over the rack tags ({label})', holding=bool(self.state.get('held')))
        if self.dry:
            return top, R
        time.sleep(0.3)
        local = np.array([(top - old_c) @ old_x, (top - old_c) @ old_y, top[2] - rack['top_z']])
        new = tp.rack_pose(self.look_tags(list(cfg['tags']['rack_ids']), f'rack_{label.replace(" ", "_")}_tags'), cfg)
        if new is None:
            self.log('rack tags not both seen; using the rack as measured before')
            return top, R
        c, x, top_z = new
        y = np.cross(tp.UP, x)
        slot = c + local[0] * x + local[1] * y
        slot[2] = top_z + local[2]
        moved = math.dist(slot[:2], top[:2])
        turned = math.degrees(math.atan2(np.cross(old_x, x)[2], old_x @ x))
        self.log(f'rack ({label}): slot {1000 * moved:.0f} mm from where it was, rack turned {turned:+.1f} deg')
        if moved > 0.05:
            raise StepFailed(f'rack ({label}) measured {1000 * moved:.0f} mm from before: tags misread, or the rack '
                             'was knocked far')
        R = tp.top_down_R(x[:2], R)
        self.state['rack'] = {'centre': c.tolist(), 'x_axis': x.tolist(), 'top_z': top_z}
        self.state.update(rack_lid_top=slot.tolist(), R_rack=R.tolist())
        self.update_table(top_z - cfg['rack']['size'][2])
        return slot, R

    # ------------------------------------------------------------------ the bowl
    def table_z(self):
        z = self.state.get('table_z')
        return z if z is not None else (self.cfg.get('scene') or {}).get('table_top_z', -0.06)

    def look_bowl(self, label, rack=None):
        """The bowl in a still snapshot from where the arm is now (spice_task/bowl.py), or None."""
        b = self.cfg.get('bowl') or {}
        if not b.get('detect', False):
            return None
        snap = self.io.snapshot(self.cfg['lid_detection'].get('depth_frames', 5))
        if snap is None:
            self.log(f'bowl ({label}): no camera picture')
            return None
        image, depth, T_base_cam = snap
        exclude = []
        if self.state.get('stand_seat'):
            exclude.append((*self.state['stand_seat'][:2], 0.06))
        if rack is not None:
            exclude.append((*np.asarray(rack[0])[:2], 0.5 * min(self.cfg['rack']['size'][:2])))
        bowl, why = detect_bowl(image, depth, self.io.K, self.io.D, T_base_cam, self.table_z(), b,
                                T_base_ee=self.io.ee_pose(),
                                gripper_clear=self.cfg['lid_detection'].get('gripper_clear', 0.165),
                                exclude=exclude)
        picture = image.copy()
        if bowl is not None:
            draw_circle(picture, bowl.center, bowl.radius, T_base_cam, self.io.K, self.io.D, (0, 165, 255),
                        f'bowl d {2000 * bowl.radius:.0f} mm', label_dy=-20)
            self.log(f'bowl ({label}): rim centre {fmt(bowl.center)}, diameter {2000 * bowl.radius:.0f} mm, '
                     f'{100 * bowl.seen:.0f}% of the rim in view, on an edge {100 * bowl.support:.0f}%, '
                     f'inside {1000 * bowl.inside_drop:.0f} mm below the rim')
        else:
            self.log(f'bowl ({label}): not found ({why})')
        self.save_picture(label, picture)
        return bowl

    def find_bowl(self, bowl, rack=None):
        """Step 1: the bowl (no tag; anywhere on the table) from the survey picture, then again
        from right above it so the whole rim is in view. Kept in the state (step 5 pours over it)
        and put into MoveIt's scene. Not found: only step 5 needs it, so step 1 goes on."""
        b = self.cfg.get('bowl') or {}
        if not b.get('detect', False):
            return
        if bowl is None and self.cfg.get('bowl_look'):
            self.free(self.cfg['bowl_look'], 'look for the bowl')
            time.sleep(0.3)
            bowl = self.look_bowl('bowl_look', rack)
        if bowl is None:
            self.state['bowl'] = None
            hint = ('put it where the survey pose sees it (or set bowl_look in config/real.yaml) and run '
                    'step 1 again')
            if 5 in getattr(self, 'steps', ()):
                raise StepFailed(f'no bowl found, and this run pours (step 5): {hint}')
            self.log(f'NO BOWL FOUND: step 5 (pouring) cannot run; {hint}')
            return
        if b.get('close_look', True) and not self.dry:
            R = self.R_current()
            if R[2, 2] > -0.9:                                   # not pointing down
                R = tp.top_down_R([1.0, 0.0])
            # high enough for the whole rim inside the 58 deg side of the picture
            height = max(self.cfg['lid_detection']['look_height'],
                         (bowl.radius + 0.03) / math.tan(math.radians(26)))
            pose = self.look_pose(bowl.center, R, height)
            if self.mover.reachable(pose):
                self.free(pose, 'camera over the bowl')
                time.sleep(0.3)
                close = self.look_bowl('bowl_close', rack)
                if close is not None and math.dist(close.center[:2], bowl.center[:2]) < 0.05:
                    bowl = close
                else:
                    self.log('bowl close-up: not found again near the survey estimate; keeping that')
            else:
                self.log('bowl close-up pose out of reach; keeping the survey estimate')
        self.state['bowl'] = {'centre': [float(v) for v in bowl.center], 'radius': float(bowl.radius)}
        known = b.get('height')
        if known:
            self.state['bowl']['height'] = float(known)
            above = bowl.rim_z - self.table_z()
            if abs(above - known) > 0.01:
                self.log(f'bowl: rim {1000 * above:.0f} mm above the table height in use '
                         f'({self.table_z():+.3f}) but the bowl is {1000 * known:.0f} mm tall: the table under '
                         f'the bowl is {1000 * (above - known):+.0f} mm off that (robot base / hand-eye tilt). '
                         'The rim height (measured) is what the pour uses.')
        self.viz('bowl', [{'type': 'cylinder', 'center': np.asarray(bowl.center) - [0, 0, 0.002],
                           'radius': bowl.radius, 'height': 0.004, 'rgba': (1.0, 0.6, 0.0, 0.6)},
                          {'type': 'text', 'pos': np.asarray(bowl.center) + [0, 0, 0.03], 'text': 'bowl',
                           'rgba': M_TEXT}])
        self.update_scene()

    def step2_grasp_top(self):
        self.need('rack_lid_top', 'R_rack')
        R = np.array(self.state['R_rack'])
        top = np.array(self.state['rack_lid_top'])
        if not getattr(self, 'lid_fresh', False):
            # resumed (not straight after step 1): the rack / the jar may have moved since
            top, R = self.track_rack(top, R, 'before the pick')
            top = self.jar_in_pocket(top, R, 'before the pick', 'rack')
            self.state.update(rack_lid_top=top.tolist(), R_rack=R.tolist())
            self.line(tp.rack_top_poses(top, R, self.cfg)[0], 'right above the lid in the rack')
        self.lid_fresh = False
        approach, grasp, _ = tp.rack_top_poses(top, R, self.cfg)
        self.line(grasp, 'down onto the jar in the rack', slow=True)
        self.close_on_jar('top grasp in the rack')
        self.line(approach, 'lift it straight out of the rack', slow=True)
        self.still_holding('lifting it out of the rack')
        self.state['held'] = 'top'
        self.state['grasp_below_top'] = self.cfg['jar']['top_grasp_below_top']
        if self.cfg['checks']['look_at_empty_slot']:
            # the camera is beside the jar in the hand, so it can see the slot from here
            self.line(self.look_pose(top, R), 'camera over the (now empty) rack slot')
            now = self.check_pocket('rack', self.rack_pocket(top), False, 'after the pick')
            self.take_reference('rack_empty', self.rack_pocket(top), now)

    def held_below(self):
        """How far below the lid top the jar in the hand is held (top grasp: step 2 or step 7)."""
        return self.state.get('grasp_below_top', self.cfg['jar']['top_grasp_below_top'])

    def lid_near(self, top, where):
        """The lid nearest `top` seen from where the arm is (Hough + depth), or None."""
        res = self.look_lids(top[2], around=top[:2], label=f'lid_{where.replace(" ", "_")}')
        if not res.lids:
            self.log(f'{where}: no lid seen ({res.reason})')
            return None
        return min(res.lids, key=lambda l: math.dist(l.center[:2], top[:2])).center

    def put_in_pocket(self, pocket, top, R, clearance, where):
        """The held (top grasp) jar into a pocket whose seated lid top is `top`: lower it to the
        floor (or the mouth: motion.release_above_mouth), wait for it to stop swinging, let go,
        back up, look at its lid. On the pocket axis and not high: done. Off it / high = resting on
        the lead-in or the rim, leaning: pick it up again where it is, over the pocket again, let
        go again (checks.reseat_tries times). Returns the lid top seen (None: no lid seen, the
        caller goes by the pictures)."""
        cfg, ck = self.cfg, self.cfg['checks']
        tol = ck.get('reseat_tolerance', 0.005)
        tries = int(ck.get('reseat_tries', 2))
        approach, _, place = tp.top_poses(top, R, cfg, clearance, self.held_below())
        for attempt in range(tries + 1):
            self.line(place, f'lower the jar into the {pocket} pocket', slow=True)
            self.still_holding(f'lowering it to the {pocket}')
            time.sleep(float(cfg['motion'].get('settle', 0.8)))          # held by the lid, it swings
            if not self.ask('let go of the jar?'):
                raise StepFailed(f'not confirmed: let go in the {pocket}')
            self.gripper(cfg['gripper']['open'], f'let go in the {pocket}')
            time.sleep(0.5)
            self.state['held'] = None
            self.line(approach, 'back up off the jar', slow=True)
            if self.dry:
                return None
            self.line(self.look_pose(top, R), f'camera over the {pocket} pocket')
            lid = self.lid_near(top, f'{where} {attempt + 1}')
            if lid is None:
                return None
            off, high = math.dist(lid[:2], top[:2]), lid[2] - top[2]
            if off <= tol and high <= tol:
                self.log(f'{where}: jar in the {pocket} pocket, lid {1000 * off:.0f} mm off the axis, '
                         f'{1000 * high:+.0f} mm high' + (f' (after {attempt} re-try)' if attempt else ''))
                return lid
            if attempt == tries:
                if off <= ck.get('seat_tolerance', 0.010) and high <= ck.get('seat_tolerance', 0.010):
                    self.log(f'{where}: jar in the {pocket} pocket, but its lid is {1000 * off:.0f} mm off the axis, '
                             f'{1000 * high:+.0f} mm high (more than {1000 * tol:.0f} mm: a little tilted?)')
                    return lid
                raise StepFailed(f'{where}: the jar is not in the {pocket} pocket' + (f' after {tries} re-tries' if tries else '') + ' (lid '
                                 f'{1000 * off:.0f} mm off the axis, {1000 * high:+.0f} mm high): push it in by '
                                 'hand, then spice resume the next step')
            self.log(f'{where}: the jar is resting on the edge of the {pocket} pocket (lid {1000 * off:.0f} mm off '
                     f'the axis, {1000 * high:+.0f} mm high): picking it up again ({attempt + 1}/{tries})')
            a2, g2, _ = tp.top_poses(lid, R, cfg, clearance, self.held_below())
            self.line(a2, 'right above the leaning jar')
            self.line(g2, 'down onto the leaning jar', slow=True)
            self.close_on_jar('picking it up again')
            self.state['held'] = 'top'
            self.line(a2, 'lift it straight up', slow=True)
            self.line(approach, f'over the {pocket} pocket again', slow=True)
        return None

    def step3_place_on_stand(self):
        cfg = self.cfg
        self.need('stand_seat')
        self.still_holding('before carrying it to the stand')
        seat = np.array(self.state['stand_seat'])
        seated = tp.seated_top(seat, cfg)
        below = self.held_below()
        R = self.pick_top_down(seat[:2], seat, lambda R: tp.stand_top_poses(seated, R, cfg, below)[0::2], 'stand')
        seat = self.track_stand(seat, R, 'before placing')          # nudged since step 1?
        seated = tp.seated_top(seat, cfg)
        approach, _, place = tp.stand_top_poses(seated, R, cfg, below)
        self.free(approach, 'carry the jar over the stand', holding=True)
        self.still_holding('carrying it to the stand')
        self.state['R_stand'] = R.tolist()

        lid = self.put_in_pocket('stand', seated, R, cfg['stand']['approach_clearance'], 'after placing')
        if lid is None:                                   # no lid seen: the pictures decide
            self.line(self.look_pose(seated, R), 'camera over the stand')
            self.check_pocket('stand', self.stand_pocket(seat), True, 'after placing')
        self.take_reference('stand_full', self.stand_pocket(seat))       # camera is over the stand
        self.state['stand_lid_top'] = seated.tolist()       # from tag 8 and the stand's shape
        self.viz_jar(seated)

    def side(self, lid_top, R_top):
        """Side-grasp poses (task_poses.side_poses) from a direction the arm can reach. Straight
        from the base needs the hand turned side-on close to the base, which the Gen3 lite often
        cannot do; so other headings and a shorter stand-off are tried (MoveIt IK, no motion).
        The choice is kept in the state, so step 6 uses the same one."""
        cfg = self.cfg
        keep = self.state.get('side_choice')
        if keep:
            return tp.side_poses(lid_top, cfg, R_top, self.camera(), keep['heading'], keep['clear'])
        base = tp.heading_to(lid_top)
        clears = [cfg['motion']['approach_clearance']]
        if clears[0] > 0.06:
            clears.append(0.06)
        tried = []
        for clear in clears:
            for dh in tp.SIDE_HEADINGS_DEG:
                heading = base + math.radians(dh)
                side = tp.side_poses(lid_top, cfg, R_top, self.camera(), heading, clear)
                keys = ('retreat_high_top', 'retreat_high', 'approach', 'grasp', 'lift')
                if self.dry or all(self.mover.reachable(side[key]) for key in keys):
                    if dh or clear != clears[0]:
                        self.log(f'side grasp: from {dh:+d} deg off straight-from-the-base, stand-off '
                                 f'{100 * clear:.0f} cm (the straight one is out of reach)')
                    self.state['side_choice'] = {'heading': heading, 'clear': clear}
                    return side
                tried.append(f'{dh:+d} deg / {100 * clear:.0f} cm')
        raise StepFailed('no side grasp around the jar in the stand that the arm can reach (tried '
                         + ', '.join(tried) + '): move the stand a little further from the arm base')

    def step4_regrasp_side(self):
        self.need('stand_lid_top', 'R_stand')
        R_top = np.array(self.state['R_stand'])
        self.state.pop('side_choice', None)
        # look again: the stand / the jar may have moved since it was put there
        seat = self.track_stand(np.array(self.state['stand_seat']), R_top, 'before the side grasp')
        top = self.jar_in_stand(seat, R_top, 'before the side grasp')
        side = self.side(top, R_top)
        # Slide back still top-down, turn side-on there (clear of the jar), come down beside it.
        self.line(side['retreat_high_top'], 'slide back from the jar (top-down)')
        self.free(side['retreat_high'], 'turn side-on')
        self.line(side['approach'], 'down level with the jar body', slow=True)
        self.gripper(self.cfg['gripper']['open'], 'open for the side grasp')
        self.line(side['grasp'], 'in around the jar body', slow=True)
        self.close_on_jar('side grasp in the stand')
        self.line(side['lift'], 'lift it straight out of the stand', slow=True)
        self.still_holding('lifting it out of the stand')
        self.state['held'] = 'side'

    # ------------------------------------------------------------------ pouring
    def pour_place(self, centre):
        """Pour poses (task_poses.pour_poses) from a direction the arm can reach upright and tipped
        either way (MoveIt IK, no motion): straight from the base first, then others."""
        cfg = self.cfg
        tilt, amp = math.radians(cfg['pour']['tilt_deg']), math.radians(cfg['pour']['shake_deg'])
        base = tp.heading_to(centre)
        for dh in tp.POUR_HEADINGS_DEG:
            p = tp.pour_poses(centre, cfg, self.camera(), base + math.radians(dh))
            if self.dry:
                p['ways'] = [1, -1]
                return p
            if not self.mover.reachable(p['over']):
                continue
            # which way(s) it can be tipped (tilt and both ends of the shake) without hitting anything
            p['ways'] = [s for s in (1, -1)
                         if all(self.mover.reachable(tp.turned(p['over'], s * a)) for a in (tilt, tilt + amp, tilt - amp))]
            if p['ways']:
                if p['raised'] > 0.001:
                    self.log(f'pour: hand {1000 * p["raised"]:.0f} mm higher than pour.above_rim, so the wrist '
                             f'camera stays {1000 * cfg["pour"].get("camera_clear", 0.02):.0f} mm above the rim '
                             'when the jar is tipped')
                if dh:
                    self.log(f'pour: hand from {dh:+d} deg off straight-from-the-base (that one is out of reach)')
                return p
        raise StepFailed('no hand pose over the bowl that the arm can reach upright and tipped: move the bowl '
                         'closer to the arm (or lower pour.above_rim)')

    def tip_direction(self, q6, turn, ways=(1, -1)):
        """+1 / -1: a way (of `ways`: the ones IK found collision-free) joint_6 can turn `turn` rad
        from q6 inside its limits (+-2.6 rad). None q6 (no joint states, sim): the first way."""
        lim = JOINT6_LIMIT - 0.03
        for s in ways:
            if q6 is None or abs(q6 + s * turn) <= lim:
                return s
        raise StepFailed(f'joint_6 at {math.degrees(q6):+.0f} deg cannot turn {math.degrees(turn):.0f} deg '
                         f'the way{"s" if len(ways) > 1 else ""} the hand can tip there: move the bowl a little')

    def turn_wrist(self, over, q_over, angle, what, ask=True):
        """Tip the side-held jar to `angle` (rad, 0 = upright) about the grasp approach: joint_6
        only (q_over: the arm's joints upright over the bowl), so the hand stays where it is and
        nothing else moves. Without joint states (sim): the same as a straight-line move."""
        pose = tp.turned(over, angle)
        self.viz_goal(pose, what)
        if ask and not self.ask(f'{what}: turn the wrist to {math.degrees(angle):+.0f} deg from upright?'):
            raise StepFailed(f'not confirmed: {what}')
        if self.dry:
            return
        if q_over is not None and hasattr(self.mover, 'move_to_joints'):
            q = list(q_over)
            q[5] += angle
            ok = self.mover.move_to_joints(ARM_JOINTS, q, tolerance=0.005)
        else:
            ok = self.mover.move_cartesian(pose)
        if not ok:
            raise StepFailed(f'{what}: the wrist turn failed (see the MoveIt message above)')
        self.arrived(pose, what)

    def arm_joints(self):
        j = getattr(self.io, 'joints', None) or {}
        return [j[n] for n in ARM_JOINTS] if all(n in j for n in ARM_JOINTS) else None

    def step5_shake_over_bowl(self):
        """tran's step 5: carry the side-held jar upright over the bowl, tip it over (joint_6:
        pour.tilt_deg), shake (pour.shake_deg either way, pour.cycles times), upright again."""
        self.need('bowl')
        if self.state.get('held') != 'side':
            raise StepFailed('the jar is not held side-on (step 4 first)')
        self.still_holding('before carrying it to the bowl')
        pr = self.cfg["pour"]
        centre = np.array(self.state['bowl']['centre'])
        self.log(f'bowl rim centre {fmt(centre)}, diameter {2000 * self.state["bowl"]["radius"]:.0f} mm')
        p = self.pour_place(centre)
        over = p['over']
        tilt, amp = math.radians(pr['tilt_deg']), math.radians(pr['shake_deg'])
        # up first (out of the stand, above the rim height), then across, still upright
        T = self.io.ee_pose()
        if T[2, 3] < over[2] - 0.005:
            self.line(tp.as_pose(T[:3, 3] + [0.0, 0.0, over[2] - T[2, 3]], T[:3, :3]), 'up to carrying height')
        self.free(over, 'carry the jar over the bowl (upright)', holding=True)
        self.still_holding('carrying it to the bowl')
        q_over = None if self.dry else self.arm_joints()
        s = self.tip_direction(None if q_over is None else q_over[5], tilt + amp, p['ways'])
        self.turn_wrist(over, q_over, s * tilt, 'tip the jar over the bowl')
        n = int(pr['cycles'])
        if not self.ask(f'shake: {n} times {pr["shake_deg"]:.0f} deg either way?'):
            raise StepFailed('not confirmed: shake')
        for k in range(n):
            self.turn_wrist(over, q_over, s * (tilt + amp), f'shake {k + 1}/{n}', ask=False)
            self.turn_wrist(over, q_over, s * (tilt - amp), f'shake {k + 1}/{n} back', ask=False)
        self.turn_wrist(over, q_over, 0.0, 'jar upright again', ask=False)
        self.still_holding('after shaking')

    def step6_return_to_stand(self):
        self.need('stand_seat', 'R_stand')
        self.still_holding('before taking it back to the stand')
        cfg = self.cfg
        seat = np.array(self.state['stand_seat'])
        seated = tp.seated_top(seat, cfg)
        R_top = np.array(self.state['R_stand'])
        side = self.side(seated, R_top)
        # Without the shake the hand is still where step 4 lifted the jar: a straight-line touch-up.
        # After it (over the bowl): carry it back above that, then straight down onto it.
        if np.linalg.norm(self.io.ee_pose()[:3, 3] - side['lift'][:3]) > 0.01:
            above = tp.as_pose(np.add(side['lift'][:3], [0.0, 0.0, 0.05]), side['R'])
            self.free(above, 'carry the jar back over the stand (upright)', holding=True)
            self.still_holding('carrying it back to the stand')
        self.line(side['lift'], 'over the stand pocket (side grasp)')
        self.still_holding('bringing it back to the stand')
        self.line(side['place'], 'lower it into the stand', slow=True)
        self.still_holding('lowering it into the stand')
        if not self.ask('let go of the jar?'):
            raise StepFailed('not confirmed: let go in the stand')
        self.gripper(cfg['gripper']['open'], 'let go in the stand')
        time.sleep(0.5)
        self.state['held'] = None
        self.line(side['approach'], 'back straight out', slow=True)
        self.line(side['retreat_high'], 'up clear of the jar')
        self.free(side['retreat_high_top'], 'turn top-down')
        self.line(self.look_pose(seated, R_top), 'camera over the stand')
        self.check_pocket('stand', self.stand_pocket(seat), True, 'back in the stand', lid_top=seated)
        self.state['stand_lid_top'] = seated.tolist()

    def step7_regrasp_top(self):
        self.need('stand_lid_top', 'R_stand')
        R = np.array(self.state['R_stand'])
        seat = self.track_stand(np.array(self.state['stand_seat']), R, 'before the top regrasp')
        top = self.jar_in_stand(seat, R, 'before the top regrasp')
        # this grasp is for the insertion into the rack (step 8): pads on the glass body, not the lid edge
        below = self.cfg['jar'].get('insert_grasp_below_top', self.cfg['jar']['top_grasp_below_top'])
        approach, grasp, _ = tp.stand_top_poses(top, R, self.cfg, below)
        self.line(approach, 'right above the jar in the stand')
        self.gripper(self.cfg['gripper']['open'], 'open for the top grasp')
        self.line(grasp, 'down onto the jar in the stand', slow=True)
        self.close_on_jar('top grasp in the stand')
        self.line(approach, 'lift it straight out of the stand', slow=True)
        self.still_holding('lifting it out of the stand')
        self.state['held'] = 'top'
        self.state['grasp_below_top'] = below

    def step8_return_to_rack(self):
        self.need('rack_lid_top', 'R_rack')
        self.still_holding('before taking it back to the rack')
        cfg = self.cfg
        R = np.array(self.state['R_rack'])
        top, R = self.track_rack(np.array(self.state['rack_lid_top']), R, 'before putting it back')
        approach, _, place = tp.rack_top_poses(top, R, cfg, self.held_below())
        self.free(approach, 'carry the jar back over its rack slot', holding=True)
        self.still_holding('carrying it back to the rack')
        if self.put_in_pocket('rack', top, R, cfg['motion']['approach_clearance'], 'back in the rack') is None \
                and not self.dry:
            self.line(self.look_pose(top, R), 'camera over the rack slot')
            self.check_pocket('rack', self.rack_pocket(top), True, 'back in the rack')

    # ================================================================== recovery
    def go_home_safely(self):
        """After a failed step / Ctrl-C: open (if empty), back out, up, home (helpers/retreat.py).
        Holding the jar: ask whether to let go here first. True if home."""
        if self.dry:
            return True
        held = self.state.get('held')
        if held and self.ask_person(f'the jar is in the hand ({held} grasp): let go of it here before going home? '
                             '(N = carry it home)'):
            self.gripper(self.cfg['gripper']['open'], 'let go before going home')
            held = None
            self.state['held'] = None
        self.update_scene()

        def ee_pose():
            T = self.io.ee_pose()
            return tp.as_pose(T[:3, 3], T[:3, :3])
        ok = retreat_and_home(self.mover, self.cfg, ee_pose, log=self.log, held=bool(held),
                              open_gripper=lambda: self.gripper(self.cfg['gripper']['open'], 'open before going home'))
        self.save_state()
        return ok

    # ================================================================== runner
    def run(self, steps):
        self.steps = list(steps)
        self.lid_fresh = False
        self.update_scene()                            # table, camera (+ rack, stand from a resumed state)
        for n in steps:
            name = f'{n}/8 {STEP_NAMES[n]}'
            self.log(f'\n===== step {name} =====')
            try:
                getattr(self, f'step{n}_{STEP_NAMES[n]}')()
            except StepFailed as e:
                self.save_state()
                self.log(f'\nFAILED at step {name}: {e}')
                return False
            self.save_state()
            self.log(f'----- step {name}: ok')
        self.log('\nAll steps done.')
        return True
