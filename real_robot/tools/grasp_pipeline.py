#!/usr/bin/env python3
"""Pick-and-place: find the shakers, the holder and the bowl from one camera snapshot,
pick a shaker (by its label, or by number), put it into the holder.

  1. observe   (optionally move to a views.yaml view, then) one snapshot: colour + depth + TF
  2. perceive  helpers/circle_detector.py (config/spice_detection_real.yaml): shaker lids by height,
               the bowl by colour. The holder from its AprilTag (april_tag_servo.STAND_TAG_ID, pose
               from the tag's corners + depth, pocket offset from the stand geometry there). Each
               lid's top height is measured from depth. With --spice NAME the paper labels are read
               (helpers/spice_ocr.py) and the shaker whose lid carries that label is picked;
               otherwise --index N.
  3. grasp     straight-down pre-grasp above the lid, fingers closing where the neighbouring lids
               leave the most room; close-up lid correction (april_tag_servo.lid_correction, seeded
               with the detected lid); go in, close, check the fingers, lift.
  4. place     into the detected holder's pocket; let go a few mm above its floor; back off.
Asks before every motion unless --auto. --record films the whole run. --dry-run: perception only.

  python3 tools/grasp_pipeline.py --dry-run                 # look and report, no motion
  python3 tools/grasp_pipeline.py --spice salt --record     # pick the shaker labelled "salt"
  python3 tools/grasp_pipeline.py --view 0 --index 0 --auto --record
"""
import argparse
import math
import os
import sys
import threading
import time
import warnings

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import april_tag_servo as servo                                                # noqa: E402
from helpers.circle_detector import (CameraModel, Detection, SpiceCircleDetector,  # noqa: E402
                                     format_table, height_map, load_config)
from helpers.drawing import put_outlined_text                                  # noqa: E402
from helpers.robot_motion import ArmMover, parse_pose                          # noqa: E402

CONFIG = os.path.join(ROOT, 'config', 'spice_detection_real.yaml')
VIEWS = os.path.join(ROOT, 'config', 'views.yaml')
OUT_DIR = os.path.join(ROOT, 'captures', 'pipeline')

DEPTH_FRAMES = 5             # depth images averaged (median) for the snapshot
FINGER_JOINT = 'right_finger_bottom_joint'
FINGER_MOVED_MIN = 0.15      # after closing: below this the fingers never moved (gripper fault)
FINGER_EMPTY_MIN = 0.70      # at or above this they closed on nothing -- calibrate on the real lid
OPEN_NEAR_NEIGHBOURS = 0.35  # gripper opening (~85 mm) when other lids are close; else fully open
NEIGHBOUR_RADIUS = 0.12      # m, lids closer than this to the target count as neighbours
PUT_BACK_DROP = 0.003        # m, putting it back: let go this far above where it was grasped
# gen3_lite_2f open fingers seen from above (team's detected_target.py): pad centres ~50 mm from the
# target's axis, ~24 mm wide across the closing direction.
FINGER_CENTRE_M, FINGER_HALF_WIDTH_M = 0.050, 0.012


# ============================================================================== snapshot
def take_snapshot(node, frames=DEPTH_FRAMES, timeout=10.0):
    """Undistorted colour image, median depth (m, NaN = none) and the camera model, from a still arm."""
    end = time.monotonic() + timeout
    while (node.image is None or node.K is None or node.depth is None) and time.monotonic() < end:
        time.sleep(0.05)
    if node.image is None or node.K is None or node.depth is None:
        raise RuntimeError('no colour image, camera_info or depth from the camera')
    depths, last = [], node.depth_count
    while len(depths) < frames and time.monotonic() < end:
        if node.depth_count != last:
            last = node.depth_count
            d = node.depth.astype(np.float32)
            d[d <= 0] = np.nan
            depths.append(d)
        time.sleep(0.01)
    image, K, D = node.image.copy(), node.K.copy(), node.D.copy()
    T = node.lookup(servo.BASE_FRAME, node.camera_frame)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        depth = np.nanmedian(np.stack(depths), axis=0)
    h, w = depth.shape
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    image = cv2.remap(image, m1, m2, cv2.INTER_LINEAR)
    depth = cv2.remap(depth, m1, m2, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=float('nan'))
    return image, depth, CameraModel(K, T[:3, :3], T[:3, 3])


def measured_top(detector, det, heights, frac=0.6, window=0.015):
    """Median measured height inside the inner `frac` of a detection's circle (near its configured
    top_z), i.e. the real top of that object at its spot on the table."""
    r_px = detector.apparent_diameter_px(det) / 2 * frac
    h, w = heights.shape
    v, u = np.ogrid[:h, :w]
    disc = (u - det.pixel[0]) ** 2 + (v - det.pixel[1]) ** 2 < r_px ** 2
    z = heights[disc]
    z = z[np.isfinite(z) & (np.abs(z - det.world_xyz[2]) < window)]
    return float(np.median(z)) if len(z) > 20 else float(det.world_xyz[2])


# ============================================================================== which shaker
def read_labels(image, spices):
    from helpers.spice_ocr import SpiceReader
    return [label for label in SpiceReader(spices).read(image) if label.name]


def label_shakers(detector, bottles, labels):
    """{bottle number: spice name}: each read label goes to the nearest lid in the image, if it lies
    within about a lid's width of its centre."""
    names = {}
    for label in labels:
        if not bottles:
            break
        nearest = min(bottles, key=lambda b: math.dist(b.pixel, label.center))
        if math.dist(nearest.pixel, label.center) < 1.2 * detector.apparent_diameter_px(nearest):
            names.setdefault(nearest.number, label.name)
    return names


# ============================================================================== grasp geometry
def finger_clearance(target_xy, others_xy, angle):
    """Smallest gap [m] between either open finger and any neighbouring lid (team's detected_target)."""
    ux, uy = math.cos(angle), math.sin(angle)
    vx, vy = -uy, ux
    gap = math.inf
    for side in (1.0, -1.0):
        cx, cy = target_xy[0] + side * FINGER_CENTRE_M * ux, target_xy[1] + side * FINGER_CENTRE_M * uy
        for nx, ny, r in others_xy:
            dx, dy = nx - cx, ny - cy
            along = dx * vx + dy * vy
            clamped = max(-FINGER_HALF_WIDTH_M, min(FINGER_HALF_WIDTH_M, along))
            gap = min(gap, math.hypot(dx * ux + dy * uy, along - clamped) - r)
    return gap


def closing_direction(target_xy, others, R_current):
    """Unit vector (base x-y) the fingers close along: the most room between the neighbours, else
    the current wrist's; of the two opposite signs, the one closer to the current wrist."""
    y_now = np.array([R_current[0, 1], R_current[1, 1]])
    y_now = y_now / (np.linalg.norm(y_now) or 1.0)
    if others:
        angles = sorted(range(-89, 91), key=abs)
        best = max(angles, key=lambda d: round(finger_clearance(target_xy, others, math.radians(d)), 4))
        y = np.array([math.cos(math.radians(best)), math.sin(math.radians(best))])
    else:
        y = y_now
    return y if y @ y_now >= 0 else -y


def top_down_poses(pad_xyz, y_xy, reach, clearance):
    """end_effector_link (above, at) poses, straight down, finger pads at pad_xyz, closing along y_xy."""
    a = np.array([0.0, 0.0, -1.0])
    y = np.array([y_xy[0], y_xy[1], 0.0])
    q = servo.matrix_to_quaternion(np.column_stack((np.cross(y, a), y, a)))
    at = np.asarray(pad_xyz, float) - reach * a
    return [*(at + [0.0, 0.0, clearance]), *q], [*at, *q]


# ============================================================================== perception
VIEW_MARGIN = 0.02           # m, added around the camera's view when setting the search region
TABLE_TILT_MAX = 4.0         # deg: above this the report warns that the camera pose looks off (info only)
TABLE_Z = -0.04              # m, table height used to work out what the camera sees


def view_region(cam, width, height):
    """Top-down search region (x_range, y_range in base_link) = where the image's corners land on the
    table, plus a margin: follows the camera instead of a fixed config range."""
    Hinv = np.linalg.inv(cam.image_from_plane(TABLE_Z))
    pts = []
    for u, v in [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]:
        p = Hinv @ [u, v, 1.0]
        pts.append(p[:2] / p[2])
    pts = np.array(pts)
    return ([float(pts[:, 0].min() - VIEW_MARGIN), float(pts[:, 0].max() + VIEW_MARGIN)],
            [float(pts[:, 1].min() - VIEW_MARGIN), float(pts[:, 1].max() + VIEW_MARGIN)])


def table_tilt(image, depth, cam):
    """Tilt [deg] of the table (its orange-wood pixels) fitted as a plane in base_link. A flat table
    coming out tilted means the camera's pose is wrong (it moved on its mount, or a bad calibration).
    None if too little table is visible."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    orange_wood = (hsv[..., 0] >= 10) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 60)
    table = orange_wood & np.isfinite(depth) & (depth < 1.0)
    v, u = np.nonzero(table)
    if len(u) < 5000:
        return None
    pick = np.linspace(0, len(u) - 1, min(len(u), 20000)).astype(int)
    u, v = u[pick], v[pick]
    z = depth[v, u]
    x_cam = (u - cam.K[0, 2]) / cam.K[0, 0] * z
    y_cam = (v - cam.K[1, 2]) / cam.K[1, 1] * z
    p = np.column_stack((x_cam, y_cam, z)) @ cam.R + cam.position              # base_link points
    A = np.column_stack((p[:, 0], p[:, 1], np.ones(len(p))))
    (a, b, _), *_ = np.linalg.lstsq(A, p[:, 2], rcond=None)
    return float(np.degrees(np.arctan(np.hypot(a, b))))


HOLDER_TAG_TIMEOUT = 2.0     # s, per snapshot: how long to look for the holder's tag


def detect_holder(node, cam):
    """The holder from its AprilTag (servo.STAND_TAG_ID), measured on the live stream with the arm still.
    -> Detection centred on the pocket, world_xyz z = pocket rim top; None if the tag isn't seen."""
    T = servo.measure_tag_depth(node, servo.STAND_TAG_ID, timeout=HOLDER_TAG_TIMEOUT)
    if T is None:
        return None
    rim = servo.stand_seat(T) + [0.0, 0.0, servo.STAND_RIM_ABOVE_SEAT]
    px = cam.project(rim)
    return Detection(kind='holder', world_xyz=tuple(float(c) for c in rim),
                     pixel=(float(px[0]), float(px[1])), radius_px=0.0, fill=1.0)


def perceive(node, args, out_dir):
    """One snapshot -> shaker / bowl / holder detections (+ OCR names), saved to out_dir.
    Returns (result, detector, heights, {shaker number: name}, annotated picture)."""
    image, depth, cam = take_snapshot(node)
    cfg = load_config(CONFIG)
    cfg['birdseye']['x_range'], cfg['birdseye']['y_range'] = view_region(cam, image.shape[1], image.shape[0])
    cfg['holder'] = {'diameter': servo.STAND_POCKET_MOUTH}          # only used to draw it
    detector = SpiceCircleDetector(cfg, cam)
    heights = height_map(depth, cam)
    result = detector.detect(image, heights, kinds=('bottle', 'bowl'))
    result.holder = detect_holder(node, cam)
    result.table_tilt = table_tilt(image, depth, cam)
    labels = read_labels(image, args.spices) if (args.spice or args.ocr) else []
    names = label_shakers(detector, result.bottles, labels)

    picture = detector.annotate(image, result)
    for b in result.bottles:
        if b.number in names:
            put_outlined_text(picture, names[b.number], (b.pixel[0] + 22, b.pixel[1] + 6), 0.7, (0, 255, 0),
                              outline=4)
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, 'snapshot.png'), image)
    np.save(os.path.join(out_dir, 'snapshot_depth.npy'), depth)
    with open(os.path.join(out_dir, 'snapshot_camera.yaml'), 'w') as f:
        yaml.safe_dump(cam.to_dict(), f)
    cv2.imwrite(os.path.join(out_dir, 'detections.png'), picture)
    return result, detector, heights, names, picture


def missing_objects(result, names, args):
    """What the run still needs and the snapshot didn't find (empty list = ready)."""
    missing = []
    try:
        choose_target(result, names, args)
    except RuntimeError:
        missing.append(f'shaker "{args.spice}"' if args.spice else f'shaker {args.index}')
    if result.holder is None and not args.hold:
        missing.append('holder')
    return missing


def perceive_until_ready(node, args, out_dir):
    """Snapshot + detection, repeated about once a second until everything needed is found; the
    detections are shown live meanwhile (move the arm or the objects to help). q / Ctrl+C quits.
    Returns (perception results..., the end-effector pose at the successful snapshot) or None."""
    print('Looking (keep the arm still while each snapshot is taken) ...')
    last_msg, attempt = None, 0
    while rclpy.ok():
        attempt += 1
        E = node.lookup(servo.BASE_FRAME, servo.EE_FRAME)
        pose = [*E[:3, 3], *servo.matrix_to_quaternion(E[:3, :3]).tolist()]
        result, detector, heights, names, picture = perceive(node, args, out_dir)
        missing = missing_objects(result, names, args)
        if args.dry_run or not missing:
            perceive_report(result, detector, heights, names, out_dir)
            return result, detector, heights, names, picture, pose
        msg = 'waiting for: ' + ', '.join(missing)
        if msg != last_msg:
            print(f'{msg} (attempt {attempt}; still looking -- adjust the view or the scene, q to quit)')
            last_msg = msg
        if args.auto:
            time.sleep(1.0)
            continue
        view = picture.copy()
        put_outlined_text(view, msg + '   (q = quit)', (10, view.shape[0] - 15), 0.7, (0, 0, 255))
        cv2.imshow('detections', view)
        if (cv2.waitKey(1000) & 0xFF) == ord('q'):
            print('Quit while looking.')
            return None
    return None


def perceive_report(result, detector, heights, names, out_dir):
    """Print what the snapshot found: table flatness, the detection table, each shaker's lid height."""
    tilt = getattr(result, 'table_tilt', None)
    if tilt is None:
        flatness = 'not enough table in view'
    elif tilt <= TABLE_TILT_MAX:
        flatness = f'{tilt:.1f} deg (ok)'
    else:
        flatness = f'{tilt:.1f} deg (warning: camera pose may be off)'
    print(f'Table flatness check: {flatness}')
    print(format_table(result))
    for b in result.bottles:
        print(f'  shaker {b.number}: lid top measured {measured_top(detector, b, heights):+.4f}'
              + (f', label "{names[b.number]}"' if b.number in names else ''))
    print(f'Snapshot and detections saved in {os.path.normpath(out_dir)}')


def choose_target(result, names, args):
    """The shaker to pick: the one labelled --spice, else shaker number --index. Raises if absent."""
    if args.spice:
        hits = [b for b in result.bottles if names.get(b.number) == args.spice.lower()]
        if not hits:
            read = sorted(names.values()) or 'none'
            raise RuntimeError(f'no shaker labelled "{args.spice}" found (read: {read})')
        return hits[0]
    hits = [b for b in result.bottles if b.number == args.index]
    if not hits:
        raise RuntimeError(f'shaker {args.index} not found ({len(result.bottles)} detected)')
    return hits[0]


# ============================================================================== motion
class FingerWatch:
    """Latest gripper finger joint position from /joint_states (None until the first message)."""

    def __init__(self, node):
        self.position = None
        node.create_subscription(JointState, '/joint_states', self._on, 10)

    def _on(self, msg):
        if FINGER_JOINT in msg.name:
            self.position = msg.position[msg.name.index(FINGER_JOINT)]


def grasp(node, mover, fingers, target, lid_top, others, recorder):
    """Pre-grasp above the lid, close-up correction, open, go in, close, check, lift.
    Returns (grasp_depth below the lid top, lifted pose, grasp pose) on success, None otherwise."""
    stage = recorder.set_stage
    lid_xy = np.array(target.world_xyz[:2])
    R_now = node.lookup(servo.BASE_FRAME, servo.EE_FRAME)[:3, :3]
    y = closing_direction(lid_xy, others, R_now)
    if others:
        gap = finger_clearance(lid_xy, others, math.atan2(y[1], y[0]))
        print(f'Fingers close at {math.degrees(math.atan2(y[1], y[0])):.0f} deg; '
              f'{1000 * gap:.0f} mm from the nearest neighbour when open')
    depth_below = servo.GRASP_BELOW_LID
    pad = np.array([lid_xy[0], lid_xy[1], lid_top - depth_below])
    pre, at = top_down_poses(pad, y, servo.FINGERTIP, servo.PREGRASP)
    print(f'Target lid at ({lid_xy[0]:.3f}, {lid_xy[1]:.3f}), top {lid_top:+.4f}; '
          f'pads close at {np.round(pad, 4)}')

    if not servo.ask('Move above the shaker (pre-grasp) with MoveIt?'):
        return None
    stage('moving above the shaker')
    if not mover.move_to(pre):
        print('Pre-grasp failed (see the MoveIt message above).')
        return None

    # close-up lid correction, seeded with the detected lid (instead of a tag)
    stage('measuring the lid up close')
    node.last_tag_xyz = np.array([lid_xy[0], lid_xy[1], lid_top])
    corrected = servo.lid_correction(node, pad, recorder)
    if corrected is not None:
        pre_c, at_c, _ = corrected
        shift = np.linalg.norm(np.subtract(at_c[:2], at[:2]))
        print(f'Lid correction moves the grasp by {1000 * shift:.1f} mm')
        if not servo.ask('Move above the corrected lid centre?'):
            return None
        stage('moving above the measured lid centre')
        if not mover.move_to(pre_c):
            print('Could not reach the corrected pre-grasp.')
            return None
        pre, at = pre_c, at_c
    else:
        print('No close-up correction; using the snapshot detection.')

    opening = OPEN_NEAR_NEIGHBOURS if others else servo.GRIPPER_OPEN
    stage('opening the gripper')
    if not node.set_gripper(opening):
        print('Could not open the gripper.')
        return None
    if not servo.ask(f'Go down {100 * servo.PREGRASP:.0f} cm onto the shaker (straight line)?'):
        return None
    stage('going in')
    if not mover.move_cartesian(at):
        print('Straight-line approach failed; staying above.')
        return None
    if not servo.ask('Close the gripper?'):
        return None
    stage('closing the gripper')
    node.set_gripper(servo.GRIPPER_CLOSED)
    time.sleep(0.5)
    p = fingers.position
    print(f'Fingers at {p}' if p is None else f'Fingers stopped at {p:.3f}')
    if p is not None and p < FINGER_MOVED_MIN:
        print('The fingers barely moved: gripper not responding (relaunch the driver).')
        return None
    if p is not None and p >= FINGER_EMPTY_MIN and not servo.ask(
            f'Fingers closed past {FINGER_EMPTY_MIN} (probably empty). Lift anyway?'):
        return None
    if not servo.ask('Lift it straight up?'):
        return None
    stage('lifting')
    if not mover.move_cartesian(pre):
        print('Lift failed; still holding it.')
        return None
    return depth_below, pre, at


def put_back(node, mover, lifted, at, recorder):
    """Put the held shaker back where it was picked up: back over that spot, straight down to
    PUT_BACK_DROP above the grasp pose, let go, straight back up."""
    stage = recorder.set_stage
    if not servo.ask('Put the shaker back down where it was?'):
        print('Still holding the shaker.')
        return False
    stage('putting it back')
    if not mover.move_cartesian(lifted) and not mover.move_rrtconnect(lifted):
        print('Could not get back over its spot; still holding the shaker.')
        return False
    down = [at[0], at[1], at[2] + PUT_BACK_DROP, *at[3:]]
    if not mover.move_cartesian(down):
        print('Could not lower it; still holding the shaker.')
        return False
    stage('letting go')
    if not node.set_gripper(servo.GRIPPER_OPEN):
        print('Gripper did not report opening; not backing off.')
        return False
    time.sleep(0.5)
    stage('backing off')
    mover.move_cartesian(lifted)
    print('Shaker put back where it was.')
    return True


def place_in_holder(node, mover, holder, holder_top, grasp_depth, recorder):
    """Carry the held shaker over the detected holder, lower it into the pocket, let go, back off."""
    stage = recorder.set_stage
    seat = np.array([holder.world_xyz[0], holder.world_xyz[1], holder_top - servo.STAND_RIM_ABOVE_SEAT])
    R_now = node.lookup(servo.BASE_FRAME, servo.EE_FRAME)[:3, :3]
    y = np.array([R_now[0, 1], R_now[1, 1]])
    y /= np.linalg.norm(y)
    pad = seat + [0.0, 0.0, servo.STAND_DROP + servo.SHAKER_HEIGHT - grasp_depth]   # jar bottom STAND_DROP up
    above, place = top_down_poses(pad, y, servo.FINGERTIP, servo.STAND_CLEARANCE - servo.STAND_DROP)
    print(f'Holder pocket floor at {np.round(seat, 4)}; letting go with the pads at {np.round(pad, 4)}')

    if not servo.ask('Carry the shaker over the holder (straight line)?'):
        return False
    stage('carrying it to the holder')
    if not mover.move_cartesian(above):
        if not servo.ask('No straight-line carry. Plan a free move instead (watch the hanging shaker)?'):
            return False
        if not mover.move_rrtconnect(above):
            print('Could not get over the holder; still holding the shaker.')
            return False
    if not servo.ask('Lower it into the holder?'):
        return False
    stage('lowering it into the holder')
    if not mover.move_cartesian(place):
        print('Lowering failed; still holding it above the holder.')
        return False
    stage('letting go')
    if not node.set_gripper(servo.GRIPPER_OPEN):
        print('Gripper did not report opening; not backing off.')
        return False
    time.sleep(0.5)
    stage('backing off')
    if not mover.move_cartesian(above):
        print('Back-off failed.')
    print('Shaker placed in the holder.')
    return True


def return_to_start(node, mover, fingers, start_pose, recorder):
    """Back to the pose the run started from (the observe pose) -- but not with the fingers closed:
    after a failure it may still be holding the shaker, and that is left where it is."""
    p = fingers.position
    if p is not None and p > FINGER_MOVED_MIN:
        print(f'Not returning to the start pose: the fingers are closed ({p:.2f}), '
              'it may be holding the shaker.')
        return
    E = node.lookup(servo.BASE_FRAME, servo.EE_FRAME)
    if np.linalg.norm(E[:3, 3] - start_pose[:3]) < 0.005:
        return                                                     # never left it
    if not servo.ask('Return to the starting (observe) pose?'):
        return
    recorder.set_stage('returning to the start pose')
    if mover.move_to(start_pose):
        print('Back at the starting pose.')
    else:
        print('Could not return to the starting pose (see the MoveIt message above).')


# ============================================================================== main
def run(node, args, recorder):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(OUT_DIR, stamp)
    mover = ArmMover(node, ee_link=servo.EE_FRAME, base_frame=servo.BASE_FRAME, speed=0.1,
                     motion='rrtconnect')
    fingers = FingerWatch(node)

    if args.view is not None:
        with open(VIEWS) as f:
            pose = parse_pose(yaml.safe_load(f)['views'][args.view], f'view {args.view}')
        if not servo.ask(f'Move to view {args.view} to look at the table?'):
            return
        if not mover.move_to(pose):
            print('Could not reach the view.')
            return
        time.sleep(1.0)

    found = perceive_until_ready(node, args, out_dir)
    if found is None:
        return
    result, detector, heights, names, picture, start_pose = found   # start_pose: returned to at the end
    if args.record:
        recorder.start(picture)
        recorder.show_still(picture, seconds=3.0)
        recorder.start_live(node)
    recorder.set_stage('perception done')
    if not args.auto:                                   # show what was found, live
        cv2.imshow('detections', picture)
        cv2.waitKey(500)
    if args.dry_run:
        print('Dry run: no motion.')
        return
    target = choose_target(result, names, args)
    lid_top = measured_top(detector, target, heights)
    others = [(b.world_xyz[0], b.world_xyz[1], detector.cfg['bottle']['diameter'] / 2)
              for b in result.bottles if b is not target
              and math.dist(b.world_xyz[:2], target.world_xyz[:2]) < NEIGHBOUR_RADIUS]
    name = f' ("{names[target.number]}")' if target.number in names else ''
    print(f'\nTarget: shaker {target.number}{name}, {len(others)} neighbouring lid(s)')

    if result.holder is None and not args.hold:
        print('No holder detected: the shaker will be put back where it was after the pick.')
    interrupted = False
    try:
        grasped = grasp(node, mover, fingers, target, lid_top, others, recorder)
        if grasped is None:
            return
        grasp_depth, lifted, at = grasped
        if args.hold:
            print('Holding the shaker (--hold).')
            return
        placed = False
        if result.holder is not None:
            holder_top = result.holder.world_xyz[2]                # pocket rim top, from the tag
            placed = place_in_holder(node, mover, result.holder, holder_top, grasp_depth, recorder)
        if not placed:
            put_back(node, mover, lifted, at, recorder)
    except KeyboardInterrupt:
        interrupted = True                                         # Ctrl+C means stop: no more motion
        raise
    finally:
        if not args.hold and not interrupted:
            return_to_start(node, mover, fingers, start_pose, recorder)
    recorder.set_stage('done')
    time.sleep(1.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spice', help='pick the shaker whose label reads this (OCR)')
    p.add_argument('--index', type=int, default=0, help="otherwise: the detector's shaker number")
    p.add_argument('--spices', nargs='+', default=['salt', 'pepper', 'paprika', 'cumin', 'oregano', 'garlic'],
                   help='OCR vocabulary')
    p.add_argument('--ocr', action='store_true', help='read the labels even without --spice (to report them)')
    p.add_argument('--view', type=int, help='move to this views.yaml view before the snapshot')
    p.add_argument('--hold', action='store_true', help='pick and hold, do not place')
    p.add_argument('--dry-run', action='store_true', help='perception only, no motion')
    p.add_argument('--auto', action='store_true', help='no prompts')
    p.add_argument('--record', action='store_true', help='film the whole run (no pop-ups)')
    args = p.parse_args()
    servo.AUTO, servo.RECORD_ALL = args.auto, args.record

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = servo.TagServoNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    recorder = servo.Recorder()
    try:
        run(node, args, recorder)
    except KeyboardInterrupt:
        print('\nInterrupted.')
    except RuntimeError as e:
        print(f'Stopped: {e}')
    finally:
        recorder.stop()
        cv2.destroyAllWindows()
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
