#!/usr/bin/env python3
"""Live perception check, no motion (unless --pose): what the task would see from here.

Draws, on the wrist camera image, about once a second:
  * the rack tags (6, 7) and the stand tag (8) with their axes;
  * the predicted stand pocket rim (cyan) from tag 8 -- it must sit on the stand's cup; if it is
    off to one side, fix stand.pocket_angle_deg / tag_to_pocket in config/real.yaml;
  * every Hough-circle lid (yellow) at the rack's lid height (or the highest lid-sized surface);
  * in red, why a tag that is in the picture was not used (e.g. its edge does not measure the
    size in tags.sizes), and why no lid was found.
Prints the numbers. Keys in the window: q = quit, s = save the picture to real/captures/vision/.

  python3 real/tools/check_vision.py                 # from where the arm is
  python3 real/tools/check_vision.py --pose survey   # move to the survey pose first (asks)
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)

import cv2                                                     # noqa: E402
import numpy as np                                             # noqa: E402
import rclpy                                                   # noqa: E402
import yaml                                                    # noqa: E402

from helpers.robot_motion import ArmMover                      # noqa: E402
from spice_task import task_poses as tp                        # noqa: E402
from spice_task.config_file import set_line                    # noqa: E402
from spice_task.io_node import TaskIO, spin_in_background      # noqa: E402
from spice_task.lid_detector import LidDetector, draw_circle   # noqa: E402
from spice_task.tags import draw_tags, measure_tags            # noqa: E402


def text(img, s, row, colour=(255, 255, 255)):
    cv2.putText(img, s, (10, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
    cv2.putText(img, s, (10, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default=os.path.join(ROOT, 'config', 'real.yaml'))
    p.add_argument('--pose', choices=['survey', 'home', 'stand_look'], help='move there first (asks)')
    p.add_argument('--top-z', type=float, help='lid-top height to cut at (default: from the rack tags, else auto)')
    p.add_argument('--once', action='store_true', help='one picture, saved, no window')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()
    io = TaskIO(cfg)
    executor, thread = spin_in_background(io)
    detector = LidDetector(cfg['lid_detection'], cfg['jar']['lid_diameter'])
    # the stand's cup (pocket mouth, seen from above) -- only to check tag_to_pocket / pocket_angle_deg
    cup_detector = LidDetector(dict(cfg['lid_detection'], search_radius=cfg['stand']['tag_to_pocket'] + 0.03),
                               cfg['stand']['pocket_mouth'])
    t = cfg['tags']
    ids = list(t['rack_ids']) + [t['stand_id']]
    sizes = {int(k): float(v) for k, v in t['sizes'].items()}
    out_dir = os.path.join(ROOT, 'captures', 'vision')
    try:
        ready, why = io.wait_ready()
        if not ready:
            raise SystemExit(f'Not ready: {why}')
        if args.pose:
            pose = cfg.get(args.pose) or (cfg['home'] if args.pose == 'survey' else None)
            if pose is None:
                raise SystemExit(f'{args.pose} is not set in {args.config}')
            if args.pose in ('survey', 'home'):                      # camera outside, like the task
                pose = tp.turn_camera_out(pose, cfg['hand_eye']['translation'])[0]
            if input(f'Move to {args.pose} {np.round(pose[:3], 3).tolist()}? [y/N] ').strip().lower() == 'y':
                mover = ArmMover(io, ee_link=cfg['ee_link'], base_frame=cfg['base_frame'], speed=cfg['speed'],
                                 planning=cfg.get('planning'))
                mover.set_scene(cfg)
                if not (mover.move_cartesian(pose) or mover.move_free(pose)):
                    raise SystemExit('could not reach it')
                time.sleep(1.0)
        switched = False
        while rclpy.ok():
            frames = io.frames(t['measure_frames'])
            snap = io.snapshot(cfg['lid_detection']['depth_frames'])
            if not frames or snap is None:
                print('no camera frames')
                continue
            image, depth, T_base_cam = snap
            dropped = {}
            tags = measure_tags(frames, {i: sizes[i] for i in ids}, io.K, io.D, use_depth=t['use_depth'],
                                report=dropped, same_plane=[t['rack_ids']])
            # The rack's and the stand's tags always face the arm: backwards = camera mirrored.
            rack = tp.rack_pose(tags, cfg)
            stand_tag = tags.get(t['stand_id'])
            mirrored = (tp.rack_backwards(rack) if rack is not None else
                        tp.stand_backwards(stand_tag, cfg) if stand_tag is not None else False)
            now = io.mount_name()
            others = [m for m in (cfg.get('hand_eye_mounts') or {}) if m != now]
            if mirrored and now is not None and len(others) == 1 and not switched:
                print(f'tags come out backwards: the camera is mounted "{others[0]}", not "{now}": switching')
                if io.set_mount(others[0]):
                    set_line(args.config, 'hand_eye', f'*{others[0]}')
                    print(f'saved hand_eye: *{others[0]} in {args.config}')
                    switched = True
                    continue                                          # measure again with it
            pic = draw_tags(image.copy(), tags, io.K, io.D, T_base_cam, sizes)
            lines = []
            top_z = args.top_z
            region = None
            rack = tp.rack_pose(tags, cfg)
            if rack is not None:
                lines.append((f'rack top z {rack[2]:+.4f}, +x {np.round(rack[1][:2], 3).tolist()}', (0, 255, 0)))
                if top_z is None:
                    top_z = tp.lid_top_in_rack(rack[2], cfg)
                region = tp.rack_region(rack, cfg)          # lids only on the rack (not the stand's cup)
                if tp.rack_backwards(rack):
                    lines.append(('STOP: ' + tp.BACKWARDS_HINT[:62], (0, 0, 255)))
                    lines.append(('  ' + tp.BACKWARDS_HINT[62:], (0, 0, 255)))
            else:
                lines.append(('rack: tags 6 and 7 not both seen', (0, 0, 255)))
            for i, why in dropped.items():
                lines.append((f'tag {i} {why}', (0, 0, 255)))
            if t['stand_id'] in tags:
                tag = tags[t['stand_id']]
                seat = tp.stand_seat(tag, cfg)
                rim = seat + [0.0, 0.0, cfg['rack']['pocket_depth']]
                draw_circle(pic, rim, cfg['stand']['pocket_mouth'] / 2, T_base_cam, io.K, io.D, (255, 255, 0),
                            'pocket (from tag)', label_dy=-14)
                lines.append((f'stand seat {np.round(seat, 4).tolist()}, tag tilt {tag.tilt_deg():.1f} deg',
                              (255, 255, 0)))
                # measure the cup itself: Hough at the rim height, around the tag
                cup = cup_detector.detect_cup(image, depth, io.K, io.D, T_base_cam, rim[2],
                                              T_base_ee=io.ee_pose(), around=tag.position[:2])
                if cup.lids:
                    c = min(cup.lids, key=lambda lid: abs(np.linalg.norm(lid.center[:2] - tag.position[:2])
                                                          - cfg['stand']['tag_to_pocket']))
                    dist, ang = tp.stand_offset(tag, c.center)
                    off = 1000 * np.linalg.norm(c.center[:2] - seat[:2])
                    draw_circle(pic, c.center, c.radius, T_base_cam, io.K, io.D, (255, 0, 255), 'cup (seen)',
                                label_dy=40, thickness=1)
                    ok = off < 5.0
                    lines.append((f'stand cup measured: tag_to_pocket {dist:.3f}, pocket_angle_deg {ang:.0f} '
                                  f'({off:.0f} mm from the cyan one{"" if ok else ": copy these to real.yaml"})',
                                  (0, 255, 0) if ok else (255, 0, 255)))
                else:
                    lines.append(('stand cup: not found in the image (the cyan circle is from the config only)',
                                  (255, 0, 255)))
            elif t['stand_id'] not in dropped:
                lines.append((f"stand: tag {t['stand_id']} not detected in the image", (0, 0, 255)))
            res = detector.detect(depth, io.K, io.D, T_base_cam, T_base_ee=io.ee_pose(), top_z=top_z,
                                  region=region, image=image)
            for k, lid in enumerate(res.lids):
                draw_circle(pic, lid.center, lid.radius, T_base_cam, io.K, io.D, (0, 255, 255),
                            f'lid {k} z {lid.center[2]:+.3f}')
            z = f'{res.top_z:+.4f}' if np.isfinite(res.top_z) else 'n/a'
            lines.append((f'{len(res.lids)} lid(s) at height {z} ({res.method})' if res.lids else
                          f'0 lids at height {z}', (0, 255, 255)))
            if not res.lids:                                         # long: on two lines
                lines += [(part.strip(), (0, 255, 255)) for part in res.reason.split(';')]
            lines.insert(0, (f'camera mount: {io.mount_name() or "?"}' + (' (switched automatically)' if switched else ''),
                             (255, 255, 255)))
            for row, (s, c) in enumerate(lines):
                text(pic, s, row, c)
            io.publish_picture(pic)                                  # RViz: "What the code detected"
            io.show_markers('lids seen', [{'type': 'cylinder', 'center': lid.center, 'radius': lid.radius,
                                           'height': 0.004, 'rgba': (1, 1, 0, 0.9)} for lid in res.lids])
            io.show_markers('tags', [{'type': 'box', 'center': tag.position, 'size': (sizes[i], sizes[i], 0.002),
                                      'yaw': float(np.arctan2(tag.T[1, 0], tag.T[0, 0])), 'rgba': (0, 1, 0, 0.9)}
                                     for i, tag in tags.items()])
            if t['stand_id'] in tags:
                io.show_markers('stand', [{'type': 'cylinder', 'center': seat + [0, 0, cfg['rack']['pocket_depth'] / 2],
                                           'radius': cfg['stand']['pocket_mouth'] / 2,
                                           'height': cfg['rack']['pocket_depth'], 'rgba': (0, 1, 1, 0.6)}])
            print(' | '.join(s for s, _ in lines))
            for k, lid in enumerate(res.lids):
                print(f'   lid {k}: centre {np.round(lid.center, 4).tolist()}, r {1000 * lid.radius:.1f} mm, '
                      f'depth {lid.fill:.2f}, edge {lid.support:.2f}')
            if args.once:
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, time.strftime('vision_%Y%m%d_%H%M%S.png'))
                cv2.imwrite(path, pic)
                if res.mask is not None:
                    cv2.imwrite(path.replace('.png', '_mask.png'), res.mask)
                if res.birdseye is not None:
                    cv2.imwrite(path.replace('.png', '_topdown.png'), res.birdseye)
                print(f'saved {path}')
                break
            cv2.imshow('check_vision (q quit, s save)', pic)
            key = cv2.waitKey(300) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, time.strftime('vision_%Y%m%d_%H%M%S.png'))
                cv2.imwrite(path, pic)
                if res.mask is not None:
                    cv2.imwrite(path.replace('.png', '_mask.png'), res.mask)
                if res.birdseye is not None:
                    cv2.imwrite(path.replace('.png', '_topdown.png'), res.birdseye)
                print(f'saved {path}')
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        thread.join(timeout=2.0)
        io.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
