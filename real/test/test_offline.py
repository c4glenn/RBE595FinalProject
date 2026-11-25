"""Offline checks of spice_task (no ROS, no robot): pose math, AprilTag + Hough-lid perception on
ray-cast images, and the whole step sequence against test/sim_world.py.

  python3 real/test/test_offline.py          (or: python3 -m pytest real/test)
"""
import copy
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml                                                    # noqa: E402

import sim_world                                               # noqa: E402
from sim_world import JAR, FakeIO, FakeMover, World            # noqa: E402
from helpers.geometry import camera_box_corners               # noqa: E402
from spice_task import task_poses as tp                        # noqa: E402
from spice_task.lid_detector import LidDetector                # noqa: E402
from spice_task.steps import Task                              # noqa: E402
from spice_task.tags import measure_tags                       # noqa: E402

with open(os.path.join(ROOT, 'config', 'real.yaml')) as f:
    CFG = yaml.safe_load(f)
# The sim's camera is HW7's ("outside") mount; real.yaml's hand_eye line is rewritten by real runs.
CFG['hand_eye'] = CFG['hand_eye_mounts']['outside']
CFG['tags']['sizes'] = {6: 0.036, 7: 0.036, 8: 0.036}    # the sim's tags (sim_world.TAG)
CFG['gripper']['reach'] = 0.130                # the sim's fingers (sim_world.World.grasp)
CFG['gripper']['side_reach'] = 0.130           # same fingers for the side grasp in the sim
CFG['jar']['top_grasp_below_top'] = 0.015   # the sim's fingers grip at the lid
HOME = CFG['home']


def test_pose_math():
    for closing in ([1, 0], [0.3, -0.7]):
        R = tp.top_down_R(closing)
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R[:, 2], [0, 0, -1])
        assert np.isclose(abs(R[:2, 1] @ np.array(closing) / np.linalg.norm(closing)), 1)
    R = tp.side_R(math.radians(40))
    assert np.isclose(np.linalg.det(R), 1) and np.allclose(R[:, 0], [0, 0, 1])
    assert np.allclose(R[:, 2], [math.cos(math.radians(40)), math.sin(math.radians(40)), 0])
    top = np.array([0.30, 0.26, 0.08])
    Rt = tp.top_down_R(top[:2])
    approach, grasp, place = tp.stand_top_poses(top, Rt, CFG)
    reach = CFG['gripper']['reach']
    assert np.isclose(grasp[2], top[2] - CFG['jar']['top_grasp_below_top'] + reach)
    assert np.isclose(approach[2] - grasp[2], CFG['stand']['approach_clearance'])
    drop = CFG['motion']['release_above_mouth']
    assert np.isclose(place[2] - grasp[2], CFG['motion']['place_drop'] if drop is None else CFG['rack']['pocket_depth'] + drop)
    side = tp.side_poses(top, CFG, Rt)
    assert np.isclose(side['lift'][2] - side['grasp'][2], CFG['motion']['approach_clearance'])
    pad = np.array(side['grasp'][:3]) + reach * side['R'][:, 2]
    assert np.allclose(pad[:2], top[:2]) and np.isclose(pad[2], top[2] - CFG['jar']['side_grasp_below_top'])
    print('pose math ok')


def test_perception():
    world = World()
    io = FakeIO(world, HOME)
    frames = io.frames(3)
    sizes = {6: 0.036, 7: 0.036, 8: 0.036}
    tags = measure_tags(frames, sizes, io.K, io.D, use_depth=True)
    assert set(tags) == {6, 7, 8}, tags.keys()
    for tag_id, c, _, _ in world.tags:
        err = 1000 * np.linalg.norm(tags[tag_id].position - c)
        print(f'tag {tag_id}: {err:.1f} mm from truth, tilt {tags[tag_id].tilt_deg():.1f} deg')
        assert err < 3.0
    seat = tp.stand_seat(tags[8], CFG)
    err = 1000 * math.dist(seat[:2], world.col)
    print(f'stand seat {err:.1f} mm from truth (z {seat[2]:+.4f} vs {world.seat_z:+.4f})')
    assert err < 4.0 and abs(seat[2] - world.seat_z) < 0.003
    # check_vision's cup measurement gives back the config's tag -> pocket offset
    cup_det = LidDetector(dict(CFG['lid_detection'], search_radius=0.09), CFG['stand']['pocket_mouth'])
    image, depth, T = io.snapshot()
    rim = seat + [0, 0, CFG['rack']['pocket_depth']]
    cup = cup_det.detect_cup(image, depth, io.K, io.D, T, rim[2], T_base_ee=io.ee_pose(),
                             around=tags[8].position[:2])
    assert cup.lids, cup.reason
    dist, ang = tp.stand_offset(tags[8], cup.lids[0].center)
    print(f'stand cup measured: tag_to_pocket {dist:.4f}, angle {ang:.1f} deg '
          f'(config {CFG["stand"]["tag_to_pocket"]}, {CFG["stand"]["pocket_angle_deg"]})')
    assert abs(dist - CFG['stand']['tag_to_pocket']) < 0.004
    assert abs((ang - CFG['stand']['pocket_angle_deg'] + 180) % 360 - 180) < 4
    rack = tp.rack_pose(tags, CFG)
    top_z = tp.lid_top_in_rack(rack[2], CFG)
    (x0, x1), (y0, y1) = tp.rack_region(rack, CFG)
    inside = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1          # noqa: E731
    assert all(inside(c) for c in world.slots) and not inside(world.col), 'lid search area'
    print('lid search area: every rack slot, not the stand')
    true_top = world.jar[2] + JAR['height']
    print(f'expected lid top {top_z:+.4f} (truth {true_top:+.4f})')
    det = LidDetector(CFG['lid_detection'], CFG['jar']['lid_diameter'])
    for z in (top_z, None):
        res = det.detect(depth, io.K, io.D, T, T_base_ee=io.ee_pose(), top_z=z, image=image)
        assert res.method == 'colour', res.method
        assert len(res.lids) == 1, res.reason
        lid = res.lids[0]
        err = 1000 * math.dist(lid.center[:2], world.jar[:2])
        print(f'lid ({"tags" if z else "auto"} height): {err:.1f} mm from truth, z {lid.center[2]:+.4f}, '
              f'r {1000 * lid.radius:.1f} mm')
        assert err < 2.0 and abs(lid.center[2] - true_top) < 0.003
    # holes in the depth on the lid (shiny spots / the "salt" writing) do not matter
    depth2 = depth.copy()
    u, v = (int(round(c)) for c in lid.pixel)
    depth2[v - 6:v + 6, u - 25:u + 25] = np.nan
    res = det.detect(depth2, io.K, io.D, T, T_base_ee=io.ee_pose(), top_z=top_z)
    assert len(res.lids) == 1 and math.dist(res.lids[0].center[:2], world.jar[:2]) < 0.002
    # a shiny lid: no depth on it at all -> the colour Hough still finds it, at the tags' height
    depth3 = depth.copy()
    yy, xx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    depth3[np.hypot(xx - lid.pixel[0], yy - lid.pixel[1]) < 40] = np.nan
    res = det.detect(depth3, io.K, io.D, T, T_base_ee=io.ee_pose(), top_z=top_z, image=image)
    assert res.method == 'colour' and len(res.lids) == 1, res.reason
    err = 1000 * math.dist(res.lids[0].center[:2], world.jar[:2])
    print(f'lid with no depth on it (shiny): {err:.1f} mm from truth, z {res.lids[0].center[2]:+.4f}')
    assert err < 2.0
    print('perception ok')


def run_task(steps, slot=(1, 0), stand_yaw=None, cfg=None, mount=None):
    world = World(slot=slot)
    cfg = cfg or CFG
    io = FakeIO(world, HOME, cfg=cfg if mount else None, mount=mount)
    mover = FakeMover(io)
    out = tempfile.mkdtemp(prefix='spice_task_')
    task = Task(io, mover, cfg, out, auto=True)
    ok = task.run(steps)
    return ok, world, task, mover


def test_full_task():
    for slot in [(1, 0), (0, 1), (2, 1)]:
        ok, world, task, mover = run_task([1, 2, 3, 4, 6, 7, 8], slot=slot)
        assert ok, f'task failed for slot {slot}'
        start = world.slots[slot[0] * 2 + slot[1]]
        err = 1000 * math.dist(world.jar[:2], start)
        print(f'slot {slot}: jar back {err:.1f} mm from its slot centre, {len(mover.moves)} moves')
        assert err < 3.0 and world.held is None
    ok, world, task, mover = run_task([1, 2, 8])
    assert ok
    print('full task ok')


def test_camera_mounted_inside():
    """The camera really is on the other side of the gripper than config says: step 1 must see the
    rack backwards, switch the hand-eye, turn the camera outside, and the task must still work."""
    cfg = copy.deepcopy(CFG)
    cfg['hand_eye'] = cfg['hand_eye_mounts']['outside']           # what the code thinks
    ok, world, task, mover = run_task([1, 2, 3, 4, 6, 7, 8], cfg=cfg, mount='inside')
    assert ok, 'task failed with the camera mounted inside'
    assert task.state.get('mount') == 'inside' and cfg['hand_eye'] == cfg['hand_eye_mounts']['inside']
    err = 1000 * math.dist(world.jar[:2], world.slots[2])
    print(f'camera inside: detected and switched, jar back {err:.1f} mm from its slot, {len(mover.moves)} moves')
    assert err < 3.0 and world.held is None
    # every top-down look/grasp after the switch has the camera outside or sideways, never inside
    cam = np.array(cfg['hand_eye']['translation'])
    from helpers.geometry import quat_to_matrix
    for kind, p in mover.moves[2:]:
        R = quat_to_matrix(*p[3:])
        if R[2, 2] < -0.9:                                          # top-down
            off = (R @ cam)[:2]
            assert off @ np.array(p[:2]) / np.linalg.norm(p[:2]) > -0.02, f'camera inside at {p[:3]}'
    print('camera mount ok')


def test_arm_does_not_follow():
    """MoveIt reports every move done but the arm stays put (driver out of low-level servoing):
    the task must stop at the first move instead of measuring / grasping from the wrong place."""
    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    mover.move_cartesian = mover.move_rrtconnect = lambda pose, **kw: True        # 'done', nothing moves
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    task.arrived = (lambda f: (lambda pose, what, **kw: f(pose, what, wait=0.0, **kw)))(task.arrived)
    assert not task.run([1, 2, 8])
    assert world.held is None
    print('arm not following: stopped')


def test_side_grasp_heading():
    """The arm cannot put its hand within 25 cm of the base (like the real Gen3 lite side-on): the
    side grasp must come from another direction and the whole sequence still work."""
    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    mover.reachable = lambda pose: math.hypot(pose[0], pose[1]) > 0.25
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    assert task.run([1, 2, 3, 4, 6, 7, 8]), 'task failed with the near-base side poses out of reach'
    choice = task.state['side_choice']
    assert abs(math.remainder(choice['heading'] - tp.heading_to(task.state['stand_seat']), math.tau)) > 0.1
    err = 1000 * math.dist(world.jar[:2], world.slots[2])
    print(f'side grasp from {math.degrees(choice["heading"]):+.0f} deg, stand-off {choice["clear"]:.2f}: '
          f'jar back {err:.1f} mm')
    assert err < 3.0


def test_dry_run():
    world = World()
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_dry_'), dry_run=True)
    assert task.run([1, 2, 3, 4, 6, 7, 8])
    assert mover.moves == [] and world.held is None, 'dry run must not move or grasp'
    print('dry run ok (no motion)')


def test_nudged():
    """The stand and the rack are pushed between steps (a grasp bumped them): the task looks again
    before each contact and still puts the jar back in its (moved) slot."""
    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    assert task.run([1])
    world.move_rack(-0.012, 0.010)                      # jar in it, before the pick (resumed step 2 looks again)
    assert task.run([2])
    world.move_stand(0.012, -0.009)                     # before placing (step 3 looks again)
    assert task.run([3])
    world.move_stand(-0.010, 0.012)                     # jar in it, before the side grasp
    assert task.run([4, 6])
    world.move_stand(0.008, 0.008)                      # before the top regrasp
    assert task.run([7])
    world.move_rack(0.011, -0.013)                      # before putting it back
    assert task.run([8]), 'task failed after the stand / rack were nudged'
    err = 1000 * math.dist(world.jar[:2], world.slots[2])
    print(f'nudged rack x2 and stand x3: jar back {err:.1f} mm from its (moved) slot')
    assert err < 3.0 and world.held is None


def test_tilted_jar_stops():
    """The jar is not seated in the stand (here: off the pocket): step 7 stops before grasping."""
    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    assert task.run([1, 2, 3])
    world.jar = world.jar + [0.02, 0.0, 0.03]           # sitting on the rim, lid off the axis
    held_before = world.held
    assert not task.run([7]), 'grasped a jar that is not standing in the pocket'
    assert world.held is held_before is None
    print('jar not seated: stopped before the grasp')


def test_retreat_home():
    """Stopped round the jar (side grasp, empty): open, back out along the approach, up, home.
    Holding it: straight up first instead."""
    from helpers.retreat import retreat_and_home

    class Mover:
        def __init__(self):
            self.moves = []

        def move_cartesian(self, pose, quiet=False):
            self.moves.append(('line', pose))
            return True

        def move_free(self, pose):
            self.moves.append(('free', pose))
            return True

    R = tp.side_R(0.0)                                   # approaching along +x
    here = tp.as_pose([0.20, -0.25, 0.03], R)
    for held, first in ((False, [0.12, -0.25, 0.03]), (True, [0.20, -0.25, 0.11])):
        m, opened = Mover(), []
        assert retreat_and_home(m, CFG, lambda: here, log=lambda *_: None, held=held,
                                open_gripper=lambda: opened.append(1))
        kinds = [k for k, _ in m.moves]
        assert kinds == ['line', 'line', 'free'], kinds
        assert np.allclose(m.moves[0][1][:3], first), m.moves[0][1][:3]
        assert np.isclose(m.moves[1][1][2], 0.25) and m.moves[2][1] == [float(v) for v in CFG['home']]
        assert bool(opened) == (not held)
    m = Mover()                                          # already home: nothing
    assert retreat_and_home(m, CFG, lambda: list(CFG['home']), log=lambda *_: None) and not m.moves
    print('retreat home ok')


def test_bowl():
    """Step 1 finds the bowl (no tag) with Hough + depth, from the survey picture (part of its rim
    in view) and then from right above it; a flat printed ring is not a bowl; no bowl: step 1 still
    works and the state says so."""
    for bowl, expect in [({}, True), ({'centre': (0.36, -0.24), 'radius': 0.12}, True),
                         ({'height': 0.002, 'floor': 0.0}, False), (None, False)]:
        world = World(slot=(1, 0), bowl=bowl)
        io = FakeIO(world, HOME)
        mover = FakeMover(io)
        task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
        assert task.run([1]), bowl
        got = task.state.get('bowl')
        if not expect:
            assert got is None, (bowl, got)
            continue
        b = world.bowl
        off = math.dist(got['centre'][:2], b['centre'])
        assert off < 0.006, (bowl, got)
        assert b['radius'] - b['wall'] - 0.006 < got['radius'] < b['radius'] + 0.004, (bowl, got)
        assert abs(got['centre'][2] - (sim_world.TABLE_Z + b['height'])) < 0.004, (bowl, got)
        assert mover.scene_objects[2] is not None
        print(f'bowl found {1000 * off:.1f} mm off, radius {1000 * got["radius"]:.0f} mm '
              f'(rim {1000 * b["radius"]:.0f}, inside {1000 * (b["radius"] - b["wall"]):.0f})')
    print('bowl ok')


def test_pour():
    """Whole task with step 5: carried upright over the bowl step 1 found, tipped (tilt + shake about
    the approach, hand not moving), upright again, then back to the stand and the rack. And a run
    that pours stops in step 1 (hand empty) if there is no bowl."""
    world = World(slot=(1, 0), bowl={})
    io = FakeIO(world, HOME)
    mover = FakeMover(io)
    task = Task(io, mover, CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    over = []
    turn = task.turn_wrist

    def spy(o, q, angle, what, ask=True):
        turn(o, q, angle, what, ask)
        T = io.ee_pose()
        cam = camera_box_corners(CFG['hand_eye']) @ T[:3, :3].T + T[:3, 3]
        over.append((math.degrees(angle), T[:3, 3] + CFG['gripper']['side_reach'] * T[:3, 2],
                     world.jar_bottom(T), cam[:, 2].min()))
    task.turn_wrist = spy
    assert task.run(list(range(1, 9)))
    b = task.state['bowl']['centre']
    tilt, amp = CFG['pour']['tilt_deg'], CFG['pour']['shake_deg']
    angles = [abs(a) for a, _, _, _ in over]
    assert max(angles) == tilt + amp and angles[-1] == 0, angles
    clear = CFG['pour']['camera_clear']
    for a, point, bottom, cam_low in over:                 # grasp point over the rim centre throughout
        assert math.dist(point[:2], b[:2]) < 0.002 and point[2] - b[2] > CFG['pour']['above_rim'] - 0.002
        assert bottom[2] > b[2], (a, bottom)               # the jar never goes below the rim
        assert cam_low > b[2] + clear - 0.002, (a, cam_low - b[2])   # nor the wrist camera
    slot = world.slots[2]
    assert world.held is None and math.dist(world.jar[:2], slot) < 0.002, world.jar
    print(f'pour ok: tipped to {max(angles):.0f} deg over the bowl, jar back in its slot')

    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    task = Task(io, FakeMover(io), CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    assert not task.run(list(range(1, 9))) and world.held is None and task.state.get('held') is None
    print('pour without a bowl stops in step 1 ok')


def test_pour_test():
    """spice pourtest: step 5 tips from pour.test_start_deg by pour.test_step_deg until a key is
    pressed (here: after 3 steps), records that angle in pour_tests.csv, then upright and back."""
    import csv
    import spice_task.steps as steps_mod
    world = World(slot=(1, 0), bowl={})
    io = FakeIO(world, HOME)
    cfg = copy.deepcopy(CFG)
    cfg['pour']['test_dwell'] = 0.0
    task = Task(io, FakeMover(io), cfg, tempfile.mkdtemp(prefix='spice_task_'), auto=True, pour_test=True)

    class Keys:                                    # "it pours" pressed while turning to the 3rd step
        def __init__(self):
            self.turns = 0

        def take(self):
            return self.turns == 4

        def hold(self):
            pass
    keys = task.pause = Keys()
    angles = []
    turn = task.turn_wrist

    def spy(o, q, angle, what, ask=True):
        turn(o, q, angle, what, ask)
        keys.turns += 1
        angles.append(round(math.degrees(abs(angle)), 3))
    task.turn_wrist = spy
    saved, csv_path = steps_mod.POUR_TESTS_CSV, os.path.join(tempfile.mkdtemp(prefix='pour_data_'), 'pour_tests.csv')
    steps_mod.POUR_TESTS_CSV = csv_path
    try:
        assert task.run(list(range(1, 9)))
    finally:
        steps_mod.POUR_TESTS_CSV = saved
    start, step = cfg['pour']['test_start_deg'], cfg['pour']['test_step_deg']
    assert angles == [start, start + step, start + 2 * step, start + 3 * step, 0.0], angles
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1 and float(rows[0]['pour_angle_deg']) == start + 3 * step, rows
    assert task.state['pour_test']['angle_deg'] == start + 3 * step
    assert world.held is None and math.dist(world.jar[:2], world.slots[2]) < 0.002
    print(f'pour test ok: stopped at {rows[0]["pour_angle_deg"]} deg, recorded, jar back in its slot')


def test_stale_reference_picture():
    """The "with jar" picture is from another time (label turned by a regrasp, other light): the
    lid seen in the pocket still says the jar is back in the stand."""
    world = World(slot=(1, 0))
    io = FakeIO(world, HOME)
    task = Task(io, FakeMover(io), CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
    assert task.run([1, 2, 3, 4])
    real = task.reference
    task.reference = lambda key: real('stand_empty') if key == 'stand_full' else real(key)   # pictures: "not there"
    lines = []
    log = task.log
    task.log = lambda m: (lines.append(m), log(m))
    assert task.run([6])
    assert any('its lid is there' in m and 'pictures disagree' in m for m in lines), lines[-5:]
    print('stale reference picture: the lid decides ok')


def test_placed_off():
    """The jar sits off in the hand: let go at the rack slot's mouth, it rests on the lead-in. The
    lid measured after letting go must say so: a little off -> logged, too far -> the step stops
    (no picking it up again: reseat_tries 0)."""
    for shift, ok in ((0.008, True), (0.015, False)):
        world = World(slot=(1, 0))
        io = FakeIO(world, HOME)
        task = Task(io, FakeMover(io), CFG, tempfile.mkdtemp(prefix='spice_task_'), auto=True)
        assert task.run([1, 2, 3, 4, 6, 7])
        world.held = world.held + [shift, 0.0, 0.0]       # end_effector_link x: horizontal, top-down
        lines = []
        log = task.log
        task.log = lambda m: (lines.append(m), log(m))
        assert task.run([8]) == ok
        assert not any('picking it up again' in m for m in lines)
        assert any(('a little tilted' if ok else 'not in the rack pocket') in m for m in lines), lines[-4:]
    print('placed off: reported (8 mm), stopped (15 mm) ok')


if __name__ == '__main__':
    test_pose_math()
    test_perception()
    test_full_task()
    test_camera_mounted_inside()
    test_arm_does_not_follow()
    test_side_grasp_heading()
    test_dry_run()
    test_retreat_home()
    test_nudged()
    test_tilted_jar_stops()
    test_bowl()
    test_pour()
    test_stale_reference_picture()
    test_placed_off()
    print('\nALL OK')
