# Spice task on the real Kinova Gen3 Lite

The tran branch's sim task (`rbe595_manipulation/steps/`: grasp, regrasp, return) on the real
arm and wrist RealSense D405. What changed from the sim:

| | Sim (tran) | Real (here) |
|---|---|---|
| Salt jar | cap AprilTag 0-5 | no tag: "salt" written on it, its round lid found with **Hough circles** on the colour image seen top-down at the lid height (like main's `circle_detector.py`; works on a shiny lid with no depth), the depth as a fallback |
| Rack | tags 6, 7 | tags 6, 7 (same) |
| Regrasp stand | tag 8 | tag 8 only (no Hough): the jar's place = tag + the stand's shape |
| Motion | `ArmMoveClient` (MoveGroup) | HW7 `helpers/robot_motion.py` `ArmMover`: straight lines with `compute_cartesian_path`, free moves with RRTConnect |
| Bringup | Gazebo launch | HW7 `bringup_robot.launch.py` / `bringup_camera.launch.py` (copied from `rbe_vbm_test`) |
| Hand-eye | sim mount | HW7 calibration (`config/real.yaml` `hand_eye`), published by `helpers/hand_eye_tf.py`; camera **outside** or **inside** the gripper found by itself (see below) |

`helpers/` and `launch/` are copies of your HW7 code (`rbe_vbm_test` / `grasp_perception/helpers`),
pointed at `config/real.yaml` instead of `views.yaml`. The task itself is in `spice_task/`:

| File | What it does |
|---|---|
| `steps.py` | the 8 steps, every motion asked first |
| `lid_detector.py` | colour image warped top-down onto the lid-height plane -> `cv2.HoughCircles` (depth only to drop circles at the wrong height, e.g. empty pockets); fallback: depth height map -> lid mask -> Hough |
| `reference.py` | "is the jar in the pocket?": the pocket now vs reference pictures (empty / with jar), compared top-down |
| `tags.py` | tag36h11 detection; flat tags get their yaw from the corners on the measured plane |
| `task_poses.py` | every end-effector pose (port of tran's `poses.py`) |
| `io_node.py` | ROS: colour, aligned depth, TF, gripper action, finger joint |

## Steps

| Step | What happens | Reference |
|---|---|---|
| 1 find_jar | survey pose: tags 6, 7, 8 and the lid; camera over the stand: tag 8 close up, look-move-look until it stops moving, + "stand empty" picture; the bowl (Hough + depth, then from right above it); camera over the lid, measure, move, until within 1.5 mm; "slot with jar" picture | tags, lid |
| 2 grasp_top | down, close, check the fingers, up; camera over the slot: it must be empty now ("slot empty" picture) | lid, pictures |
| 3 place_on_stand | carry over the stand pocket, lower, let go, back up; jar there? ("stand with jar" picture) | tag 8, pictures |
| 4 regrasp_side | slide back top-down, turn side-on, come in level, close, lift; from straight out of the base if the arm can reach that, else 30/60/90 deg round and/or a 6 cm stand-off (MoveIt IK picks) | tag 8 |
| 5 shake_over_bowl | up, carry the side-held jar upright over the bowl (its grasp point `pour.above_rim` above the rim centre), tip it over by turning joint_6 only (`pour.tilt_deg`, whichever way MoveIt finds collision-free and joint_6's +-149 deg allows), shake `pour.cycles` times +- `pour.shake_deg`, upright again. Tipped past 90 deg the wrist camera swings below the hand, so the hand goes higher than `above_rim` when needed to keep the camera box `pour.camera_clear` (2 cm) above the rim (the log says by how much). MoveIt's bowl: a cylinder from the measured rim down by `bowl.height` (37 mm, measured by hand). A run with step 5 stops in step 1 (hand empty) if no bowl was found | bowl (step 1) |
| 6 return_to_stand | (after step 5: carry it back above the stand first) lower into the pocket, let go, back out, turn top-down; jar there? | tag 8, pictures |
| 7 regrasp_top | top grasp in the stand | tag 8 |
| 8 return_to_rack | back over its slot, lower, let go, back up; jar there? | slot from step 1, pictures |

Any failed check (lid not seen, not seated, fingers closed on nothing) stops the run with the
reason. The state (where the jar, rack and stand are) is saved after every step in
`captures/<time>/state.json` with a picture of every look, so you can fix something and go on
with `--resume`.

## Camera mount: outside or inside (automatic)

The D405 can sit on either side of the gripper ("outside" = HW7: at home, on the side away from
the robot; "inside" = turned 180 deg about the gripper axis). You do not have to set this:

1. The rack's tags (6, 7) and the stand's tag 8 always face the arm. If, with the current
   `hand_eye`, the rack comes out facing away from the robot (or the stand pocket lands on the
   robot's side of tag 8), the camera must be mounted the other way round.
2. Step 1 and `spice vision` then switch the hand-eye TF live (`helpers/hand_eye_tf.py`, topic
   `/spice/hand_eye_mount`), print `camera mount: ... (switched automatically)`, and rewrite the
   `hand_eye: *outside` / `*inside` line in `config/real.yaml` so the next run starts right.
3. Every top-down pose (survey, over the lid, over the slot / stand) is picked so the camera is
   **outside** (away from the robot) when joint 6 can turn that far; the side grasp keeps the
   camera on top. If the tags still come out backwards after the switch, step 1 stops.

Check it in RViz: the white camera box must sit on the real camera, and in `spice vision` the
rack's +x should be about [1, 0] (pointing away from the robot).

## Before the first run

1. **Stick the tags** (tag36h11; `tags.sizes` is the **black square's** edge, not the sticker's: a 36 mm
   sticker with its white margin, as in the rack's pockets, is 28.8 mm black): 6 and 7 on the rack's front ledge (6 on the
   rack's left as seen from the arm, i.e. base +y), 8 on the stand's plate. On our stand the tag's
   +x (its right, red axis) points away from the column (`pocket_angle_deg: 180`; `spice vision`
   checks it). Put the stand so the **column does not hide its
   tag** from the survey pose (tag on the arm's side of the column).
2. **Measure** the lines marked `MEASURE` in `config/real.yaml`: jar height and lid diameter,
   tag sizes, `gripper.holding_max` (close on the jar, read `right_finger_bottom_joint` in
   `ros2 topic echo /joint_states`; set it ~0.1 above that).
3. The camera runs at 1280x720, 15 fps (`spice.sh` / `bringup_camera.launch.py`; the D405's own
   default, 848x480, makes the tags ~25 px wide from the survey pose and their size check flaky).
   Another: `CAMERA_PROFILE=848x480x30 spice ...`.

## Run

New computer? Once: `real/install.sh` (apt packages + builds the Kinova driver from the repo's
`underlay_ws` submodule). Nothing from HW7 is needed: `helpers/` and `launch/` are copies.

Check everything is installed and plugged in (robot reachable, D405 on USB):

```bash
real/spice.sh check
```

Shortcut for the current terminal only (nothing is written to `~/.bashrc`; close the terminal
or run `spice_off` and it is gone). It also sources ROS and the Kinova driver, so `ros2 ...` works:

```bash
source real/activate.sh
```

Each command below sources ROS and the Kinova driver, starts the robot (the arm goes **home**)
and the camera in the background, waits for them, runs, and stops them at the end or on Ctrl-C.
Logs: `real/captures/logs/`. Go one level at a time.

| Command | What it does |
|---|---|
| `spice vision` | level 0, no motion: tags 6, 7, 8, cyan predicted stand pocket, yellow salt lid (`q` quit, `s` save) |
| `spice dry` | no motion: what it sees + every pose it would go to |
| `spice level1` | level 1: pick the jar out of the rack, put it back (steps 1, 2, 8) |
| `spice full` | the whole task, steps 1-8 with the shake over the bowl (`spice full --steps 1-4,6-8`: without it) |
| `spice resume 4,6,7,8` | carry on from the last run's saved state after fixing a failure |
| `spice home` | get the arm home safely from where a run stopped: open (if empty), back out along the approach (holding the jar: straight up), up to z 0.25, then a planned move home. A failed step, `q` after a pause or Ctrl-C asks the same (`[y/N]`), and the robot bringup does it too before its own home move |
| (automatic, step 1) find the bowl | no tag, so found anywhere on the table by its rim (`spice_task/bowl.py`): the survey picture is warped onto a horizontal plane (rim = true circle), `cv2.HoughCircles` finds circles of `bowl.diameter_range`, and the depth decides which is a bowl (rim raised `min_height` above the table, inside `min_depth` lower than the rim; a printed ring, plate or lid is not). The rim height is measured and the warp redone at it. Then the camera goes right above it and measures again with the whole rim in view. Kept in the state (`bowl`: rim centre + radius) for pouring, drawn in RViz and put in MoveIt's scene. Not seen: step 1 goes on with a warning (set `bowl_look`, or put it in the survey view) |
| (automatic) look again before every contact | a grasp can nudge the stand or the rack, so the task measures again right before it touches anything: step 3 and 4 and 7: tag 8 from above the stand (look-move-look; from above the tag if the cup hides a corner); step 4 and 7: the jar's lid (Hough), and it stops if the lid is more than `checks.seat_tolerance` (10 mm) off the pocket or low (jar tilted / on the rim); step 2 when resumed (not straight after step 1): rack tags 6 and 7, then the jar's lid in the slot, with the same tilt stop; step 8: rack tags 6 and 7 again (wrist turned so the held jar does not hide one), and the slot moves with the rack. The log says how far each moved |
| `spice jog` | move the arm by hand from the terminal (no camera needed): `3 +10` = joint 3 by +10°, `z +2` = hand up 2 cm, `open`/`close`, `home`, `save survey` = write the current pose into `config/real.yaml` |
| `spice bringup` | only robot + camera, e.g. to keep them up between runs; then `NO_BRINGUP=1 spice level1` in another terminal |

(Without `activate.sh`: `real/spice.sh vision` etc. from the repo root.) Extra arguments go to the task:
`spice full --index 1 --home`.

**Speed:** `speed: 0.3` for moves through free space, `speed_contact: 0.1` for the moves onto,
into and out of something (grasps, lifting out, putting in, backing off). **Putting the jar into a
pocket (steps 3, 8):** lowered all the way in (slowly), let go 4 mm above the floor
(`motion.place_drop`; dropping it from the mouth, `release_above_mouth`, tilted it). Then the lid seen in the pocket decides whether it is in (within 5 mm of the axis: in; up to
10 mm: in, logged as a bit off; more: the step stops). Step 7, the grasp the jar is inserted into
the rack with, holds it on the glass body (`jar.insert_grasp_below_top`, 25 mm below the top) so it
cannot pivot in the fingers; step 2 still picks it at the lid (15 mm).

**Runs are automatic: no `[y/N]` before each move. Press any key to PAUSE**: the move in progress
is cancelled, so the arm stops where it is, and nothing new starts. Then **Enter** (or space, `c`)
carries on: the stopped move is planned again from where the arm is. **`q`** stops the run, like
Ctrl-C (which also cancels the move in progress now). After a stop or a failed step it asks
whether to go home safely (`--home`: goes without asking). Letting go of the jar is always asked,
never automatic. `--ask` brings back a `[y/N]` before every move (`spice full --ask`); keep a hand
on the e-stop either way (`helpers/key_pause.py`, test: `python3 real/test/test_key_pause.py`).

The stand pocket (cyan circle in `spice vision`) is **tag 8 + a fixed offset** from the config:
`stand.tag_to_pocket` (58 mm, from the CAD) along the tag's red axis turned by
`stand.pocket_angle_deg` (180: on our stand the red axis points away from the column). `spice vision`
also finds the cup itself (magenta circle) and prints `stand cup measured: tag_to_pocket ...,
pocket_angle_deg ...`. If the cyan circle is off the cup, copy those two numbers into
`config/real.yaml`. The jar checks compare pictures (`captures/<run>/*compare_*.png` shows
empty / with jar / now side by side).

The Kinova driver is looked for in `$KORTEX_WS`, then `underlay_ws/install`, then
`~/kortex_ws/install`.

**Next levels (not done yet):**

- Level 3: several jars in the rack, pick the one that says "salt". Now, with more than one lid
  in view, `--index N` picks lid N from `captures/<time>/*survey_lids.png`; reading the label
  is the next thing to add.
- Level 4: step 5, shake over the bowl (in: the bowl from step 1, `pour:` in `config/real.yaml`).

## Seeing what the robot sees (RViz)

RViz opens by itself with the robot (`config/spice.rviz`; `RVIZ=false spice ...` to skip it):

| In RViz | What it is |
|---|---|
| the arm | live joint states; MoveIt's planned path is shown before each move |
| **Camera 3D (depth cloud)** | the D405's coloured point cloud in 3D, in place on the table |
| **Camera (live)** panel | the wrist camera picture |
| **What the code detected** panel | the last annotated picture: tags, cyan stand pocket, yellow lid circles |
| **MotionPlanning** | drag the orange goal arm (or set angles in its **Joints** tab), then **Plan & Execute**: moves the real arm |
| **camera (hand-eye)** | white D405-sized box + red arrow (its view): where the code thinks the camera is. It must sit on the real camera; if not, `hand_eye` in `config/real.yaml` is not your mount |
| **Scene geometry** (MotionPlanning) | the table (big flat box), the D405 box on the wrist, and once step 1 has measured them the rack (box), the regrasp stand (plate + column) and the bowl (cylinder): what MoveIt plans around (`scene:` in `config/real.yaml`, `helpers/scene.py`; like the sim's `scene_setup.py`) |
| **Spice task** markers | rack (blue box), tags (green), stand pocket (cyan), the salt jar (orange, "salt"), every lid seen (yellow), and the **next move** (green arrow to where the finger pads go, with what it is) |

The green arrow is drawn before the `[y/N]` prompt, so look at it in RViz before answering `y`.
The point cloud is heavy at 1280x720; untick it if RViz gets slow.

## Offline test

Planning without the robot: `FAKE_ROBOT=1 spice bringup` starts MoveIt (with Pilz), the planning
scene and RViz on a simulated arm that follows every command; drag the goal in RViz, or run
`spice jog`-style moves, and see where the arm goes. (Step 4 replayed this way from a real run's
stand position: side grasp from +30 deg, every free move by PTP, nothing below 3 cm.)

No ROS or robot needed: a ray-cast stand-in for the cell (`test/sim_world.py`) runs perception and
the whole step sequence for three rack slots. It moves the hand straight to each pose (no arm, no
MoveIt), so it cannot show collisions of the arm's links: those are MoveIt's job (`scene:`).

```bash
python3 real/test/test_offline.py
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Not ready: no TF base_link -> camera_color_optical_frame` | bringup_robot (hand-eye TF) or the camera is not running |
| `no lid-sized circle at height ...` | see the saved picture; with `--top-z` in `check_vision.py` try the real lid height; check `jar.lid_diameter` |
| `N lids seen; pick one with --index N` | more than one jar in view: pass `--index` |
| `the jar is not in the ... pocket` | the jar caught on the pocket lead-in, or a different light/shadow: look at `*compare_*.png`; check the stand offset with `spice vision` |
| `the rack comes out backwards` / the arm drives off to a wrong spot | normally switched by itself (see *Camera mount*). If it still says backwards after the switch: a tag is stuck the wrong way round, or the rack / stand really faces away from the arm; turn it to face the arm. To force a mount: `hand_eye: *inside` (or `*outside`) in `config/real.yaml` |
| `The robot driver is already running` | another `spice ...` terminal still has the robot up: run with `NO_BRINGUP=1 spice ...`, or Ctrl-C that one first. Never two bringups at once: the second driver takes the arm out of low-level servoing |
| `MoveIt says done, but the hand is N mm ... from the goal` | the arm ignored the move. Almost always: two bringups were running (`WRONG_SERVOING_MODE` in `captures/logs/robot.log`). Stop every spice terminal, then start one. Still there: power-cycle the arm |
| `tag 7: depth on it was off ...; used the height of the other tag(s)` | fine: the rack's two tags lie on one surface, so a tag with bad depth is measured at the other one's height |
| `Joint 'joint_3' ... outside bounds` / MoveIt error -26 / the arm does not go home | the arm is in Kinova's Retract pose (joint 3 at ~161 deg; MoveIt allows 149.5). Home at startup, the task and `spice jog` (`unwind`) now turn it back inside first; more than ~34 deg out: move it by hand (Kinova Web App, joint control) |
| a free move swings the arm somewhere odd | free moves are Pilz PTP (all joints in a straight line to the IK goal nearest the current joints) and only then RRTConnect; both keep the base towards the target and the elbow bent the normal way (`planning.guard`, as the sim's arm client). The log says which one moved (`moved with PTP`) |
| an elbow / wrist touches the table | MoveIt did not know the table: look for `MoveIt scene: table top at z ...` at the start; check `scene.table_top_z` (rack top z - 0.024) |
| `no side grasp around the jar in the stand that the arm can reach` | the stand is too close to the arm base for a side-on hand: move it a few cm further out |
| `tag N seen, but its edge measures X mm` | the printed tag is not the size in `tags.sizes`: measure the black square, set it |
| `fingers closed on nothing` | grasp missed, or `gripper.holding_max` too low for your jar |
| `Straight line only N% feasible` | the pose is near a joint limit: move the rack/stand closer to the arm |
