# RBE595FinalProject

Spice jar fetch with the Kinova Gen3 lite (2-finger gripper) in Gazebo Harmonic, ROS 2 Jazzy and
MoveIt. The arm finds the jar by its cap AprilTag, picks it out of the rack, regrasps it on the
stand, (shakes it over the bowl), regrasps it again and puts it back in the same rack slot.

Everything the arm grasps or places is aimed at an AprilTag the wrist camera just measured
(tag36h11):

| Tag ID | On | Size | Used for |
|---|---|---|---|
| 0-5 | jar caps | 16 mm | which jar is which, jar top position |
| 6, 7 | rack front ledge | 36 mm | rack position and yaw |
| 8 | regrasp stand plate | 36 mm | stand pocket position and yaw |

## Packages

| Package | What it has |
|---|---|
| `rbe595_sim` | Gazebo world, rack, stand, spice jars, bowl, wrist camera, `sim.launch.py` |
| `rbe595_manipulation` | the pick and place action server, tag detection, the 8 steps (`steps/`) |
| `rbe595_manipulation_interfaces` | `PickPlaceSpice.action` |

The `ros2_kortex` driver (arm, gripper, MoveIt config) is a submodule in `underlay_ws/src`.

## Build (one time)

Open the repo in the devcontainer (`.devcontainer/`), or use your own Ubuntu 24.04 with ROS 2
Jazzy and Gazebo Harmonic. Get the submodule first if you did not clone with `--recursive`:

```bash
git submodule update --init --recursive
```

Build the underlay (kortex and its deps). This one takes a while:

```bash
./underlay_ws/build.sh
```

Then build our packages:

```bash
cd workspace
source ../underlay_ws/install/setup.bash
colcon build --symlink-install
```

With `--symlink-install` you only need to restart the launches after changing a `.py`, `.yaml`,
`.xacro` or mesh file. Rebuild if you change the `.action` file, `setup.py` or `package.xml`.

## Run

Source these in every terminal first (the devcontainer's `.bashrc` already does it):

```bash
source /opt/ros/jazzy/setup.bash
source underlay_ws/install/setup.bash
source workspace/install/setup.bash
```

If someone else runs ROS on the same network, also pick your own `export ROS_DOMAIN_ID=<n>` in
all three terminals.

Terminal A, start Gazebo and leave it running:

```bash
ros2 launch rbe595_sim sim.launch.py seed:=7 shuffle_tags:=true bowl_x:=0.0 bowl_y:=0.5
```

Terminal B, after Gazebo is up, start MoveIt and the action server:

```bash
ros2 launch rbe595_manipulation pick_place_spice.launch.py rviz:=true
```

Terminal C, after RViz shows the arm, send a goal. `spice_id` is the cap tag ID of the jar you
want (`'0'` to `'5'`, `'tag3'` also works):

```bash
ros2 action send_goal --feedback /pick_place_spice rbe595_manipulation_interfaces/action/PickPlaceSpice "{spice_id: '3'}"
```

The feedback prints each step (`1/8 ...` to `8/8 ...`). The result is `success: true` with
message `ok`, or `failed at step <name>: <reason>`.

To run again, Ctrl-C terminals A and B and start both again. The jar and arm stay where the last
run left them otherwise. A traceback from `grasp_assist` on Ctrl-C is fine.

### Launch arguments

`sim.launch.py`:

| Argument | Default | What it does |
|---|---|---|
| `seed` | random | seed for the bowl position and the jar order |
| `shuffle_tags` | `false` | `true` puts the jars (tags 0-5) in the slots in a random order from `seed` |
| `bowl_x`, `bowl_y` | random | bowl centre on the table |
| `bottle_mass` | random | all jars this mass [kg]; else each random in `bottle_mass_min`..`bottle_mass_max` |
| `headless` | `false` | `true` runs Gazebo without the GUI |

`pick_place_spice.launch.py`:

| Argument | Default | What it does |
|---|---|---|
| `rviz` | `false` | open RViz with the planning scene and the planned paths |
| `shake` | `false` | run step 5, shake the jar over the bowl |

## Steps

| Step | What happens | Reference |
|---|---|---|
| 1 find_jar | look at the rack from above, find the jar, aim the hand over its cap (look again until within 1.5 mm) | rack tags 6, 7 and the cap tag |
| 2 grasp_top | top grasp, lift straight out of the slot | cap tag |
| 3 place_on_stand | find the stand, set the jar in its pocket, check it is upright and seated | stand tag 8, then cap tag |
| 4 regrasp_side | side grasp, lift out | cap tag |
| 5 shake_over_bowl | shake over the bowl (off by default) | bowl position |
| 6 return_to_stand | set the jar back in the stand pocket, check it again | stand tag 8, then cap tag |
| 7 regrasp_top | aim on the cap again, top grasp | cap tag |
| 8 return_to_rack | put it back in its slot, check it is upright and seated | cap tag (slot from step 1) |

Any failed check (tag not seen, jar tilted more than 5 deg, not seated, fingers closed on
nothing) stops the goal with the step name in the message.

## Sim only

Some things in the sim are only there because of Gazebo physics, the real arm does not have
them:

- `grasp_assist` attaches the jar to the gripper once the fingers settle on it, and lets go when
  they open.
- The jar has a thin square foot as its bottom collision, a flat cylinder bottom rocks on small
  contacts in bullet-featherstone.
- The rack floor is one wide hidden collision square and the stand column a box, for the same
  reason.

## Real arm notes

- Stick a 36 mm tag36h11 ID 8 on the stand plate, the tag image's right side pointing to the
  column.
- Measure the camera mount and update `CAMERA_IN_EE_XYZ` in `poses.py` and
  `rbe595_sim/config/wrist_camera.yaml`.
- The stand is printed from the parametric CAD; its sizes are in
  `rbe595_sim/config/regrasp_stand.yaml`.

## Run on the real arm

The real-robot pipeline is in `real/` (details: [real/README.md](real/README.md)). It does not
use colcon: one script starts the robot and camera drivers, runs the task, then stops them.

**One-time setup:** install the Kinova driver (`ros2_kortex`) in `$KORTEX_WS`,
`underlay_ws/install` or `~/kortex_ws/install`. Then, **in each new terminal** (nothing is written
to `~/.bashrc`), run:

```bash
source real/activate.sh
```

This sources ROS 2 Jazzy and the driver, and adds the `spice` command. Without it, use
`real/spice.sh <command>` instead.

**Commands:**

| Command | What it does |
|---|---|
| `spice check` | checks everything is installed and plugged in (camera on USB 3, arm reachable) |
| `spice vision` | level 0: live tags and lid detection, no motion |
| `spice dry` | perception and every planned pose, no motion and no gripper |
| `spice level1` | takes the jar out of the rack and puts it back (steps 1, 2, 8) |
| `spice full` | the whole task, steps 1-8 (find, top grasp, stand, side regrasp, shake over the bowl, stand, top regrasp, rack) |
| `spice full --steps 1-4,6-8` | the whole task without the shake |
| `spice resume 7-8` | carries on from the last run's saved state with the steps given |
| `spice home` | gets the arm home safely from where a run stopped (back out, up, home) |
| `spice jog` | moves the arm by hand from the terminal |
| `spice bringup` | only starts the robot and camera (stays in the foreground) |

**Task options** (add them after the command, e.g. `spice full --index 1 --home`):

| Option | Meaning |
|---|---|
| `--steps 1,2,8` / `--steps 1-4` | which steps to run (default `1-8`) |
| `--resume [dir]` | starts from a saved run's state (default: the latest) |
| `--index N` | several lids seen: pick lid N (numbered in the survey picture) |
| `--ask` | asks `[y/N]` before every motion (the default is automatic) |
| `--dry-run` | no motion and no gripper |
| `--show` | pops up each picture as it is taken |
| `--home` | goes home at the end without asking |
| `--go-home` | no steps, just goes home safely (what `spice home` uses) |

**While it runs:** moves are automatic. **Any key pauses**: the move in progress is cancelled and
the arm stops where it is. Then **Enter / space / c** carries on (the move is planned again from
there), and **q** (or Ctrl-C at any time) stops the run. After a stop, `spice home` gets the arm
out safely, and `spice resume <steps>` carries on.

**Environment variables** (put them in front, e.g. `NO_BRINGUP=1 spice full`):

| Variable | Meaning |
|---|---|
| `NO_BRINGUP=1` | the robot and camera already run in another terminal (`spice bringup`) |
| `CAMERA_PROFILE=848x480x30` | another camera mode (default 1280x720 at 15 fps) |
| `RVIZ=false` | no RViz window |
| `FAKE_ROBOT=1` | no arm: MoveIt and RViz with a simulated arm, no camera (`FAKE_ROBOT=1 spice bringup`) |
| `KORTEX_WS=<path>` | where the Kinova driver is installed |

**Settings:** `real/config/real.yaml` (hand-eye mount, speeds `speed` / `speed_contact`, gripper,
jar sizes, bowl, `pour` tilt and shake). Each run saves its pictures and `state.json` in
`real/captures/<time>/` (used by `--resume`), and its log in `real/captures/logs/`.

**Offline tests** (no robot; run after `source real/activate.sh`):

```bash
python3 real/test/test_offline.py
```

```bash
python3 real/test/test_key_pause.py
```
