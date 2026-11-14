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
