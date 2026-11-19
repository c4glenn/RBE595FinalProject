# RBE595FinalProject

```bash
# Build
cd /rbe595/workspace
colcon build --symlink-install
source install/local_setup.bash
```

```bash
# Terminal 1: simulation
ros2 launch rbe595_sim sim.launch.py
```

```bash
# Terminal 2: MoveIt + pick/place action server
ros2 launch rbe595_manipulation pick_place_spice.launch.py
```

```bash
# Terminal 3: spice detector (wrist camera snapshot + annotated image viewer)
ros2 launch rbe595_manipulation spice_detector.launch.py
```

```bash
# Terminal 4: pick bottle N (number shown in the annotated image)
ros2 action send_goal /pick_place_spice rbe595_manipulation_interfaces/action/PickPlaceSpice "{spice_id: '4'}" --feedback

# Or select a bottle on the detector, then send an empty goal
ros2 param set /spice_detector selected_bottle 4
ros2 action send_goal /pick_place_spice rbe595_manipulation_interfaces/action/PickPlaceSpice "{spice_id: ''}" --feedback

# Retake the snapshot
ros2 service call /spice_detector/snapshot std_srvs/srv/Trigger
```

```bash
# Stop everything
pkill -9 -f "gz sim"; pkill -f ros2
```
