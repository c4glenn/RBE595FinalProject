# Wrist Twist Shake

Moves the arm over the bowl, then shakes salt out of the (already closed) gripper by twisting `joint_6` back and forth.

## Run

```bash
ros2 run rbe595_manipulation <script_entry_point> [mode]
```

Start arm bringup and MoveIt first. Clear the area and keep the e-stop in reach.

| Mode | Does |
|---|---|
| `goto` | Moves to `BOWL_Z_POS` only. Use to check the position. |
| `shake` (default) | Moves to `BOWL_Z_POS`, then does `ITERATIONS` twists and returns to the start pose. |

## Controls (top of script)

| Setting | Meaning |
|---|---|
| `TWIST_DEG` | Wrist rotation per twist (deg). Sign sets direction. |
| `ITERATIONS` | Number of twists (~2-3 g of salt each). |
| `BOWL_Z_POS` | End-effector `[x, y, z]` over the bowl (world frame, m). |
| `FORWARD_RPY` | Orientation that points the gripper straight forward. |
| `TWIST_TIMING` | how fast the twist motion should be for rotating the wrist once. |
| `JOINT_LIMITS_DEG` | Limits for `joint_6`. Exceeding the real limits can freeze the robot. |

