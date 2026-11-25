#!/usr/bin/env bash
# One command for the real spice task: sources ROS + the Kinova driver, starts the robot and the
# camera in the background, waits until both are up, runs what you ask for, and stops them again
# when it ends (or on Ctrl-C).
#
#   real/spice.sh check            is everything installed / plugged in?
#   real/spice.sh vision           level 0: live tags + lid picture, no motion
#   real/spice.sh dry              perception + every planned pose, no motion
#   real/spice.sh level1           pick the salt jar out of the rack and put it back (steps 1,2,8)
#   real/spice.sh full             the whole task, steps 1-8 (--steps 1-4,6-8 skips the shake)
#   real/spice.sh pourtest         the whole task, but step 5 tips slowly: press a key when it pours (angle ->
#                                  real/data/pour_tests.csv). After a failure: spice resume 5,6,7,8 --pour-test
#   real/spice.sh resume 4,6,7,8   carry on from the last run's saved state
#   real/spice.sh home             get the arm home safely from where a run stopped (back out, up, home)
#   real/spice.sh jog              move the arm by hand from the terminal (joints, x/y/z, save poses)
#   real/spice.sh bringup          only start robot + camera (stays in the foreground)
#
# Shortcut for this terminal only (nothing written to ~/.bashrc):  source real/activate.sh
# then just: spice vision, spice level1, ...
# Extra arguments go to the task, e.g.  real/spice.sh full --spice pepper --home
# Which jar: the one whose paper label reads --spice (default salt); --index N = lid N instead.
#
# Robot/camera already running in another terminal? Add NO_BRINGUP=1 in front (spice refuses to
# start a second driver: two drivers on one arm leave it not moving at all).
# Camera: 1280x720 at 15 fps; another: CAMERA_PROFILE=848x480x30 spice ...
# No robot (at home): FAKE_ROBOT=1 spice bringup  -- MoveIt + RViz with a simulated arm, no camera.
# RViz (robot, camera, 3D cloud, detections) opens with the robot; RVIZ=false in front to skip it.
# Where the Kinova driver is installed: $KORTEX_WS, else <repo>/underlay_ws/install,
# else ~/kortex_ws/install.

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(dirname "$HERE")"
LOG_DIR="$HERE/captures/logs"
cmd="${1:-help}"
shift || true

red() { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
say() { printf '\033[36m[spice]\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------- environment
find_kortex() {
  for ws in "${KORTEX_WS:-}" "$REPO/underlay_ws/install" "$HOME/kortex_ws/install"; do
    [ -n "$ws" ] && [ -f "$ws/setup.bash" ] && [ -d "$ws/kortex_driver" ] && { echo "$ws"; return 0; }
  done
  return 1
}

source_ros() {
  if [ ! -f /opt/ros/jazzy/setup.bash ]; then
    red "ROS 2 Jazzy not found (/opt/ros/jazzy). Run real/install.sh first."; exit 1
  fi
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  # Cyclone on loopback: over Wi-Fi / USB Ethernet the big camera images never arrive (config/cyclonedds.xml)
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$HERE/config/cyclonedds.xml}"
  # The ros2 CLI daemon keeps the DDS settings it was started with: one left over from a terminal
  # without the line above sees none of our nodes, so `ros2 topic list` waits forever. Fresh one.
  ros2 daemon stop >/dev/null 2>&1 || true
  if KWS="$(find_kortex)"; then
    # shellcheck disable=SC1090
    source "$KWS/setup.bash"
  else
    red "Kinova driver (kortex_driver) not found. Run real/install.sh, or point KORTEX_WS at its install/ folder."
    exit 1
  fi
}

check() {
  local ok=0
  source_ros
  green "ROS 2 Jazzy: ok"
  green "Kinova driver: $KWS"
  for pkg in kinova_gen3_lite_moveit_config kortex_driver realsense2_camera; do
    if ros2 pkg prefix "$pkg" >/dev/null 2>&1; then green "  $pkg: ok"; else red "  $pkg: MISSING"; ok=1; fi
  done
  if python3 -c "import cv2, numpy, scipy, yaml; cv2.aruco.DICT_APRILTAG_36h11" 2>/dev/null; then
    green "Python (opencv with aruco, numpy, scipy, yaml): ok"
  else
    red "Python packages missing: sudo apt install python3-opencv python3-scipy python3-yaml"; ok=1
  fi
  ip=$(python3 -c "import yaml; print(yaml.safe_load(open('$HERE/config/real.yaml'))['robot_ip'])")
  if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then green "Robot at $ip: reachable"
  else red "Robot at $ip: no answer (cable, power, your PC's IP on that subnet?)"; ok=1; fi
  if command -v rs-enumerate-devices >/dev/null && rs-enumerate-devices -s 2>/dev/null | grep -q D4; then
    green "RealSense: $(rs-enumerate-devices -s 2>/dev/null | grep D4 | head -1 | awk '{$1=$1};1')"
  elif lsusb 2>/dev/null | grep -qi 'Intel.*RealSense\|8086:0b5b'; then green "RealSense: on USB"
  else red "RealSense: not found on USB"; ok=1; fi
  [ $ok -eq 0 ] && green "All good." || red "Fix the red lines above."
  return $ok
}

# ---------------------------------------------------------------- bringup
PIDS=()
stop_bringup() {
  [ ${#PIDS[@]} -eq 0 ] && return
  say "stopping robot + camera ..."
  for pid in "${PIDS[@]}"; do kill -INT -- "-$pid" 2>/dev/null; done
  sleep 3
  for pid in "${PIDS[@]}"; do kill -KILL -- "-$pid" 2>/dev/null; done
  PIDS=()
}

wait_for() {   # wait_for <what> <seconds> <command...>
  local what=$1 secs=$2; shift 2
  local end=$((SECONDS + secs))
  until "$@" >/dev/null 2>&1; do
    if [ $SECONDS -ge $end ]; then red "$what did not come up in ${secs}s; see $LOG_DIR"; return 1; fi
    sleep 2
  done
  green "$what: up"
}

# The camera is up when a colour image actually arrives (the topic names exist before the driver
# has streamed anything). realsense2_camera_node sometimes segfaults right after opening the
# streams ("process has died ... exit code -11" in camera.log): then it is started again.
start_camera() {   # start_camera <profile>
  local profile=$1 try pid end
  : >"$LOG_DIR/camera.log"
  for try in 1 2 3; do
    say "starting the camera (log: $LOG_DIR/camera.log)$([ "$try" -gt 1 ] && echo ", try $try of 3")"
    echo "===== spice.sh: camera start, try $try =====" >>"$LOG_DIR/camera.log"
    setsid ros2 launch "$HERE/launch/bringup_camera.launch.py" profile:="$profile" >>"$LOG_DIR/camera.log" 2>&1 &
    pid=$!
    PIDS+=("$pid")
    end=$((SECONDS + 45))
    while [ $SECONDS -lt $end ]; do
      if timeout 5 ros2 topic echo --once --field header.frame_id /camera/camera/color/image_raw >/dev/null 2>&1; then
        green "camera: up (colour images arriving)"
        return 0
      fi
      if ! kill -0 "$pid" 2>/dev/null || tail -n 5 "$LOG_DIR/camera.log" | grep -q 'process has died'; then
        red "the camera driver crashed while starting (see camera.log); starting it again"
        break
      fi
      sleep 1
    done
    kill -INT -- "-$pid" 2>/dev/null; sleep 2; kill -KILL -- "-$pid" 2>/dev/null
    local keep=() p; for p in "${PIDS[@]}"; do [ "$p" != "$pid" ] && keep+=("$p"); done; PIDS=("${keep[@]}")
  done
  red "no colour image from the camera after 3 tries; see $LOG_DIR/camera.log"
  red "  (unplug/replug the D405 straight into the laptop, not through the dock)"
  return 1
}

robot_running() { pgrep -f '^[^ ]*/lib/controller_manager/ros2_control_node' >/dev/null; }

start_bringup() {
  if [ "${NO_BRINGUP:-0}" = 1 ]; then
    if ! robot_running; then
      red "NO_BRINGUP=1, but no robot bringup is running. Start one in another terminal (spice bringup),"
      red "or leave out NO_BRINGUP=1."; exit 1
    fi
    say "NO_BRINGUP=1: using the robot + camera already running"; return
  fi
  if robot_running; then
    red "The robot driver is already running (another 'spice ...' or 'spice bringup' terminal)."
    red "A second driver on the arm knocks it out of low-level servoing: MoveIt then reports every"
    red "move as done while the arm does not move at all. Either:"
    red "  * use the running one:   NO_BRINGUP=1 spice $cmd"
    red "  * or stop the other terminal (Ctrl-C there; wait until it has exited), then run this again."
    exit 1
  fi
  mkdir -p "$LOG_DIR"
  [ "${FAKE_ROBOT:-0}" = 1 ] && NO_CAMERA=1
  if [ "${NO_CAMERA:-0}" != 1 ]; then check_camera_usb; fi
  local profile="${CAMERA_PROFILE:-1280x720x15}"
  trap 'stop_bringup; exit 130' INT TERM
  trap stop_bringup EXIT
  say "starting the robot + MoveIt (log: $LOG_DIR/robot.log); the arm goes HOME when MoveIt is up"
  local fake=()
  if [ "${FAKE_ROBOT:-0}" = 1 ]; then
    say "FAKE_ROBOT=1: no real arm, a simulated one that follows every command (MoveIt, Pilz, RViz as usual)"
    fake=(use_fake_hardware:=true robot_ip:=0.0.0.0)
  fi
  setsid ros2 launch "$HERE/launch/bringup_robot.launch.py" rviz:="${RVIZ:-true}" "${fake[@]}" >"$LOG_DIR/robot.log" 2>&1 &
  PIDS+=($!)
  if [ "${NO_CAMERA:-0}" != 1 ]; then start_camera "$profile" || exit 1; fi
  wait_for "MoveIt" 120 bash -c "ros2 action list | grep -q '^/move_action$'" || exit 1
  wait_for "robot driver" 60 bash -c "ros2 action list | grep -q gripper_cmd" || exit 1
  say "waiting for the arm to finish going home ..."
  sleep 12
}

check_camera_usb() {
  if ! lsusb 2>/dev/null | grep -qi '8086:0b5b\|RealSense'; then
    red "The D405 is not on USB (lsusb shows no Intel RealSense): the driver cannot find it."
    red "  Unplug and replug its USB-C cable at BOTH ends (camera and PC), use a USB 3 port, then"
    red "  check with:  lsusb | grep 8086    (expect: Intel ... 8086:0b5b). Then run this again."
    red "  Keeps dropping out? The wrist cable is strained/loose: secure it so the arm cannot tug it."
    exit 1
  fi
  # Link speed of the D405 itself (Mbit/s), not of every uvcvideo device: the laptop's
  # built-in webcam is always 480M and must not trip this check.
  local d405_speed=""
  for d in /sys/bus/usb/devices/*; do
    if [ "$(cat "$d/idVendor" 2>/dev/null)" = 8086 ] && [ "$(cat "$d/idProduct" 2>/dev/null)" = 0b5b ]; then
      d405_speed=$(cat "$d/speed" 2>/dev/null); break
    fi
  done
  if [ -n "$d405_speed" ] && [ "${d405_speed%%.*}" -lt 5000 ] && [ -z "${CAMERA_PROFILE:-}" ]; then
    red "The D405 is on a USB 2 link (480M). On this PC its frames then arrive cut off (even at"
    red "640x480, 5 fps), so the code never gets a picture. It needs USB 3 (5000M):"
    red "  * a USB 3 cable: 'SS' logo / blue tongue in the USB-A end (many USB-C cables are USB 2 only);"
    red "  * a USB 3 port on the PC (blue), not a hub / keyboard / monitor port;"
    red "  * the plug pushed fully into the camera (half-seated USB-C falls back to USB 2)."
    red "  Check:  lsusb -t | grep uvc   ->  must say 5000M.   (Force a try anyway: CAMERA_PROFILE=640x480x5 spice vision)"
    exit 1
  fi
}

task() { python3 "$HERE/run_task.py" "$@"; }

# ---------------------------------------------------------------- commands
case "$cmd" in
  check)   check ;;
  vision)  source_ros; start_bringup; python3 "$HERE/tools/check_vision.py" "$@" ;;
  dry)     source_ros; start_bringup; task --dry-run "$@" ;;
  level1)  source_ros; start_bringup; task --steps 1,2,8 "$@" ;;
  full)    source_ros; start_bringup; task "$@" ;;
  pourtest) source_ros; start_bringup; task --steps 1-8 --pour-test "$@" ;;
  resume)  source_ros; start_bringup; steps="${1:?give the steps, e.g. real/spice.sh resume 4,6,7,8}"; shift
           task --steps "$steps" --resume "$@" ;;
  home)    source_ros; start_bringup; task --go-home --resume "$@" ;;
  jog)     source_ros; NO_CAMERA=1 start_bringup; python3 "$HERE/tools/jog.py" ;;
  bringup) source_ros; start_bringup; say "robot + camera running; Ctrl-C to stop"; wait ;;
  alias)   say "Per-terminal shortcut, nothing saved:  source $HERE/activate.sh" ;;
  *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0" ;;
esac
