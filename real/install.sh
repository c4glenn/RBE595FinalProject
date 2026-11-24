#!/usr/bin/env bash
# One-time setup on a new computer (Ubuntu 24.04 with ROS 2 Jazzy installed from apt):
#   * apt: MoveIt, the RealSense driver, OpenCV/scipy/yaml for Python
#   * the Kinova driver (ros2_kortex, not on apt for Jazzy) built from the repo's submodule
#     into <repo>/underlay_ws/install (takes a while)
#
#   real/install.sh
#
# Already have ros2_kortex built somewhere else? Skip this and set KORTEX_WS=/path/to/install.
set -eo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(dirname "$HERE")"

if [ ! -f /opt/ros/jazzy/setup.bash ]; then
  echo "Install ROS 2 Jazzy first: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html"
  exit 1
fi

sudo apt-get update
sudo apt-get install -y ros-jazzy-moveit ros-jazzy-realsense2-camera ros-jazzy-kortex-description \
  python3-opencv python3-scipy python3-yaml python3-vcstool python3-rosdep python3-colcon-common-extensions
sudo rosdep init 2>/dev/null || true
rosdep update

cd "$REPO"
git submodule update --init --recursive underlay_ws/src/ros2_kortex
"$REPO/underlay_ws/build.sh"

echo
echo "Done. Check with:  real/spice.sh check"
echo "Shortcut (this terminal only):  source real/activate.sh   then: spice vision, spice level1, ..."
