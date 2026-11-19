#!/usr/bin/env bash
# Fetch ros2_kortex's source dependencies, install system deps, and build the underlay.
# Run from anywhere inside the devcontainer.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
# ROS setup scripts reference unset variables, so relax -u while sourcing.
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

KORTEX=src/ros2_kortex
for repos in \
    "$KORTEX/ros2_kortex.$ROS_DISTRO.repos" \
    "$KORTEX/ros2_kortex-not-released.$ROS_DISTRO.repos" \
    "$KORTEX/simulation.$ROS_DISTRO.repos"; do
  vcs import src --skip-existing --input "$repos"
done

sudo apt-get update
rosdep install --ignore-src --from-paths src -y -r

# Limit parallelism: an unbounded build runs out of memory. Peak load is roughly
# PARALLEL_WORKERS x BUILD_JOBS compiler processes, each up to ~1.5 GB on the ros2_control
# packages; the defaults stay within ~4 GB. Raise them on a machine with more RAM.
PARALLEL_WORKERS=${PARALLEL_WORKERS:-1}
BUILD_JOBS=${BUILD_JOBS:-2}
export MAKEFLAGS="-j${BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${BUILD_JOBS}"

# By default build only what workspace/ uses (and their dependencies), not all ~75 packages the
# .repos files pull in. Any arguments replace this selection and are passed to colcon as-is,
# e.g. `./build.sh --packages-select kortex_description`.
if [ $# -eq 0 ]; then
  set -- --packages-up-to \
    kortex_description kinova_gen3_lite_moveit_config robotiq_description \
    controller_manager joint_state_broadcaster joint_trajectory_controller gripper_controllers \
    gz_ros2_control ros_gz_sim ros_gz_bridge
fi

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  --parallel-workers "$PARALLEL_WORKERS" "$@"
