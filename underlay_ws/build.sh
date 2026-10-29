#!/usr/bin/env bash
# Fetch ros2_kortex's source dependencies, install system deps, and build the underlay.
# Run from anywhere inside the devcontainer.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"

KORTEX=src/ros2_kortex
for repos in \
    "$KORTEX/ros2_kortex.$ROS_DISTRO.repos" \
    "$KORTEX/ros2_kortex-not-released.$ROS_DISTRO.repos" \
    "$KORTEX/simulation.$ROS_DISTRO.repos"; do
  vcs import src --skip-existing --input "$repos"
done

sudo apt-get update
rosdep install --ignore-src --from-paths src -y -r

# Limit parallelism: an unbounded colcon build can freeze or crash the container.
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release --parallel-workers 3 "$@"
