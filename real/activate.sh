# Project shortcut for THIS terminal only (nothing is written to ~/.bashrc):
#
#   source real/activate.sh        (from the repo root; or give the full path)
#
# Afterwards `spice vision`, `spice level1`, `spice full`, ... work from any folder in this
# terminal, and ROS 2 + the Kinova driver are sourced so plain `ros2 ...` commands work too.
# Close the terminal (or run `spice_off`) and it is gone.

if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "Run it with 'source', not directly:  source real/activate.sh"
  exit 1
fi

SPICE_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
export SPICE_DIR

if [ -f /opt/ros/jazzy/setup.bash ]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
# Cyclone on loopback: over Wi-Fi / USB Ethernet the big camera images never arrive (config/cyclonedds.xml)
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://$SPICE_DIR/config/cyclonedds.xml}"
command -v ros2 >/dev/null && ros2 daemon stop >/dev/null 2>&1   # an old daemon has the old DDS settings
for _ws in "${KORTEX_WS:-}" "$(dirname "$SPICE_DIR")/underlay_ws/install" "$HOME/kortex_ws/install"; do
  if [ -n "$_ws" ] && [ -f "$_ws/setup.bash" ] && [ -d "$_ws/kortex_driver" ]; then
    # shellcheck disable=SC1090
    source "$_ws/setup.bash"
    break
  fi
done
unset _ws

spice() { "$SPICE_DIR/spice.sh" "$@"; }
spice_off() { unset -f spice spice_off; unset SPICE_DIR; echo "spice shortcut removed from this terminal"; }

echo "spice ready in this terminal:  spice check | vision | dry | level1 | full | resume 4,6,7,8 | bringup"
