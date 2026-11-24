# Starts RealSense D405
#   ros2 launch real/launch/bringup_camera.launch.py   (copied from HW7 rbe_vbm_test)
#   ... profile:=640x480x5   lighter stream for a USB 2 link (spice.sh sets it by itself then);
#   default 1280x720x15: big enough for the tags from the survey pose (the D405's own default,
#   848x480, makes them ~25 px wide and their size check flaky)

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

FILTER_ARGS = ['spatial_filter.enable', 'temporal_filter.enable',
              'decimation_filter.enable', 'hole_filling_filter.enable']


def generate_launch_description():
    return LaunchDescription([
        *(DeclareLaunchArgument(name, default_value='false') for name in FILTER_ARGS),
        DeclareLaunchArgument('profile', default_value='1280x720x15',
                              description='WIDTHxHEIGHTxFPS for colour and depth (D405: both from the depth module)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
            launch_arguments={'pointcloud.enable': 'true', 'align_depth.enable': 'true',
                              'depth_module.depth_profile': LaunchConfiguration('profile'),
                              'depth_module.color_profile': LaunchConfiguration('profile'),
                              **{name: LaunchConfiguration(name) for name in FILTER_ARGS}}.items(),
        ),
    ])
