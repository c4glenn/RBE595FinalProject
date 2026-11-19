"""Spice bottle / bowl detector (Hough circles on a wrist camera snapshot), plus an image
viewer showing the numbered bottles. Start it after the sim and pick_place_spice.launch.py: it
moves the arm to the observe pose for its snapshot.

  ros2 launch rbe595_manipulation spice_detector.launch.py
  ros2 param set /spice_detector selected_bottle 3                 # pick bottle 3
  ros2 service call /spice_detector/snapshot std_srvs/srv/Trigger  # look again
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("selected_bottle", default_value="-1"),
            DeclareLaunchArgument(
                "config", default_value="", description="detection YAML; empty for the default"
            ),
            DeclareLaunchArgument(
                "snapshot_on_start",
                default_value="true",
                description="move to the observe pose and take a snapshot at startup",
            ),
            DeclareLaunchArgument(
                "viewer", default_value="true", description="open rqt_image_view on the result"
            ),
            Node(
                package="rbe595_manipulation",
                executable="spice_detector",
                name="spice_detector",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "config": LaunchConfiguration("config"),
                        "selected_bottle": LaunchConfiguration("selected_bottle"),
                        "snapshot_on_start": LaunchConfiguration("snapshot_on_start"),
                    }
                ],
            ),
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                arguments=["/spice_detector/annotated_image"],
                condition=IfCondition(LaunchConfiguration("viewer")),
            ),
        ]
    )
