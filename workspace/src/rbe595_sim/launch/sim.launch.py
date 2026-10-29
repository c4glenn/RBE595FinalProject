import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def spawn_spice_bottle(name, tag_id, x, y, z=0.75):
    """Spawn a spice bottle (models/spice_bottle.sdf.xacro) with its base centre at (x, y, z).

    The default z puts it on the table top.
    """
    sdf = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "spice_bottle.sdf.xacro"]),
            f" name:={name} tag_id:={tag_id}",
        ]
    )
    return Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", sdf, "-x", str(x), "-y", str(y), "-z", str(z)],
    )


def generate_launch_description():
    robot_name = LaunchConfiguration("robot_name")
    gripper = LaunchConfiguration("gripper")
    # Position of the arm base in the world; the default places it on the table top
    # (table top is at z = 0.75 m, see worlds/table_world.sdf).
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")

    # Kortex 7 DoF Gen3 controller configuration
    controllers_file = PathJoinSubstitution(
        [FindPackageShare("kortex_description"), "arms/gen3/7dof/config/ros2_controllers.yaml"]
    )

    robot_description_content = ParameterValue(
        Command(
            [
                PathJoinSubstitution([FindExecutable(name="xacro")]),
                " ",
                PathJoinSubstitution(
                    [FindPackageShare("kortex_description"), "robots", "kinova.urdf.xacro"]
                ),
                " robot_ip:=xxx.yyy.zzz.www",
                " name:=", robot_name,
                " arm:=gen3",
                " dof:=7",
                " vision:=false",
                " prefix:=''",
                " sim_gazebo:=true",
                " simulation_controllers:=", controllers_file,
                " gripper:=", gripper,
            ]
        ),
        value_type=str,
    )

    world = PathJoinSubstitution([FindPackageShare("rbe595_sim"), "worlds", "table_world.sdf"])

    robotiq_share = os.path.join(get_package_prefix("robotiq_description"), "share")

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_name", default_value="gen3"),
            DeclareLaunchArgument(
                "gripper",
                default_value="robotiq_2f_85",
                choices=["robotiq_2f_85", "robotiq_2f_140", ""],
            ),
            DeclareLaunchArgument("x", default_value="-0.4"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.75"),
            AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", robotiq_share),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
                ),
                launch_arguments={
                    "gz_args": [
                        "-r -v 3 ",
                        world,
                        " --physics-engine gz-physics-bullet-featherstone-plugin",
                    ]
                }.items(),
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-string", robot_description_content,
                    "-name", robot_name,
                    "-allow_renaming", "true",
                    "-x", x, "-y", y, "-z", z,
                ],
            ),
            spawn_spice_bottle("spice_bottle_0", tag_id=0, x=0.1, y=0.2),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[
                    {"robot_description": robot_description_content},
                    {"use_sim_time": True},
                ],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["robotiq_gripper_controller", "-c", "/controller_manager"],
            ),
        ]
    )
