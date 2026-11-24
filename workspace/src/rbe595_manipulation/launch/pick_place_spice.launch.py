import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # Same Gen3 lite 6 DoF + gen3_lite_2f description that rbe595_sim spawns in Gazebo.
    moveit_config = (
        MoveItConfigsBuilder(
            "gen3_lite_gen3_lite_2f", package_name="kinova_gen3_lite_moveit_config"
        )
        .robot_description(
            mappings={
                "robot_ip": "xxx.yyy.zzz.www",
                "use_fake_hardware": "false",
                "gripper": "gen3_lite_2f",
                "dof": "6",
                "sim_gazebo": "true",
            }
        )
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": True,
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                # The Gen3 lite joints are all bounded, and the arm can overshoot a limit by a
                # hair as it settles at the end of a move. With fix_start_state left at its
                # default (false) the CheckStartStateBounds request adapter then rejects the
                # next request with "Start state out of bounds" instead of clamping it.
                "ompl.fix_start_state": True,
                # MoveIt waits for the controller in wall-clock time, but the trajectory's
                # duration is in sim time. When Gazebo runs below real time (a loaded CPU) a
                # long move overruns the default 1.2x + 0.5 s allowance and gets aborted
                # mid-way. A real arm runs in real time, so this only matters in sim.
                "trajectory_execution.allowed_execution_duration_scaling": 3.0,
                "trajectory_execution.allowed_goal_duration_margin": 2.0,
            },
        ],
    )

    # RViz with MoveIt's MotionPlanning display: the robot, the planning scene's collision
    # objects (table, rack, jars, bowl) and each planned path. MoveIt itself has no window.
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=[
            "-d",
            os.path.join(
                get_package_share_directory("kinova_gen3_lite_moveit_config"),
                "config",
                "moveit.rviz",
            ),
        ],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Open RViz to watch MoveIt's planning scene and planned paths",
            ),
            DeclareLaunchArgument(
                "shake",
                default_value="false",
                description="Run step 5 (shake the bottle over the bowl)",
            ),
            move_group_node,
            rviz_node,
            Node(
                package="rbe595_manipulation",
                executable="pick_place_spice_action_server",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": True,
                        "shake": ParameterValue(LaunchConfiguration("shake"), value_type=bool),
                    }
                ],
            ),
        ]
    )
