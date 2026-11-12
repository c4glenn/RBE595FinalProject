from launch import LaunchDescription
from launch_ros.actions import Node
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
            },
        ],
    )

    return LaunchDescription(
        [
            move_group_node,
            Node(
                package="rbe595_manipulation",
                executable="pick_place_spice_action_server",
                output="screen",
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
