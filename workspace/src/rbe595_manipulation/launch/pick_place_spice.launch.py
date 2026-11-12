from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "gen3", package_name="kinova_gen3_7dof_robotiq_2f_85_moveit_config"
    ).to_moveit_configs()

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "use_sim_time": False,
                "publish_robot_description_semantic": True,
                "allow_trajectory_execution": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                # joint_1/3/5/7 are continuous (unbounded) per the URDF, so their raw
                # /joint_states value keeps accumulating past +/-pi as the arm turns through
                # several moves. The OMPL pipeline's CheckStartStateBounds request adapter
                # (default_planning_request_adapters/CheckStartStateBounds, params namespace
                # "ompl") then needs to normalize that value back into [-pi, pi] before
                # checking bounds -- and with fix_start_state left at its default (false), it
                # REJECTS the request outright instead of just doing that normalization,
                # raising "Start state out of bounds" (START_STATE_INVALID) even though the
                # state is not actually violating any real joint limit. Confirmed against
                # moveit_ros_planning's check_start_state_bounds.cpp source and this node's
                # own `ros2 param get /move_group ompl.fix_start_state` (defaulted to False).
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
                parameters=[{"use_sim_time": False}],
            ),
        ]
    )
