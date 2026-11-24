# Kinova driver + MoveIt + hand-eye TF + RViz (the spice layout: robot, camera image, 3D depth cloud,
# and what the task detected)
#   ros2 launch real/launch/bringup_robot.launch.py              (copied from HW7 rbe_vbm_test)
#   ros2 launch real/launch/bringup_robot.launch.py rviz:=false  (no RViz)


import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def spice_rviz(robot_ip):
    """RViz with config/spice.rviz and the MoveIt parameters its MotionPlanning display needs (the
    same ones kinova_gen3_lite_moveit_config's robot.launch.py gives its own RViz)."""
    from moveit_configs_utils import MoveItConfigsBuilder
    mappings = {'robot_ip': robot_ip, 'use_fake_hardware': 'false', 'gripper': 'gen3_lite_2f',
                'gripper_joint_name': 'right_finger_bottom_joint', 'dof': '6',
                'gripper_max_velocity': '100.0', 'gripper_max_force': '100.0'}
    moveit_config = (MoveItConfigsBuilder('gen3_lite_gen3_lite_2f', package_name='kinova_gen3_lite_moveit_config')
                     .robot_description(mappings=mappings)
                     .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
                     .to_moveit_configs())
    return Node(package='rviz2', executable='rviz2', name='rviz2_spice', output='log',
                arguments=['-d', os.path.join(HERE, '..', 'config', 'spice.rviz')],
                parameters=[moveit_config.robot_description, moveit_config.robot_description_semantic,
                            moveit_config.robot_description_kinematics, moveit_config.planning_pipelines,
                            moveit_config.joint_limits])


def launch_setup(context, *args, **kwargs):
    config_path = os.path.expanduser(LaunchConfiguration('config').perform(context))
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    robot_ip = LaunchConfiguration('robot_ip').perform(context) or str(cfg['robot_ip'])
    go_home = LaunchConfiguration('go_home').perform(context).lower() in ('true', '1', 'yes')
    rviz = LaunchConfiguration('rviz').perform(context).lower() in ('true', '1', 'yes')

    actions = [
        IncludeLaunchDescription(
            # kinova_gen3_lite_moveit_config's robot.launch.py, + the Pilz planner (launch/kinova_robot.launch.py)
            PythonLaunchDescriptionSource(os.path.join(HERE, 'kinova_robot.launch.py')),
            # its own RViz (plain MoveIt layout) off: ours below shows the camera too
            launch_arguments={'robot_ip': robot_ip, 'launch_rviz': 'false',
                              'use_fake_hardware': LaunchConfiguration('use_fake_hardware')}.items(),
        ),
        # hand-eye TF (end_effector_link -> camera_link); switchable to the other camera mount
        # while running (the task does that if the rack tags show the camera is on the other side)
        ExecuteProcess(cmd=['python3', os.path.join(HERE, '..', 'helpers', 'hand_eye_tf.py'), config_path],
                       name='hand_eye_tf', output='screen'),
    ]
    if rviz:
        actions.append(TimerAction(period=5.0, actions=[spice_rviz(robot_ip)]))
    if go_home:
        # go_home.py
        actions.append(TimerAction(period=8.0, actions=[ExecuteProcess(
            cmd=['python3', os.path.join(HERE, '..', 'helpers', 'go_home.py'), config_path],
            name='go_home', output='screen')]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=os.path.join(HERE, '..', 'config', 'real.yaml'),
                              description='Per-robot YAML with robot_ip, hand_eye, home, views, workspace'),
        DeclareLaunchArgument('robot_ip', default_value='', description='Overrides robot_ip from the YAML'),
        DeclareLaunchArgument('go_home', default_value='true', description='Move to home once MoveIt is ready'),
        DeclareLaunchArgument('rviz', default_value='true', description='Open RViz with the spice layout'),
        DeclareLaunchArgument('use_fake_hardware', default_value='false',
                              description='No robot: a simulated arm that follows every command (test MoveIt at home)'),
        OpaqueFunction(function=launch_setup),
    ])
