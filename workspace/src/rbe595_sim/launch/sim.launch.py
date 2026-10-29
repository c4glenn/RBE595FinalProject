import math
import os
import random
import re

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription, Substitution
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Table top half extents (worlds/table_world.sdf)
TABLE_HALF_X, TABLE_HALF_Y = 0.6, 0.75
BOWL_RADIUS = 0.09
# Distance from the arm base [m] the bowl centre is sampled in: near enough for a top-down drop,
# far enough that the arm does not have to fold over its own base.
BOWL_REACH = (0.35, 0.65)
BOWL_RACK_CLEARANCE = 0.05  # rim to rack, room for the gripper fingers
BOWL_EDGE_MARGIN = 0.02     # rim to table edge


class WithoutSensorsPlugin(Substitution):
    """A URDF/SDF substitution with any gz Sensors system plugin removed.

    kinova.urdf.xacro always adds a Sensors system when sim_gazebo:=true, but the world already
    loads one; two in the same server crash the renderer.
    """

    def __init__(self, description):
        super().__init__()
        self.description = description

    def perform(self, context):
        return re.sub(
            r"<plugin[^>]*gz::sim::systems::Sensors.*?</plugin>",
            "",
            self.description.perform(context),
            flags=re.DOTALL,
        )


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


def spawn_spice_rack(name, x, y, z=0.75, yaw=0.0):
    """Spawn the spice rack (models/spice_rack.sdf.xacro) with its base-plate underside centred at (x, y, z)."""
    sdf = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "spice_rack.sdf.xacro"]),
            f" name:={name}",
        ]
    )
    return Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", sdf, "-x", str(x), "-y", str(y), "-z", str(z), "-Y", str(yaw)],
    )


def spawn_bowl(name, x, y, z=0.75):
    """Spawn a bowl (models/bowl.sdf.xacro) with its floor underside centred at (x, y, z)."""
    sdf = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "bowl.sdf.xacro"]),
            f" name:={name} radius:={BOWL_RADIUS}",
        ]
    )
    return Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", sdf, "-x", str(x), "-y", str(y), "-z", str(z)],
    )


def load_spice_rack_cfg():
    cfg_file = os.path.join(get_package_share_directory("rbe595_sim"), "config", "spice_rack.yaml")
    with open(cfg_file) as f:
        return yaml.safe_load(f)


def spice_rack_slots(x, y, z=0.75, yaw=0.0):
    """World (x, y, z) of every slot floor centre, row-major, for a rack spawned at (x, y, z, yaw).

    Slot layout comes from config/spice_rack.yaml (see the formula there).
    """
    cfg = load_spice_rack_cfg()
    rows, cols, pitch = cfg["rows"], cfg["cols"], cfg["pitch"]
    c, s = math.cos(yaw), math.sin(yaw)
    slots = []
    for row in range(rows):
        for col in range(cols):
            sx = (row - (rows - 1) / 2) * pitch
            sy = (col - (cols - 1) / 2) * pitch
            slots.append((x + c * sx - s * sy, y + s * sx + c * sy, z + cfg["plate_thickness"]))
    return slots


def random_bowl_position(rng, base_x, base_y, rack_x, rack_y):
    """A random (x, y) for the bowl centre: on the table, within reach of the arm, clear of the rack.

    Assumes an unrotated rack (yaw = 0).
    """
    cfg = load_spice_rack_cfg()
    wall_t = cfg["pitch"] - cfg["cell"]
    rack_half_x = (cfg["rows"] * cfg["pitch"] + wall_t) / 2
    rack_half_y = (cfg["cols"] * cfg["pitch"] + wall_t) / 2
    # Bowl centre limits, keeping the whole rim on the table
    max_x = TABLE_HALF_X - BOWL_RADIUS - BOWL_EDGE_MARGIN
    max_y = TABLE_HALF_Y - BOWL_RADIUS - BOWL_EDGE_MARGIN
    for _ in range(10000):
        bx, by = rng.uniform(-max_x, max_x), rng.uniform(-max_y, max_y)
        if not BOWL_REACH[0] <= math.hypot(bx - base_x, by - base_y) <= BOWL_REACH[1]:
            continue
        # Gap between the bowl rim and the rack footprint
        dx = max(abs(bx - rack_x) - rack_half_x, 0.0)
        dy = max(abs(by - rack_y) - rack_half_y, 0.0)
        if math.hypot(dx, dy) - BOWL_RADIUS < BOWL_RACK_CLEARANCE:
            continue
        return bx, by
    raise RuntimeError("no valid bowl position; check the arm base and rack poses")


def spawn_random_bowl(context, rack_x, rack_y):
    base_x = float(LaunchConfiguration("x").perform(context))
    base_y = float(LaunchConfiguration("y").perform(context))
    bowl_x = LaunchConfiguration("bowl_x").perform(context)
    bowl_y = LaunchConfiguration("bowl_y").perform(context)
    seed = LaunchConfiguration("seed").perform(context)
    rng = random.Random(int(seed) if seed else None)
    rx, ry = random_bowl_position(rng, base_x, base_y, rack_x, rack_y)
    bx = float(bowl_x) if bowl_x else rx
    by = float(bowl_y) if bowl_y else ry
    return [
        LogInfo(msg=f"bowl at ({bx:.3f}, {by:.3f})"),
        spawn_bowl("bowl", bx, by),
    ]


def generate_launch_description():
    robot_name = LaunchConfiguration("robot_name")
    gripper = LaunchConfiguration("gripper")
    # Position of the arm base in the world; the default places it on the table top
    # (table top is at z = 0.75 m, see worlds/table_world.sdf).
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    z = LaunchConfiguration("z")
    # Server only, rendering offscreen (sensors still work); for machines with no display.
    headless = LaunchConfiguration("headless")

    # Kortex 7 DoF Gen3 controller configuration
    controllers_file = PathJoinSubstitution(
        [FindPackageShare("kortex_description"), "arms/gen3/7dof/config/ros2_controllers.yaml"]
    )

    robot_description_content = Command(
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
    )

    world = PathJoinSubstitution([FindPackageShare("rbe595_sim"), "worlds", "table_world.sdf"])

    robotiq_share = os.path.join(get_package_prefix("robotiq_description"), "share")

    # Rack centred 0.5 m in front of the arm base: every slot is within 0.65 m of it.
    rack_x, rack_y = 0.1, 0.0
    # A bottle in every slot, tag IDs 0.. in row-major order. Spawned 1 mm above the slot floor
    # so it does not start in contact with the rack.
    spice_bottles = [
        spawn_spice_bottle(f"spice_bottle_{i}", tag_id=i, x=sx, y=sy, z=sz + 0.001)
        for i, (sx, sy, sz) in enumerate(spice_rack_slots(rack_x, rack_y))
    ]

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
            DeclareLaunchArgument("headless", default_value="false", choices=["true", "false"]),
            # Bowl centre on the table; each left empty is randomised (see random_bowl_position()).
            DeclareLaunchArgument("bowl_x", default_value=""),
            DeclareLaunchArgument("bowl_y", default_value=""),
            # Random seed for the bowl position; empty for a different position every launch.
            DeclareLaunchArgument("seed", default_value=""),
            AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", robotiq_share),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
                ),
                launch_arguments={
                    "gz_args": [
                        "-r -v 3 ",
                        PythonExpression(["'-s --headless-rendering ' if '", headless, "' == 'true' else ''"]),
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
                    "-string", WithoutSensorsPlugin(robot_description_content),
                    "-name", robot_name,
                    "-allow_renaming", "true",
                    "-x", x, "-y", y, "-z", z,
                ],
            ),
            spawn_spice_rack("spice_rack", x=rack_x, y=rack_y),
            *spice_bottles,
            OpaqueFunction(function=spawn_random_bowl, args=[rack_x, rack_y]),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[
                    {"robot_description": ParameterValue(robot_description_content, value_type=str)},
                    {"use_sim_time": True},
                ],
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                    "/scene_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/scene_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
                ],
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
