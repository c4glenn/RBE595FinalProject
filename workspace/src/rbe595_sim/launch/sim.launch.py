import math
import os
import random
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, Substitution
from launch.actions import (
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
# A standard cereal/soup bowl: ~15 cm across, ~6.5 cm tall, ~5 mm walls
BOWL_RADIUS = 0.075
BOWL_HEIGHT = 0.065
BOWL_WALL = 0.005
# Distance from the arm base [m] the bowl centre is sampled in: near enough for a top-down drop,
# far enough that the arm does not have to fold over its own base.
BOWL_REACH = (0.35, 0.65)
BOWL_RACK_CLEARANCE = 0.05  # rim to rack, room for the gripper fingers
BOWL_EDGE_MARGIN = 0.02     # rim to table edge
# Regrasp stand (models/regrasp_stand.sdf.xacro): its column centred on the staging spot
# (rbe595_manipulation poses.STAGING_XY), 0.5 m from the arm base, turned so its tag pad points
# 90 deg left of the base -> stand direction. The side grasp then comes in from the base side
# across the plate's short edge, and from above the stand the wrist camera (on the gripper's
# outer side) sits right over the tag.
STAND_XY = (0.0, 0.30)
STAND_YAW = math.atan2(STAND_XY[1] - 0.0, STAND_XY[0] - (-0.4)) + math.pi / 2  # base at the x/y defaults
# Bowl rim to the stand column's centre: clears the plate (it reaches 0.10 m from the column).
BOWL_STAND_CLEARANCE = 0.11
# Spice jar mass range [kg]: the real jar is not weighed yet; somewhere between empty and full.
BOTTLE_MASS_RANGE = ("0.15", "0.30")


def load_finger_collision_boxes():
    cfg_file = os.path.join(
        get_package_share_directory("rbe595_sim"), "config", "gen3_lite_2f_finger_collision.yaml"
    )
    with open(cfg_file) as f:
        return yaml.safe_load(f)


def box_collisions(link, boxes):
    return "".join(
        f'<collision name="{link}_box_{i}">'
        f'<origin xyz="{" ".join(map(str, b["xyz"]))}" rpy="0 0 0"/>'
        f'<geometry><box size="{" ".join(map(str, b["size"]))}"/></geometry>'
        "</collision>"
        for i, b in enumerate(boxes)
    )


def grasp_assist_plugins(bottles):
    """One DetachableJoint per bottle, driven by scripts/grasp_assist.py."""
    return "".join(
        "<gazebo>"
        '<plugin filename="gz-sim-detachable-joint-system" '
        'name="gz::sim::systems::DetachableJoint">'
        "<parent_link>end_effector_link</parent_link>"
        f"<child_model>{b}</child_model>"
        "<child_link>link</child_link>"
        f"<attach_topic>/rbe595_sim/grasp_assist/{b}/attach</attach_topic>"
        f"<detach_topic>/rbe595_sim/grasp_assist/{b}/detach</detach_topic>"
        f"<output_topic>/rbe595_sim/grasp_assist/{b}/state</output_topic>"
        "<suppress_child_warning>true</suppress_child_warning>"
        "</plugin>"
        "</gazebo>"
        for b in bottles
    )


def load_wrist_camera_cfg():
    cfg_file = os.path.join(get_package_share_directory("rbe595_sim"), "config", "wrist_camera.yaml")
    with open(cfg_file) as f:
        return yaml.safe_load(f)


def wrist_camera_urdf(cfg):
    """camera_link (RealSense convention: +x out of the lens) fixed to end_effector_link where
    config/wrist_camera.yaml mounts it, with the D405 body behind the lens, and
    camera_color_optical_frame (+z out of the lens, +x right, +y down on the image), the frame
    the images are in. Plus the gz camera sensor, which looks along camera_link +x."""
    xyz = " ".join(map(str, cfg["mount_xyz"]))
    rpy = " ".join(map(str, cfg["mount_rpy"]))
    size = " ".join(map(str, cfg["body_size"]))
    body = f'<origin xyz="{-cfg["body_size"][0] / 2} 0 0" rpy="0 0 0"/><geometry><box size="{size}"/></geometry>'
    return (
        '<link name="camera_link">'
        f'<visual>{body}<material name="d405_grey"><color rgba="0.6 0.6 0.62 1"/></material></visual>'
        f"<collision>{body}</collision>"
        '<inertial><mass value="0.06"/><inertia ixx="1e-5" ixy="0" ixz="0" iyy="1e-5" iyz="0" izz="1e-5"/></inertial>'
        "</link>"
        '<joint name="camera_mount_joint" type="fixed">'
        f'<parent link="end_effector_link"/><child link="camera_link"/><origin xyz="{xyz}" rpy="{rpy}"/>'
        "</joint>"
        '<link name="camera_color_optical_frame"/>'
        '<joint name="camera_color_optical_joint" type="fixed">'
        '<parent link="camera_link"/><child link="camera_color_optical_frame"/>'
        f'<origin xyz="0 0 0" rpy="{-math.pi / 2} 0 {-math.pi / 2}"/>'
        "</joint>"
        '<gazebo reference="camera_link">'
        '<sensor name="wrist_camera" type="camera">'
        f'<topic>{cfg["topic"]}</topic>'
        f'<update_rate>{cfg["update_rate"]}</update_rate>'
        "<always_on>true</always_on>"
        "<camera>"
        f'<horizontal_fov>{cfg["horizontal_fov"]}</horizontal_fov>'
        f'<image><width>{cfg["width"]}</width><height>{cfg["height"]}</height><format>R8G8B8</format></image>'
        "<clip><near>0.02</near><far>3</far></clip>"
        "</camera>"
        "</sensor>"
        "</gazebo>"
    )


class WristCameraDescription(Substitution):
    """The robot URDF with the wrist camera (wrist_camera_urdf()) added, for both
    robot_state_publisher (so the camera frames are on /tf) and Gazebo."""

    def __init__(self, description):
        super().__init__()
        self.description = description

    def perform(self, context):
        urdf = self.description.perform(context)
        return urdf.replace("</robot>", wrist_camera_urdf(load_wrist_camera_cfg()) + "</robot>")


class GazeboRobotDescription(Substitution):
    """The robot URDF as spawned into Gazebo:

    - any gz Sensors system plugin removed: kinova.urdf.xacro always adds one when
      sim_gazebo:=true, but the world already loads one; two in the same server crash the
      renderer;
    - the finger links' mesh collisions replaced by higher-friction boxes
      (config/gen3_lite_2f_finger_collision.yaml), which bullet-featherstone handles at full
      speed while the gripper holds a jar;
    - a DetachableJoint per bottle for the grasp assist (scripts/grasp_assist.py).
    """

    def __init__(self, description, bottles=()):
        super().__init__()
        self.description = description
        self.bottles = list(bottles)

    def perform(self, context):
        urdf = re.sub(
            r"<plugin[^>]*gz::sim::systems::Sensors.*?</plugin>",
            "",
            self.description.perform(context),
            flags=re.DOTALL,
        )
        fingers = load_finger_collision_boxes()
        for link, boxes in fingers["links"].items():
            urdf, n = re.subn(
                rf'(<link name="{link}">.*?)<collision>.*?</collision>(.*?</link>)',
                lambda m: m.group(1) + box_collisions(link, boxes) + m.group(2),
                urdf,
                count=1,
                flags=re.DOTALL,
            )
            if n != 1:
                raise RuntimeError(f"no mesh collision found for {link} in the robot description")
            mu = fingers["friction"]
            urdf = urdf.replace(
                "</robot>",
                f'<gazebo reference="{link}"><mu1>{mu}</mu1><mu2>{mu}</mu2></gazebo></robot>',
            )
        return urdf.replace("</robot>", grasp_assist_plugins(self.bottles) + "</robot>")


def spawn_spice_bottle(name, tag_id, x, y, z=0.75, mass=0.15):
    """Spawn a spice bottle (models/spice_bottle.sdf.xacro) with its base centre at (x, y, z).

    The default z puts it on the table top.
    """
    sdf = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            ' "',
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "spice_bottle.sdf.xacro"]),
            f'" name:={name} tag_id:={tag_id} mass:={mass}',
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
            ' "',
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "spice_rack.sdf.xacro"]),
            f'" name:={name}',
        ]
    )
    return Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", sdf, "-x", str(x), "-y", str(y), "-z", str(z), "-Y", str(yaw)],
    )


def spawn_regrasp_stand(name, x, y, z=0.75, yaw=0.0):
    """Spawn the regrasp stand (models/regrasp_stand.sdf.xacro), its column centred over (x, y) on
    the table at height z, its tag pad along `yaw`."""
    sdf = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            ' "',
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "regrasp_stand.sdf.xacro"]),
            f'" name:={name}',
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
            ' "',
            PathJoinSubstitution([FindPackageShare("rbe595_sim"), "models", "bowl.sdf.xacro"]),
            f'" name:={name} radius:={BOWL_RADIUS} height:={BOWL_HEIGHT} wall:={BOWL_WALL}',
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
    """World (x, y, z) of every slot floor centre, for a rack spawned at (x, y, z, yaw).

    Order matches the rack frame drawing's S0..S(rows*cols-1) numbering: bottom-to-top within a
    column, then left-to-right across columns. Slot layout comes from config/spice_rack.yaml (see
    the formula there).
    """
    cfg = load_spice_rack_cfg()
    rows, cols = cfg["rows"], cfg["cols"]
    pitch_x, pitch_y, col_offset = cfg["pitch_x"], cfg["pitch_y"], cfg["col_offset"]
    c, s = math.cos(yaw), math.sin(yaw)
    slots = []
    for col in range(cols):
        for row in range(rows):
            sx = (col - (cols - 1) / 2) * pitch_x + col_offset
            sy = (row - (rows - 1) / 2) * pitch_y
            slots.append((x + c * sx - s * sy, y + s * sx + c * sy, z + cfg["floor_thickness"]))
    return slots


def random_bowl_position(rng, base_x, base_y, rack_x, rack_y):
    """A random (x, y) for the bowl centre: on the table, within reach of the arm, clear of the rack
    and of the regrasp stand.

    Assumes an unrotated rack (yaw = 0).
    """
    cfg = load_spice_rack_cfg()
    rack_half_x = cfg["len_x"] / 2
    rack_half_y = cfg["len_y"] / 2
    # The rack's hidden floor collision is wider than the rack itself (see spice_rack.yaml);
    # the bowl must not sit on it either.
    pad_half = cfg["floor_collision_size"] / 2
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
        # Bowl rim clear of the hidden collision plate
        pdx = max(abs(bx - rack_x) - pad_half, 0.0)
        pdy = max(abs(by - rack_y) - pad_half, 0.0)
        if math.hypot(pdx, pdy) < BOWL_RADIUS:
            continue
        if math.hypot(bx - STAND_XY[0], by - STAND_XY[1]) - BOWL_RADIUS < BOWL_STAND_CLEARANCE:
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


def spawn_spice_bottles(context, names, slots):
    """One bottle per rack slot, each with its own mass drawn from [bottle_mass_min,
    bottle_mass_max] (or all bottle_mass, if set). Spawned 1 mm above the slot floor so they do
    not start in contact with the rack.

    Cap tag IDs are 0.. in slot order S0.., or, with shuffle_tags:=true, in a random order, so
    which jar is where has to be found from the tags."""
    fixed = LaunchConfiguration("bottle_mass").perform(context)
    low = float(LaunchConfiguration("bottle_mass_min").perform(context))
    high = float(LaunchConfiguration("bottle_mass_max").perform(context))
    seed = LaunchConfiguration("seed").perform(context)
    # Their own streams, so the bowl position for a given seed does not change with them.
    rng = random.Random(f"bottle_mass/{seed}" if seed else None)
    tag_ids = list(range(len(slots)))
    if LaunchConfiguration("shuffle_tags").perform(context) == "true":
        random.Random(f"tag_order/{seed}" if seed else None).shuffle(tag_ids)
    actions = []
    for i, (name, tag_id, (sx, sy, sz)) in enumerate(zip(names, tag_ids, slots)):
        mass = float(fixed) if fixed else rng.uniform(low, high)
        actions.append(LogInfo(msg=f"{name} in slot S{i}: tag {tag_id}, mass {mass * 1000:.0f} g"))
        actions.append(spawn_spice_bottle(name, tag_id=tag_id, x=sx, y=sy, z=sz + 0.001, mass=f"{mass:.4f}"))
    return actions


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

    # Kortex 6 DoF Gen3 Lite controller configuration
    controllers_file = PathJoinSubstitution(
        [FindPackageShare("kortex_description"), "arms/gen3_lite/6dof/config/ros2_controllers.yaml"]
    )

    # Every file path below goes through a command line, so it is quoted: the workspace may sit
    # in a folder whose name has spaces.
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            ' "',
            PathJoinSubstitution(
                [FindPackageShare("kortex_description"), "robots", "kinova.urdf.xacro"]
            ),
            '" robot_ip:=xxx.yyy.zzz.www',
            " name:=", robot_name,
            " arm:=gen3_lite",
            " dof:=6",
            " vision:=false",
            " prefix:=''",
            " sim_gazebo:=true",
            ' simulation_controllers:="', controllers_file, '"',
            " gripper:=", gripper,
            # Start in the real arm's home pose rather than all zeros (straight up).
            ' initial_positions_file:="',
            PathJoinSubstitution(
                [FindPackageShare("rbe595_sim"), "config", "gen3_lite_initial_positions.yaml"]
            ),
            '"',
        ]
    )

    # Plus the D405 wrist camera (config/wrist_camera.yaml), for robot_state_publisher and Gazebo.
    robot_description_with_camera = WristCameraDescription(robot_description_content)

    world = PathJoinSubstitution([FindPackageShare("rbe595_sim"), "worlds", "table_world.sdf"])

    # Rack centred 0.4 m in front of the arm base. Pointing straight down, the Gen3 lite's hand
    # reaches only ~0.52 m from the base at grasp height (MoveIt IK), and less 10 cm higher for
    # the approach above a jar: at 0.5 m the rack's far column (0.56 m out) cannot be picked.
    rack_x, rack_y = 0.0, 0.0
    # A bottle in every slot, tag IDs 0.. matching the S0..S5 slot numbering from
    # spice_rack_slots(); see spawn_spice_bottles().
    slots = spice_rack_slots(rack_x, rack_y)
    bottle_names = [f"spice_bottle_{i}" for i in range(len(slots))]
    spice_bottles = OpaqueFunction(
        function=spawn_spice_bottles, args=[bottle_names, slots]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_name", default_value="gen3_lite"),
            DeclareLaunchArgument(
                "gripper",
                default_value="gen3_lite_2f",
                choices=["gen3_lite_2f", ""],
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
            # Spice jar mass [kg]: each jar random in [bottle_mass_min, bottle_mass_max], or all
            # bottle_mass if given.
            DeclareLaunchArgument("bottle_mass", default_value=""),
            DeclareLaunchArgument("bottle_mass_min", default_value=BOTTLE_MASS_RANGE[0]),
            DeclareLaunchArgument("bottle_mass_max", default_value=BOTTLE_MASS_RANGE[1]),
            # Put the jars (cap tags 0-5) in the slots in a random order (from seed) instead of
            # tag i in slot Si.
            DeclareLaunchArgument("shuffle_tags", default_value="false", choices=["true", "false"]),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]
                ),
                launch_arguments={
                    "gz_args": [
                        "-r -v 3 ",
                        PythonExpression(["'-s --headless-rendering ' if '", headless, "' == 'true' else ''"]),
                        '"', world, '"',
                        " --physics-engine gz-physics-bullet-featherstone-plugin",
                    ]
                }.items(),
            ),
            Node(
                package="ros_gz_sim",
                executable="create",
                output="screen",
                arguments=[
                    "-string", GazeboRobotDescription(robot_description_with_camera, bottle_names),
                    "-name", robot_name,
                    "-allow_renaming", "true",
                    "-x", x, "-y", y, "-z", z,
                ],
            ),
            spawn_spice_rack("spice_rack", x=rack_x, y=rack_y),
            spawn_regrasp_stand("regrasp_stand", x=STAND_XY[0], y=STAND_XY[1], yaw=STAND_YAW),
            spice_bottles,
            OpaqueFunction(function=spawn_random_bowl, args=[rack_x, rack_y]),
            # Sim only: backs the physical grasp with a fixed joint (see scripts/grasp_assist.py).
            Node(
                package="rbe595_sim",
                executable="grasp_assist.py",
                output="screen",
                parameters=[
                    {"robot_name": robot_name, "bottles": bottle_names, "use_sim_time": True}
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="both",
                parameters=[
                    {"robot_description": ParameterValue(robot_description_with_camera, value_type=str)},
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
                    "/wrist_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
                    "/wrist_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
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
                arguments=["gen3_lite_2f_gripper_controller", "-c", "/controller_manager"],
            ),
        ]
    )
