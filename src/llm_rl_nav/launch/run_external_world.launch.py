import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from llm_rl_nav.tools.external_worlds import WORLD_REGISTRY, gazebo_model_path, project_root, resolve_world_path


def _launch_setup(context, *args, **kwargs):
    world_id = LaunchConfiguration("world_id").perform(context)
    custom_world = LaunchConfiguration("world").perform(context)
    spawn_robot = LaunchConfiguration("spawn_robot")
    use_sim_time = LaunchConfiguration("use_sim_time")
    x_arg = LaunchConfiguration("x_pose").perform(context)
    y_arg = LaunchConfiguration("y_pose").perform(context)

    root = project_root()
    world_spec = WORLD_REGISTRY.get(world_id)
    if world_spec is None:
        raise RuntimeError(f"Unknown world_id '{world_id}'. Available: {', '.join(WORLD_REGISTRY)}")

    world_path = Path(custom_world).expanduser() if custom_world else resolve_world_path(world_id, root)
    if not world_path.is_absolute():
        world_path = (root / world_path).resolve()
    if not world_path.exists():
        raise RuntimeError(f"World file not found: {world_path}")

    x_pose = x_arg if x_arg else str(world_spec.spawn[0])
    y_pose = y_arg if y_arg else str(world_spec.spawn[1])

    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_tb3_gazebo = get_package_share_directory("turtlebot3_gazebo")

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")),
        launch_arguments={"world": str(world_path)}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")),
    )
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, "launch", "robot_state_publisher.launch.py")),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
        condition=IfCondition(spawn_robot),
    )
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_tb3_gazebo, "launch", "spawn_turtlebot3.launch.py")),
        launch_arguments={"x_pose": x_pose, "y_pose": y_pose}.items(),
        condition=IfCondition(spawn_robot),
    )

    return [
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", gazebo_model_path(root)),
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world_id",
                default_value="aws_small_house",
                description=f"Open-source world id. Available: {', '.join(WORLD_REGISTRY)}",
            ),
            DeclareLaunchArgument(
                "world",
                default_value="",
                description="Optional explicit world path. If set, this overrides world_id path but keeps model paths.",
            ),
            DeclareLaunchArgument("spawn_robot", default_value="true", description="Spawn TurtleBot3 into the world."),
            DeclareLaunchArgument("use_sim_time", default_value="true", description="Use Gazebo simulation time."),
            DeclareLaunchArgument("x_pose", default_value="", description="Robot x spawn override."),
            DeclareLaunchArgument("y_pose", default_value="", description="Robot y spawn override."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
