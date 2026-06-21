import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from llm_rl_nav.envs.semantic_world_source import (
    ALL_SEMANTIC_MAP_IDS,
    ensure_semantic_3d_worlds,
    project_root,
    semantic_metadata_path,
    semantic_world_path,
)


def _launch_setup(context, *args, **kwargs):
    map_id = LaunchConfiguration("map_id").perform(context)
    spawn_robot = LaunchConfiguration("spawn_robot")
    use_sim_time = LaunchConfiguration("use_sim_time")
    x_arg = LaunchConfiguration("x_pose").perform(context)
    y_arg = LaunchConfiguration("y_pose").perform(context)

    if map_id not in ALL_SEMANTIC_MAP_IDS:
        raise RuntimeError(f"Unknown map_id '{map_id}'. Available: {', '.join(ALL_SEMANTIC_MAP_IDS)}")

    root = project_root()
    ensure_semantic_3d_worlds(root, map_ids=(map_id,), overwrite=True)
    world_path = semantic_world_path(map_id, root)
    metadata = _read_metadata(semantic_metadata_path(map_id, root))
    x_pose = x_arg if x_arg else str(metadata["spawn"][0])
    y_pose = y_arg if y_arg else str(metadata["spawn"][1])

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
    return [gzserver_cmd, gzclient_cmd, robot_state_publisher_cmd, spawn_turtlebot_cmd]


def _read_metadata(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_id",
                default_value="reference_family_flat",
                description=f"Generated semantic 3D map id. Available: {', '.join(ALL_SEMANTIC_MAP_IDS)}",
            ),
            DeclareLaunchArgument("spawn_robot", default_value="true", description="Spawn TurtleBot3 into the world."),
            DeclareLaunchArgument("use_sim_time", default_value="true", description="Use Gazebo simulation time."),
            DeclareLaunchArgument("x_pose", default_value="", description="Robot x spawn override."),
            DeclareLaunchArgument("y_pose", default_value="", description="Robot y spawn override."),
            OpaqueFunction(function=_launch_setup),
        ]
    )
