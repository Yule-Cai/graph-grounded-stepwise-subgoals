import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration


def _project_root():
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def generate_launch_description():
    hospital_root = _project_root() / "external_worlds" / "aws-robomaker-hospital-world"
    default_world = hospital_root / "worlds" / "hospital_primitives.world"
    model_paths = [
        str(hospital_root / "models"),
        str(hospital_root / "fuel_models"),
        os.environ.get("GAZEBO_MODEL_PATH", ""),
    ]
    gazebo_model_path = ":".join(path for path in model_paths if path)

    world = LaunchConfiguration("world")

    gzserver_cmd = ExecuteProcess(
        cmd=["gzserver", world, "--verbose"],
        output="both",
    )

    gzclient_cmd = ExecuteProcess(
        cmd=["gzclient", "--verbose"],
        output="both",
    )

    return LaunchDescription(
        [
            SetEnvironmentVariable("GAZEBO_MODEL_PATH", gazebo_model_path),
            DeclareLaunchArgument(
                "world",
                default_value=str(default_world),
                description="AWS RoboMaker Hospital world file.",
            ),
            gzserver_cmd,
            gzclient_cmd,
        ]
    )
