from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExternalWorld:
    world_id: str
    label: str
    repo: str
    world: str
    model_paths: tuple[str, ...]
    spawn: tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: str = ""


WORLD_REGISTRY: dict[str, ExternalWorld] = {
    "aws_small_house": ExternalWorld(
        world_id="aws_small_house",
        label="AWS RoboMaker Small House",
        repo="aws-robotics/aws-robomaker-small-house-world",
        world="external_worlds/aws-robomaker-small-house-world/worlds/small_house.world",
        model_paths=("external_worlds/aws-robomaker-small-house-world/models",),
        spawn=(0.0, 0.0, 0.0),
        notes="Multi-room furnished residential world.",
    ),
    "aws_small_warehouse": ExternalWorld(
        world_id="aws_small_warehouse",
        label="AWS RoboMaker Small Warehouse",
        repo="aws-robotics/aws-robomaker-small-warehouse-world",
        world="external_worlds/aws-robomaker-small-warehouse-world/worlds/small_warehouse.world",
        model_paths=("external_worlds/aws-robomaker-small-warehouse-world/models",),
        spawn=(0.0, 0.0, 0.0),
        notes="Warehouse/logistics world with shelves and pallets.",
    ),
    "aws_small_warehouse_no_roof": ExternalWorld(
        world_id="aws_small_warehouse_no_roof",
        label="AWS RoboMaker Small Warehouse No Roof",
        repo="aws-robotics/aws-robomaker-small-warehouse-world",
        world="external_worlds/aws-robomaker-small-warehouse-world/worlds/no_roof_small_warehouse.world",
        model_paths=("external_worlds/aws-robomaker-small-warehouse-world/models",),
        spawn=(0.0, 0.0, 0.0),
        notes="Same warehouse with easier top-down visualization.",
    ),
    "aws_hospital": ExternalWorld(
        world_id="aws_hospital",
        label="AWS RoboMaker Hospital",
        repo="aws-robotics/aws-robomaker-hospital-world",
        world="external_worlds/aws-robomaker-hospital-world/worlds/hospital.world",
        model_paths=(
            "external_worlds/aws-robomaker-hospital-world/models",
            "external_worlds/aws-robomaker-hospital-world/fuel_models",
        ),
        spawn=(0.0, -25.0, 1.57),
        notes="Large detailed hospital world. Heavier than primitive maps on macOS.",
    ),
    "aws_hospital_light": ExternalWorld(
        world_id="aws_hospital_light",
        label="AWS RoboMaker Hospital Light",
        repo="aws-robotics/aws-robomaker-hospital-world",
        world="external_worlds/aws-robomaker-hospital-world/worlds/hospital_light.world",
        model_paths=(
            "external_worlds/aws-robomaker-hospital-world/models",
            "external_worlds/aws-robomaker-hospital-world/fuel_models",
        ),
        spawn=(0.0, -25.0, 1.57),
        notes="Reduced hospital variant for faster loading.",
    ),
    "aws_hospital_primitives": ExternalWorld(
        world_id="aws_hospital_primitives",
        label="AWS Hospital Primitive Collision Proxy",
        repo="aws-robotics/aws-robomaker-hospital-world",
        world="external_worlds/aws-robomaker-hospital-world/worlds/hospital_primitives.world",
        model_paths=(
            "external_worlds/aws-robomaker-hospital-world/models",
            "external_worlds/aws-robomaker-hospital-world/fuel_models",
        ),
        spawn=(0.0, -25.0, 1.57),
        notes="Lightweight primitive collision world used for stable Gazebo tests.",
    ),
    "tb3_world": ExternalWorld(
        world_id="tb3_world",
        label="TurtleBot3 World",
        repo="ROBOTIS-GIT/turtlebot3_simulations",
        world="external_worlds/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_world.world",
        model_paths=("external_worlds/turtlebot3_simulations/turtlebot3_gazebo/models",),
        spawn=(-2.0, -0.5, 0.0),
        notes="Classic TurtleBot3 benchmark obstacle world.",
    ),
    "tb3_house": ExternalWorld(
        world_id="tb3_house",
        label="TurtleBot3 House",
        repo="ROBOTIS-GIT/turtlebot3_simulations",
        world="external_worlds/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_house.world",
        model_paths=("external_worlds/turtlebot3_simulations/turtlebot3_gazebo/models",),
        spawn=(-2.0, -0.5, 0.0),
        notes="TurtleBot3 multi-room house world.",
    ),
    "tb3_dqn_stage4": ExternalWorld(
        world_id="tb3_dqn_stage4",
        label="TurtleBot3 DQN Stage 4",
        repo="ROBOTIS-GIT/turtlebot3_simulations",
        world="external_worlds/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage4.world",
        model_paths=("external_worlds/turtlebot3_simulations/turtlebot3_gazebo/models",),
        spawn=(0.0, 0.0, 0.0),
        notes="Community RL navigation benchmark stage.",
    ),
    "tb3_autorace": ExternalWorld(
        world_id="tb3_autorace",
        label="TurtleBot3 Autorace 2020",
        repo="ROBOTIS-GIT/turtlebot3_simulations",
        world="external_worlds/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_autorace_2020.world",
        model_paths=("external_worlds/turtlebot3_simulations/turtlebot3_gazebo/models",),
        spawn=(0.8, -1.747, 0.0),
        notes="Lane/sign navigation world; useful as a visual-navigation stress case.",
    ),
}


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def external_model_paths(root: Path | None = None) -> list[str]:
    root = root or project_root()
    paths: list[str] = []
    for world in WORLD_REGISTRY.values():
        for relative_path in world.model_paths:
            path = root / relative_path
            if path.exists():
                paths.append(str(path))
    return sorted(set(paths))


def gazebo_model_path(root: Path | None = None, include_existing: bool = True) -> str:
    paths = external_model_paths(root)
    if include_existing:
        existing = os.environ.get("GAZEBO_MODEL_PATH", "")
        paths.extend(path for path in existing.split(":") if path)
    return ":".join(dict.fromkeys(paths))


def resolve_world_path(world_id: str, root: Path | None = None) -> Path:
    root = root or project_root()
    if world_id not in WORLD_REGISTRY:
        raise KeyError(f"Unknown external world_id '{world_id}'. Available: {', '.join(WORLD_REGISTRY)}")
    return root / WORLD_REGISTRY[world_id].world


def main() -> None:
    parser = argparse.ArgumentParser(description="List configured open-source Gazebo worlds.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    root = project_root()
    rows = []
    for item in WORLD_REGISTRY.values():
        world_path = root / item.world
        rows.append(
            {
                **asdict(item),
                "world_path": str(world_path),
                "available": world_path.exists(),
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"Project root: {root}")
    for row in rows:
        status = "OK" if row["available"] else "MISSING"
        print(f"{status:7} {row['world_id']:<28} {row['label']}  {row['world_path']}")


if __name__ == "__main__":
    main()
