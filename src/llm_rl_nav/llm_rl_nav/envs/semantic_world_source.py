from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from xml.sax.saxutils import escape

from llm_rl_nav.envs.hospital_2d_env import (
    ALL_SEMANTIC_MAP_IDS,
    Hospital2DNavEnv,
    RectObstacle,
    SemanticMapSpec,
    semantic_map_specs,
)


MAP_SOURCE_SEMANTIC_2D = "semantic_2d"
MAP_SOURCE_GAZEBO_3D = "gazebo_3d_projection"
MAP_SOURCE_CHOICES = (MAP_SOURCE_SEMANTIC_2D, MAP_SOURCE_GAZEBO_3D)


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def generated_world_dir(root: Path | None = None) -> Path:
    return (root or project_root()) / "generated_worlds" / "semantic_3d"


def semantic_world_path(map_id: str, root: Path | None = None) -> Path:
    return generated_world_dir(root) / f"{map_id}.world"


def semantic_metadata_path(map_id: str, root: Path | None = None) -> Path:
    return generated_world_dir(root) / f"{map_id}.json"


def ensure_semantic_3d_worlds(
    root: Path | None = None,
    map_ids: tuple[str, ...] = ALL_SEMANTIC_MAP_IDS,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Materialize Gazebo SDF worlds for the training semantic map set.

    The generated SDF files are the geometry source used by the 3D projection
    environment. The 2D view is only a navigation projection of these collision
    boxes.
    """

    root = root or project_root()
    out_dir = generated_world_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = semantic_map_specs()
    paths: list[Path] = []
    for map_id in map_ids:
        if map_id not in specs:
            raise KeyError(f"Unknown semantic map id: {map_id}")
        spec = specs[map_id]
        world_path = semantic_world_path(map_id, root)
        metadata_path = semantic_metadata_path(map_id, root)
        needs_write = overwrite or not world_path.exists() or not metadata_path.exists()
        if not needs_write:
            try:
                _read_json_with_retry(metadata_path, attempts=1)
            except (OSError, json.JSONDecodeError):
                needs_write = True
        if needs_write:
            _atomic_write_text(world_path, _render_world(spec))
            _write_metadata(spec, world_path, metadata_path)
        paths.append(world_path)
    return tuple(paths)


def semantic_3d_map_specs(
    root: Path | None = None,
    map_ids: tuple[str, ...] = ALL_SEMANTIC_MAP_IDS,
) -> dict[str, SemanticMapSpec]:
    root = root or project_root()
    ensure_semantic_3d_worlds(root, map_ids=map_ids, overwrite=False)
    specs: dict[str, SemanticMapSpec] = {}
    for map_id in map_ids:
        metadata_path = semantic_metadata_path(map_id, root)
        world_path = semantic_world_path(map_id, root)
        metadata = _read_json_with_retry(metadata_path)
        obstacles = _parse_world_collision_rects(world_path)
        specs[map_id] = SemanticMapSpec(
            map_id=map_id,
            label=f"{metadata['label']} (Gazebo 3D source projection)",
            spawn=tuple(metadata["spawn"]),
            obstacles=tuple(obstacles),
            goal_points=tuple(tuple(point) for point in metadata["goal_points"]),
            x_bounds=tuple(metadata["x_bounds"]),
            y_bounds=tuple(metadata["y_bounds"]),
        )
    return specs


class Gazebo3DProjectionNavEnv(Hospital2DNavEnv):
    """Fast RL env whose geometry is loaded from generated Gazebo SDF worlds."""

    def __init__(self, *args, project_root_path: Path | None = None, **kwargs):
        map_ids = tuple(kwargs.get("map_ids") or (kwargs.get("map_id", "tb3_house"),))
        root = project_root_path or project_root()
        map_specs = semantic_3d_map_specs(root, map_ids=tuple(map_ids))
        kwargs["map_specs_override"] = map_specs
        super().__init__(*args, **kwargs)
        self.map_source = MAP_SOURCE_GAZEBO_3D


def build_nav_env(
    map_source: str,
    *,
    seed: int,
    map_ids: tuple[str, ...] | None = None,
    map_id: str | None = None,
    **kwargs,
):
    if map_source == MAP_SOURCE_GAZEBO_3D:
        if map_ids is not None:
            kwargs["map_ids"] = map_ids
        if map_id is not None:
            kwargs["map_id"] = map_id
        return Gazebo3DProjectionNavEnv(seed=seed, **kwargs)
    if map_source == MAP_SOURCE_SEMANTIC_2D:
        if map_ids is not None:
            kwargs["map_ids"] = map_ids
        if map_id is not None:
            kwargs["map_id"] = map_id
        return Hospital2DNavEnv(seed=seed, **kwargs)
    raise ValueError(f"Unknown map_source: {map_source}")


def _render_world(spec: SemanticMapSpec) -> str:
    width = spec.x_bounds[1] - spec.x_bounds[0]
    height = spec.y_bounds[1] - spec.y_bounds[0]
    floor_x = (spec.x_bounds[0] + spec.x_bounds[1]) / 2.0
    floor_y = (spec.y_bounds[0] + spec.y_bounds[1]) / 2.0
    models = [
        _render_floor(spec.map_id, floor_x, floor_y, width, height),
        *(_render_box(obstacle) for obstacle in spec.obstacles),
    ]
    return "\n".join(
        [
            '<?xml version="1.0" ?>',
            '<sdf version="1.6">',
            f'  <world name="{escape(spec.map_id)}">',
            "    <gravity>0 0 -9.8</gravity>",
            "    <scene>",
            "      <ambient>0.65 0.65 0.65 1</ambient>",
            "      <background>0.78 0.84 0.88 1</background>",
            "    </scene>",
            "    <include>",
            "      <uri>model://sun</uri>",
            "    </include>",
            *models,
            "  </world>",
            "</sdf>",
            "",
        ]
    )


def _render_floor(map_id: str, x: float, y: float, width: float, height: float) -> str:
    return "\n".join(
        [
            f'    <model name="{escape(_safe_model_name(map_id + "_floor"))}">',
            "      <static>true</static>",
            f"      <pose>{x:.3f} {y:.3f} -0.025 0 0 0</pose>",
            "      <link name=\"floor_link\">",
            "        <collision name=\"floor_collision\">",
            "          <geometry>",
            f"            <box><size>{width:.3f} {height:.3f} 0.050</size></box>",
            "          </geometry>",
            "        </collision>",
            "        <visual name=\"floor_visual\">",
            "          <geometry>",
            f"            <box><size>{width:.3f} {height:.3f} 0.050</size></box>",
            "          </geometry>",
            "          <material><ambient>0.88 0.92 0.88 1</ambient><diffuse>0.88 0.92 0.88 1</diffuse></material>",
            "        </visual>",
            "      </link>",
            "    </model>",
        ]
    )


def _render_box(obstacle: RectObstacle) -> str:
    height = _height_for_obstacle(obstacle.name)
    x, y = obstacle.center
    sx, sy = obstacle.size
    color = _color_for_obstacle(obstacle.name)
    name = _safe_model_name(obstacle.name)
    return "\n".join(
        [
            f'    <model name="{escape(name)}">',
            "      <static>true</static>",
            f"      <pose>{x:.3f} {y:.3f} {height / 2.0:.3f} 0 0 0</pose>",
            f'      <link name="{escape(name)}_link">',
            f'        <collision name="{escape(name)}_collision">',
            "          <geometry>",
            f"            <box><size>{sx:.3f} {sy:.3f} {height:.3f}</size></box>",
            "          </geometry>",
            "        </collision>",
            f'        <visual name="{escape(name)}_visual">',
            "          <geometry>",
            f"            <box><size>{sx:.3f} {sy:.3f} {height:.3f}</size></box>",
            "          </geometry>",
            f"          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>",
            "        </visual>",
            "      </link>",
            "    </model>",
        ]
    )


def _write_metadata(spec: SemanticMapSpec, world_path: Path, metadata_path: Path) -> None:
    data = {
        **asdict(spec),
        "world_path": str(world_path),
        "source": "gazebo_3d_projection",
        "note": "Collision geometry is stored in the SDF world; goals/spawn/bounds are sidecar metadata.",
    }
    data["obstacles"] = [asdict(item) for item in spec.obstacles]
    _atomic_write_text(metadata_path, json.dumps(data, indent=2))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _read_json_with_retry(path: Path, attempts: int = 5, delay_s: float = 0.05):
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_s * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(path)


def _parse_world_collision_rects(world_path: Path) -> tuple[RectObstacle, ...]:
    root = ET.parse(world_path).getroot()
    obstacles: list[RectObstacle] = []
    for model in root.findall(".//model"):
        name = model.attrib.get("name", "model")
        if name.endswith("_floor"):
            continue
        model_pose = _pose6(model.findtext("pose"))
        for link in model.findall("link"):
            link_pose = _combine_pose(model_pose, _pose6(link.findtext("pose")))
            for collision in link.findall("collision"):
                pose = _combine_pose(link_pose, _pose6(collision.findtext("pose")))
                size_text = collision.findtext("./geometry/box/size")
                if not size_text:
                    continue
                sx, sy, sz = [float(value) for value in size_text.split()]
                if sz < 0.15:
                    continue
                obstacles.append(RectObstacle(name, (pose[0], pose[1]), (sx, sy)))
    return tuple(obstacles)


def _pose6(text: str | None) -> tuple[float, float, float, float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(item) for item in text.split()]
    values += [0.0] * (6 - len(values))
    return tuple(values[:6])  # type: ignore[return-value]


def _combine_pose(
    base: tuple[float, float, float, float, float, float],
    local: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    bx, by, bz, br, bp, byaw = base
    lx, ly, lz, lr, lp, lyaw = local
    cos_yaw = math.cos(byaw)
    sin_yaw = math.sin(byaw)
    return (
        bx + cos_yaw * lx - sin_yaw * ly,
        by + sin_yaw * lx + cos_yaw * ly,
        bz + lz,
        br + lr,
        bp + lp,
        byaw + lyaw,
    )


def _height_for_obstacle(name: str) -> float:
    lower = name.lower()
    if any(token in lower for token in ("wall", "outer", "spine", "divider", "partition", "core")):
        return 1.25
    if any(token in lower for token in ("shelf", "rack", "cabinet", "wardrobe")):
        return 1.10
    if any(token in lower for token in ("bed", "sofa", "counter", "desk", "table", "bench")):
        return 0.55
    return 0.75


def _color_for_obstacle(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("wall", "outer", "divider", "spine", "partition")):
        return "0.27 0.32 0.34 1"
    if any(token in lower for token in ("bed", "sofa", "chair")):
        return "0.55 0.65 0.72 1"
    if any(token in lower for token in ("shelf", "rack", "cabinet")):
        return "0.33 0.38 0.40 1"
    return "0.42 0.48 0.50 1"


def _safe_model_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    return safe.strip("_") or "obstacle"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and list Gazebo 3D worlds for the semantic training maps.")
    parser.add_argument("--maps", default="all", help="Comma-separated map ids, or all.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    map_ids = ALL_SEMANTIC_MAP_IDS if args.maps == "all" else tuple(item.strip() for item in args.maps.split(",") if item.strip())
    root = project_root()
    paths = ensure_semantic_3d_worlds(root, map_ids=tuple(map_ids), overwrite=True)
    rows = []
    for map_id, path in zip(map_ids, paths, strict=True):
        metadata = json.loads(semantic_metadata_path(map_id, root).read_text(encoding="utf-8"))
        rows.append(
            {
                "map_id": map_id,
                "label": metadata["label"],
                "world_path": str(path),
                "available": path.exists(),
                "spawn": metadata["spawn"],
                "goal_count": len(metadata["goal_points"]),
            }
        )

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(f"Generated Gazebo 3D semantic worlds under: {generated_world_dir(root)}")
    for row in rows:
        status = "OK" if row["available"] else "MISSING"
        print(f"{status:7} {row['map_id']:<24} goals={row['goal_count']:<2d} {row['world_path']}")


if __name__ == "__main__":
    main()
