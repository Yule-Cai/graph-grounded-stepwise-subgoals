from __future__ import annotations

import argparse
import math
from collections import Counter, deque
from dataclasses import dataclass

from llm_rl_nav.envs.hospital_2d_env import ALL_SEMANTIC_MAP_IDS, RectObstacle, semantic_map_specs
from llm_rl_nav.envs.semantic_world_source import (
    MAP_SOURCE_CHOICES,
    MAP_SOURCE_SEMANTIC_2D,
    semantic_3d_map_specs,
)


@dataclass(frozen=True)
class GeometryAudit:
    map_id: str
    clearance: float
    components: int
    free_cells: int
    spawn_reachable_cells: int
    spawn_reachable_ratio: float
    reachable_goals: int
    total_goals: int
    blocked_goals: tuple[int, ...]
    unreachable_goals: tuple[int, ...]
    spawn_blocked: bool

    @property
    def ok(self) -> bool:
        return (
            not self.spawn_blocked
            and self.components <= 1
            and not self.blocked_goals
            and not self.unreachable_goals
            and self.spawn_reachable_ratio >= 0.95
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit semantic navigation maps for connectivity, narrow doors, and unreachable goals."
    )
    parser.add_argument(
        "--maps",
        default="all",
        help="Comma-separated semantic map ids, or 'all'.",
    )
    parser.add_argument(
        "--map-source",
        default=MAP_SOURCE_SEMANTIC_2D,
        choices=MAP_SOURCE_CHOICES,
        help="semantic_2d audits the legacy projection; gazebo_3d_projection audits generated SDF collision geometry.",
    )
    parser.add_argument(
        "--clearances",
        default="0.18,0.22,0.30,0.40,0.55",
        help="Comma-separated obstacle expansion radii in meters. 0.18 approximates robot body radius.",
    )
    parser.add_argument("--resolution", type=float, default=0.35, help="Audit grid resolution in meters.")
    parser.add_argument("--fail-on-issues", action="store_true", help="Exit non-zero when any map fails.")
    args = parser.parse_args()

    map_ids = ALL_SEMANTIC_MAP_IDS if args.maps == "all" else tuple(item.strip() for item in args.maps.split(",") if item.strip())
    clearances = tuple(float(item.strip()) for item in args.clearances.split(",") if item.strip())
    specs = _specs_for_source(args.map_source, map_ids)

    had_issue = False
    print(f"map_source={args.map_source}")
    print(f"resolution={args.resolution:.2f}m")
    for map_id in map_ids:
        if map_id not in specs:
            raise SystemExit(f"Unknown map id: {map_id}")
        print(f"\n=== {map_id}: {specs[map_id].label} ===")
        for clearance in clearances:
            audit = audit_map(map_id, clearance, args.resolution, specs)
            had_issue = had_issue or not audit.ok
            status = "OK" if audit.ok else "CHECK"
            blocked = _format_indices(audit.blocked_goals)
            unreachable = _format_indices(audit.unreachable_goals)
            print(
                f"{status:5} clearance={clearance:>4.2f} "
                f"components={audit.components:<2d} "
                f"spawn_area={audit.spawn_reachable_ratio:>5.2f} "
                f"goals={audit.reachable_goals}/{audit.total_goals} "
                f"spawn_blocked={str(audit.spawn_blocked):<5} "
                f"blocked_goals={blocked:<8} "
                f"unreachable_goals={unreachable}"
            )

    if had_issue and args.fail_on_issues:
        raise SystemExit(2)


def audit_map(
    map_id: str,
    clearance: float,
    resolution: float,
    specs: dict[str, object] | None = None,
) -> GeometryAudit:
    spec = (specs or semantic_map_specs())[map_id]
    obstacles = tuple(item.expanded(clearance) for item in spec.obstacles)
    free_cells = _build_free_cells(spec.x_bounds, spec.y_bounds, obstacles, resolution)
    components = _label_components(free_cells)
    component_counts = Counter(components.values())
    spawn_cell = _nearest_free_cell((spec.spawn[0], spec.spawn[1]), free_cells, spec.x_bounds, spec.y_bounds, resolution)
    spawn_blocked = spawn_cell is None
    spawn_component = components.get(spawn_cell) if spawn_cell is not None else None
    spawn_reachable_cells = component_counts.get(spawn_component, 0)
    spawn_reachable_ratio = spawn_reachable_cells / max(len(free_cells), 1)

    blocked_goals: list[int] = []
    unreachable_goals: list[int] = []
    for index, goal in enumerate(spec.goal_points):
        if _point_in_collision(goal[0], goal[1], spec.x_bounds, spec.y_bounds, obstacles):
            blocked_goals.append(index)
            continue
        goal_cell = _nearest_free_cell(goal, free_cells, spec.x_bounds, spec.y_bounds, resolution)
        if goal_cell is None or components.get(goal_cell) != spawn_component:
            unreachable_goals.append(index)

    return GeometryAudit(
        map_id=map_id,
        clearance=clearance,
        components=len(component_counts),
        free_cells=len(free_cells),
        spawn_reachable_cells=spawn_reachable_cells,
        spawn_reachable_ratio=spawn_reachable_ratio,
        reachable_goals=len(spec.goal_points) - len(blocked_goals) - len(unreachable_goals),
        total_goals=len(spec.goal_points),
        blocked_goals=tuple(blocked_goals),
        unreachable_goals=tuple(unreachable_goals),
        spawn_blocked=spawn_blocked,
    )


def _specs_for_source(map_source: str, map_ids: tuple[str, ...]):
    if map_source == MAP_SOURCE_SEMANTIC_2D:
        return semantic_map_specs()
    return semantic_3d_map_specs(map_ids=map_ids)


def _build_free_cells(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    obstacles: tuple[RectObstacle, ...],
    resolution: float,
) -> set[tuple[int, int]]:
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    nx = int(math.ceil((xmax - xmin) / resolution))
    ny = int(math.ceil((ymax - ymin) / resolution))
    free: set[tuple[int, int]] = set()
    for ix in range(nx):
        for iy in range(ny):
            x = xmin + (ix + 0.5) * resolution
            y = ymin + (iy + 0.5) * resolution
            if not _point_in_collision(x, y, x_bounds, y_bounds, obstacles):
                free.add((ix, iy))
    return free


def _label_components(free_cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    labels: dict[tuple[int, int], int] = {}
    component_id = 0
    for start in free_cells:
        if start in labels:
            continue
        queue: deque[tuple[int, int]] = deque([start])
        labels[start] = component_id
        while queue:
            current = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in free_cells and neighbor not in labels:
                    labels[neighbor] = component_id
                    queue.append(neighbor)
        component_id += 1
    return labels


def _nearest_free_cell(
    point: tuple[float, float],
    free_cells: set[tuple[int, int]],
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    resolution: float,
) -> tuple[int, int] | None:
    if not free_cells:
        return None
    ix = int((point[0] - x_bounds[0]) / resolution)
    iy = int((point[1] - y_bounds[0]) / resolution)
    best: tuple[int, int] | None = None
    best_distance = float("inf")
    for radius in range(0, 16):
        found = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if radius and max(abs(dx), abs(dy)) != radius:
                    continue
                cell = (ix + dx, iy + dy)
                if cell not in free_cells:
                    continue
                found = True
                cx = x_bounds[0] + (cell[0] + 0.5) * resolution
                cy = y_bounds[0] + (cell[1] + 0.5) * resolution
                distance = math.hypot(cx - point[0], cy - point[1])
                if distance < best_distance:
                    best_distance = distance
                    best = cell
        if found:
            return best
    return None


def _point_in_collision(
    x: float,
    y: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    obstacles: tuple[RectObstacle, ...],
) -> bool:
    if not (x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]):
        return True
    return any(obstacle.contains(x, y) for obstacle in obstacles)


def _format_indices(indices: tuple[int, ...]) -> str:
    return "-" if not indices else ",".join(str(index) for index in indices)


if __name__ == "__main__":
    main()
