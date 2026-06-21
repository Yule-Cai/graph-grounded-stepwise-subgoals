#!/usr/bin/env python3
"""Generate Paper A multi-map generalization cases.

The generated CSVs are intentionally compatible with
run_map_conditioned_llm_planning.py.  They provide held-out projected layouts
with varied start-goal-risk tuples, while keeping the same route-interface
evaluation protocol as the main Paper A experiments.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import Any

from llm_rl_nav.envs.semantic_world_source import build_nav_env
from llm_rl_nav.training.eval_llm_route_planning import _base_env, _candidate_route


FIELDNAMES = [
    "case_id",
    "scenario",
    "map_id",
    "start_x",
    "start_y",
    "goal_x",
    "goal_y",
    "yaw",
    "euclidean_distance",
    "candidate_count",
    "straight_line_safe",
    "difficulty_tag",
    "route_constraint",
    "risk_center_x",
    "risk_center_y",
    "risk_radius",
]


def parse_maps(text: str) -> list[str]:
    return [item.strip() for item in text.replace(",", " ").split() if item.strip()]


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def line_free(base: Any, a: tuple[float, float], b: tuple[float, float], step: float = 0.35) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(dist / step)))
    for i in range(n + 1):
        t = i / n
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if base._in_collision(x, y):
            return False
    return True


def sample_free_point(base: Any, rng: random.Random, margin: float = 1.2) -> tuple[float, float]:
    x0, x1 = base.x_bounds
    y0, y1 = base.y_bounds
    for _ in range(2000):
        x = rng.uniform(x0 + margin, x1 - margin)
        y = rng.uniform(y0 + margin, y1 - margin)
        if not base._in_collision(x, y):
            return (round(x, 3), round(y, 3))
    raise RuntimeError("Could not sample a collision-free point")


def midpoint_risk(
    base: Any,
    start: tuple[float, float],
    goal: tuple[float, float],
    rng: random.Random,
    radius: float,
) -> tuple[float, float]:
    mx = (start[0] + goal[0]) / 2.0
    my = (start[1] + goal[1]) / 2.0
    dx = goal[0] - start[0]
    dy = goal[1] - start[1]
    length = math.hypot(dx, dy) or 1.0
    px, py = -dy / length, dx / length
    for scale in (0.0, 0.35, -0.35, 0.7, -0.7, 1.05, -1.05):
        jitter = rng.uniform(-0.25, 0.25)
        candidate = (mx + px * radius * (scale + jitter), my + py * radius * (scale + jitter))
        if not base._in_collision(candidate[0], candidate[1]):
            return (round(candidate[0], 3), round(candidate[1], 3))
    return (round(mx, 3), round(my, 3))


def build_case_rows(
    map_id: str,
    scenario: str,
    episodes: int,
    seed: int,
    map_source: str,
    min_distance: float,
    max_distance: float,
    risk_radius: float,
) -> list[dict[str, Any]]:
    env = build_nav_env(map_source, seed=seed, map_id=map_id, max_steps=900, reward_profile="v8_goal")
    base = _base_env(env)
    rng = random.Random(f"{seed}:{map_id}:{scenario}")
    goal_points = [tuple(map(float, point)) for point in getattr(base, "goal_points", [])]
    rows: list[dict[str, Any]] = []
    attempts = 0
    while len(rows) < episodes and attempts < episodes * 400:
        attempts += 1
        if goal_points and rng.random() < 0.45:
            start = rng.choice(goal_points)
            goal = rng.choice(goal_points)
            if start == goal:
                continue
        else:
            start = sample_free_point(base, rng)
            goal = sample_free_point(base, rng)
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if distance < min_distance or distance > max_distance:
            continue
        if base._in_collision(start[0], start[1]) or base._in_collision(goal[0], goal[1]):
            continue
        if hasattr(base, "_same_reachable_component") and not base._same_reachable_component(start, goal):
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        env.reset(seed=seed + attempts, options={"map_id": map_id, "start": start, "goal": goal, "yaw": yaw})
        base = _base_env(env)
        candidate = _candidate_route(base, goal, spacing=2.2, resolution=0.55)
        if len(candidate) < 2:
            continue
        risk_center = midpoint_risk(base, start, goal, rng, risk_radius)
        straight_safe = int(line_free(base, start, goal))
        case_index = len(rows)
        route_constraint = (
            "Prefer the safer route and avoid the marked risk/narrow zone even if the path is longer."
            if scenario == "semantic_constraint"
            else "Decompose the long-horizon route into stable intermediate waypoints."
        )
        difficulty = "semantic_route_preference" if scenario == "semantic_constraint" else "long_horizon_topological"
        rows.append(
            {
                "case_id": f"{map_id}_{scenario}_{case_index:03d}",
                "scenario": scenario,
                "map_id": map_id,
                "start_x": f"{start[0]:.3f}",
                "start_y": f"{start[1]:.3f}",
                "goal_x": f"{goal[0]:.3f}",
                "goal_y": f"{goal[1]:.3f}",
                "yaw": f"{yaw:.4f}",
                "euclidean_distance": f"{distance:.3f}",
                "candidate_count": len(candidate),
                "straight_line_safe": straight_safe,
                "difficulty_tag": difficulty,
                "route_constraint": route_constraint,
                "risk_center_x": f"{risk_center[0]:.3f}",
                "risk_center_y": f"{risk_center[1]:.3f}",
                "risk_radius": f"{risk_radius:.3f}",
            }
        )
    if len(rows) < episodes:
        raise RuntimeError(f"Generated only {len(rows)} / {episodes} cases for map={map_id} scenario={scenario}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", default="reference_villa_ground studio_apartment townhouse_long luxury_villa")
    parser.add_argument("--scenarios", default="long_horizon semantic_constraint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--map-source", default="gazebo_3d_projection")
    parser.add_argument("--min-distance", type=float, default=8.0)
    parser.add_argument("--max-distance", type=float, default=32.0)
    parser.add_argument("--risk-radius", type=float, default=2.6)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "experiments" / "paper_a" / "cases" / "multimap_generalization",
    )
    args = parser.parse_args()

    for map_id in parse_maps(args.maps):
        for scenario in parse_maps(args.scenarios):
            rows = build_case_rows(
                map_id=map_id,
                scenario=scenario,
                episodes=args.episodes,
                seed=args.seed,
                map_source=args.map_source,
                min_distance=args.min_distance,
                max_distance=args.max_distance,
                risk_radius=args.risk_radius,
            )
            path = args.out_dir / f"proactive_{map_id}_{scenario}_{args.episodes}.csv"
            write_csv(path, rows)
            print(f"wrote {path} rows={len(rows)}")


if __name__ == "__main__":
    main()
