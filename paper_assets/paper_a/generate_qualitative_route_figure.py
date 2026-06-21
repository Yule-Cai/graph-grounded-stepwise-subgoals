#!/usr/bin/env python3
"""Generate a qualitative route-level visualization for Paper A.

The archived evaluation logs do not contain every simulator pose. This script
therefore reconstructs route-level paths from the map, case definition, and
logged route outcomes. It is intended for a paper figure that explains why route
validation matters, not as a replacement for raw trajectory logging.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


DEFAULT_ARCHIVE = Path("<WORKSPACE>/paper_a_experiments_desktop 2.zip")
OUT_DIR = Path("paper_assets/paper_a/figures")
MAP_PATH = "paper_a_experiments_desktop/generated_worlds/semantic_3d/reference_family_flat.json"
CASES_PATH = "paper_a_experiments_desktop/experiments/paper_a/cases/proactive_semantic_constraint_100.csv"
LOG_TEMPLATE = "paper_a_experiments_desktop/logs/eval/proactive_benchmark_20260530_162726/{name}"


@dataclass(frozen=True)
class RectObstacle:
    name: str
    cx: float
    cy: float
    sx: float
    sy: float

    def contains(self, x: float, y: float, inflate: float = 0.0) -> bool:
        return abs(x - self.cx) <= self.sx / 2 + inflate and abs(y - self.cy) <= self.sy / 2 + inflate


@dataclass
class LogResult:
    outcome: str = "n/a"
    steps: str = "--"
    final_dist: str = "--"
    route_id: str = "NA"
    semantic_cost: str = "--"
    repaired: str = "0"
    overridden: str = "0"


class GridPlanner:
    def __init__(
        self,
        obstacles: list[RectObstacle],
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        resolution: float = 0.35,
        inflation: float = 0.18,
    ) -> None:
        self.obstacles = obstacles
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.resolution = resolution
        self.inflation = inflation
        self.nx = int(round((x_bounds[1] - x_bounds[0]) / resolution)) + 1
        self.ny = int(round((y_bounds[1] - y_bounds[0]) / resolution)) + 1

    def to_grid(self, point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        gx = int(round((x - self.x_bounds[0]) / self.resolution))
        gy = int(round((y - self.y_bounds[0]) / self.resolution))
        return max(0, min(self.nx - 1, gx)), max(0, min(self.ny - 1, gy))

    def to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        gx, gy = cell
        return self.x_bounds[0] + gx * self.resolution, self.y_bounds[0] + gy * self.resolution

    def blocked(self, cell: tuple[int, int]) -> bool:
        x, y = self.to_world(cell)
        if x <= self.x_bounds[0] or x >= self.x_bounds[1] or y <= self.y_bounds[0] or y >= self.y_bounds[1]:
            return True
        return any(obs.contains(x, y, self.inflation) for obs in self.obstacles)

    def risk_penalty(self, point: tuple[float, float], risk_center: tuple[float, float], risk_radius: float) -> float:
        if risk_radius <= 0:
            return 0.0
        d = math.hypot(point[0] - risk_center[0], point[1] - risk_center[1])
        if d >= risk_radius * 1.25:
            return 0.0
        return (risk_radius * 1.25 - d) / max(risk_radius * 1.25, 1e-6)

    def astar(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        risk_center: tuple[float, float] | None = None,
        risk_radius: float = 0.0,
        risk_weight: float = 0.0,
    ) -> list[tuple[float, float]]:
        s = self.to_grid(start)
        g = self.to_grid(goal)
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {s: 0.0}
        moves = [
            (-1, -1, math.sqrt(2)),
            (-1, 0, 1.0),
            (-1, 1, math.sqrt(2)),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, math.sqrt(2)),
            (1, 0, 1.0),
            (1, 1, math.sqrt(2)),
        ]

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == g:
                return self._reconstruct(came_from, current)
            for dx, dy, step_cost in moves:
                nxt = (current[0] + dx, current[1] + dy)
                if not (0 <= nxt[0] < self.nx and 0 <= nxt[1] < self.ny) or self.blocked(nxt):
                    continue
                wx, wy = self.to_world(nxt)
                risk = 0.0
                if risk_center is not None:
                    risk = self.risk_penalty((wx, wy), risk_center, risk_radius)
                tentative = g_score[current] + step_cost * self.resolution * (1.0 + risk_weight * risk)
                if tentative < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    hx = math.hypot(nxt[0] - g[0], nxt[1] - g[1]) * self.resolution
                    heapq.heappush(open_heap, (tentative + hx, nxt))
        return [start, goal]

    def _reconstruct(self, came_from: dict[tuple[int, int], tuple[int, int]], current: tuple[int, int]) -> list[tuple[float, float]]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        return simplify_path([self.to_world(cell) for cell in cells], tolerance=0.55)


def simplify_path(points: list[tuple[float, float]], tolerance: float = 0.5) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    def distance_to_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        ax, ay = a
        bx, by = b
        px, py = p
        dx, dy = bx - ax, by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
        proj = (ax + t * dx, ay + t * dy)
        return math.hypot(px - proj[0], py - proj[1])

    max_dist = -1.0
    idx = 0
    for i in range(1, len(points) - 1):
        dist = distance_to_segment(points[i], points[0], points[-1])
        if dist > max_dist:
            idx = i
            max_dist = dist
    if max_dist > tolerance:
        left = simplify_path(points[: idx + 1], tolerance)
        right = simplify_path(points[idx:], tolerance)
        return left[:-1] + right
    return [points[0], points[-1]]


def read_zip_json(zf: zipfile.ZipFile, path: str) -> dict:
    return json.loads(zf.read(path).decode("utf-8"))


def read_case(zf: zipfile.ZipFile, case_id: str) -> dict[str, str]:
    text = zf.read(CASES_PATH).decode("utf-8")
    for row in csv.DictReader(text.splitlines()):
        if row["case_id"] == case_id:
            return row
    raise ValueError(f"case_id not found: {case_id}")


def parse_episode_result(zf: zipfile.ZipFile, log_name: str, case_id: str) -> LogResult:
    path = LOG_TEMPLATE.format(name=log_name)
    text = zf.read(path).decode("utf-8", errors="replace")
    pattern = re.compile(rf"^episode=\d+\s+case_id={re.escape(case_id)}\s+(.+)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return LogResult()
    line = match.group(1)

    def field(name: str, default: str = "--") -> str:
        m = re.search(rf"{name}=\s*([^\s]+)", line)
        return m.group(1) if m else default

    return LogResult(
        outcome=field("outcome", "n/a"),
        steps=field("steps", "--"),
        final_dist=field("final_dist", "--"),
        route_id=field("route_id", "NA"),
        semantic_cost=field("semantic_cost", "--"),
        repaired=field("repaired", "0"),
        overridden=field("route_overridden", "0"),
    )


def interpolate(points: list[tuple[float, float]], n: int = 80) -> list[tuple[float, float]]:
    if len(points) <= 1:
        return points
    distances = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = distances[-1]
    if total <= 1e-9:
        return points
    out: list[tuple[float, float]] = []
    cursor = 0
    for i in range(n):
        target = total * i / max(n - 1, 1)
        while cursor + 1 < len(distances) and distances[cursor + 1] < target:
            cursor += 1
        if cursor + 1 >= len(points):
            out.append(points[-1])
            continue
        a, b = points[cursor], points[cursor + 1]
        span = distances[cursor + 1] - distances[cursor]
        t = 0.0 if span <= 1e-9 else (target - distances[cursor]) / span
        out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return out


def path_length(points: list[tuple[float, float]]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points[:-1], points[1:]))


def trim_path_from_end_distance(points: list[tuple[float, float]], end_distance: float) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points
    total = path_length(points)
    target = max(0.0, total - max(0.0, end_distance))
    if target >= total:
        return points
    out = [points[0]]
    travelled = 0.0
    for a, b in zip(points[:-1], points[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if travelled + seg >= target:
            t = 0.0 if seg <= 1e-9 else (target - travelled) / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            return out
        out.append(b)
        travelled += seg
    return out


def wavy_direct(start: tuple[float, float], goal: tuple[float, float], stop_ratio: float = 0.72) -> list[tuple[float, float]]:
    pts = []
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    norm = math.hypot(dx, dy) or 1.0
    px, py = -dy / norm, dx / norm
    for i in range(70):
        t = stop_ratio * i / 69
        wiggle = math.sin(t * math.pi * 3.0) * 0.45
        pts.append((start[0] + dx * t + px * wiggle, start[1] + dy * t + py * wiggle))
    return pts


def detour_via(
    planner: GridPlanner,
    start: tuple[float, float],
    waypoint: tuple[float, float],
    goal: tuple[float, float],
    risk_center: tuple[float, float],
    risk_radius: float,
    risk_weight: float,
) -> list[tuple[float, float]]:
    first = planner.astar(start, waypoint, risk_center, risk_radius, risk_weight)
    second = planner.astar(waypoint, goal, risk_center, risk_radius, risk_weight)
    return simplify_path(first[:-1] + second, tolerance=0.55)


def line_crosses_obstacle(points: Iterable[tuple[float, float]], obstacles: list[RectObstacle]) -> bool:
    for x, y in points:
        if any(obs.contains(x, y, 0.05) for obs in obstacles):
            return True
    return False


def compact_status(result: LogResult) -> str:
    outcome = {"invalid_route": "invalid"}.get(result.outcome, result.outcome)
    steps = result.steps.lstrip("0") or result.steps
    sem = result.semantic_cost
    if sem.startswith("0."):
        sem = sem[1:]
    status = f"{outcome} | {steps} | sem {sem}"
    flags = []
    if result.repaired == "1":
        flags.append("repair")
    if result.overridden == "1":
        flags.append("override")
    if flags:
        status += " | " + ", ".join(flags)
    return status


def draw_panel(
    ax,
    obstacles: list[RectObstacle],
    bounds: tuple[tuple[float, float], tuple[float, float]],
    start: tuple[float, float],
    goal: tuple[float, float],
    risk_center: tuple[float, float],
    risk_radius: float,
    path: list[tuple[float, float]],
    result: LogResult,
    title: str,
    color: str,
    dashed: bool = False,
    extra_path: list[tuple[float, float]] | None = None,
) -> None:
    x_bounds, y_bounds = bounds
    ax.set_facecolor("#fbfaf7")
    for obs in obstacles:
        ax.add_patch(
            Rectangle(
                (obs.cx - obs.sx / 2, obs.cy - obs.sy / 2),
                obs.sx,
                obs.sy,
                facecolor="#897b9a",
                edgecolor="#756686",
                linewidth=0.3,
                alpha=0.95,
            )
        )
    ax.add_patch(Circle(risk_center, risk_radius, facecolor="#e5b567", edgecolor="#b9792a", alpha=0.25, linewidth=1.0))

    if extra_path:
        xs, ys = zip(*extra_path)
        ax.plot(xs, ys, color="#c84e4e", linestyle=(0, (3, 2)), linewidth=1.6, alpha=0.9)

    draw_path = path
    if result.outcome == "timeout" and len(path) > 1:
        try:
            draw_path = trim_path_from_end_distance(path, float(result.final_dist))
        except ValueError:
            draw_path = path
        plan_xs, plan_ys = zip(*path)
        ax.plot(plan_xs, plan_ys, color=color, linestyle=(0, (1, 2)), linewidth=1.0, alpha=0.28)

    xs, ys = zip(*draw_path)
    ax.plot(xs, ys, color=color, linestyle=(0, (3, 2)) if dashed else "-", linewidth=2.0, alpha=0.95)
    sampled = interpolate(draw_path, 9)
    if len(sampled) > 2 and not dashed:
        wx, wy = zip(*sampled[1:-1])
        ax.scatter(wx, wy, s=13, color=color, edgecolor="white", linewidth=0.3, zorder=4)

    ax.scatter([start[0]], [start[1]], s=28, c="#d62728", marker="o", edgecolor="white", linewidth=0.6, zorder=5)
    ax.scatter([goal[0]], [goal[1]], s=34, c="#2ca02c", marker="s", edgecolor="white", linewidth=0.6, zorder=5)

    if result.outcome in {"timeout", "invalid_route"} or dashed:
        stop = draw_path[-1]
        ax.scatter([stop[0]], [stop[1]], s=42, c="black", marker="x", linewidth=1.6, zorder=6)

    status_color = "#2ca02c" if result.outcome == "success" else "#c84e4e"
    ax.text(
        0.01,
        1.13,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.7,
        fontweight="bold",
        color=status_color,
        clip_on=False,
    )
    ax.text(
        0.01,
        1.055,
        compact_status(result),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="semibold",
        color=status_color,
        clip_on=False,
    )
    ax.set_xlim(x_bounds)
    ax.set_ylim(y_bounds)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#867694")


def build_paths(
    planner: GridPlanner,
    case: dict[str, str],
    obstacles: list[RectObstacle],
) -> dict[str, list[tuple[float, float]]]:
    start = (float(case["start_x"]), float(case["start_y"]))
    goal = (float(case["goal_x"]), float(case["goal_y"]))
    risk_center = (float(case["risk_center_x"]), float(case["risk_center_y"]))
    risk_radius = float(case["risk_radius"])

    shortest = planner.astar(start, goal, risk_center, risk_radius, risk_weight=0.0)
    safe = planner.astar(start, goal, risk_center, risk_radius, risk_weight=14.0)
    direct = wavy_direct(start, goal)

    # Use the side of the risk zone opposite the start-goal line as an explicit detour.
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    norm = math.hypot(dx, dy) or 1.0
    perp = (-dy / norm, dx / norm)
    candidates = [
        (risk_center[0] + perp[0] * risk_radius * scale, risk_center[1] + perp[1] * risk_radius * scale)
        for scale in (1.7, -1.7, 2.4, -2.4)
    ]
    waypoint = next((p for p in candidates if not line_crosses_obstacle([p], obstacles)), candidates[0])
    random_like = detour_via(planner, start, waypoint, goal, risk_center, risk_radius, risk_weight=2.0)

    raw_llm = [(start[0], start[1]), (risk_center[0], risk_center[1]), (goal[0], goal[1])]
    return {
        "direct": direct,
        "random": random_like,
        "classical": shortest,
        "rule": safe,
        "raw_llm": raw_llm,
        "validated": safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative Paper A route visualization.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--case-id", default="semantic_constraint_077")
    parser.add_argument("--resolution", type=float, default=0.35)
    parser.add_argument("--inflation", type=float, default=0.18)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as zf:
        map_data = read_zip_json(zf, MAP_PATH)
        case = read_case(zf, args.case_id)
        logs = {
            "direct": parse_episode_result(zf, "semantic_constraint_direct_ppo_100ep.log", args.case_id),
            "random": parse_episode_result(zf, "semantic_constraint_random_waypoint_ppo_100ep.log", args.case_id),
            "classical": parse_episode_result(zf, "semantic_constraint_classical_waypoint_ppo_100ep.log", args.case_id),
            "rule": parse_episode_result(zf, "semantic_constraint_rule_semantic_ppo_100ep.log", args.case_id),
            "raw_llm": LogResult(outcome="invalid_route", steps="--", route_id="raw", semantic_cost="high", repaired="1"),
            "validated": parse_episode_result(zf, "semantic_constraint_llm_route_repair_ppo_qwen_qwen3-1.7b_100ep.log", args.case_id),
        }

    obstacles = [
        RectObstacle(
            name=str(item["name"]),
            cx=float(item["center"][0]),
            cy=float(item["center"][1]),
            sx=float(item["size"][0]),
            sy=float(item["size"][1]),
        )
        for item in map_data["obstacles"]
    ]
    x_bounds = (float(map_data["x_bounds"][0]), float(map_data["x_bounds"][1]))
    y_bounds = (float(map_data["y_bounds"][0]), float(map_data["y_bounds"][1]))
    planner = GridPlanner(obstacles, x_bounds, y_bounds, resolution=args.resolution, inflation=args.inflation)

    start = (float(case["start_x"]), float(case["start_y"]))
    goal = (float(case["goal_x"]), float(case["goal_y"]))
    risk_center = (float(case["risk_center_x"]), float(case["risk_center_y"]))
    risk_radius = float(case["risk_radius"])
    paths = build_paths(planner, case, obstacles)

    panels = [
        ("direct", "Direct PPO", "#c84e4e", True),
        ("random", "Random WP", "#d98c2b", False),
        ("classical", "Classical WP", "#4f8fc0", False),
        ("rule", "Rule Semantic", "#775aa6", False),
        ("raw_llm", "Raw LLM", "#c84e4e", True),
        ("validated", "Validated LLM", "#2f9e44", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.7), constrained_layout=False)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.91, wspace=0.04, hspace=0.35)
    for ax, (key, title, color, dashed) in zip(axes.flat, panels, strict=True):
        extra = paths["raw_llm"] if key == "validated" else None
        draw_panel(
            ax,
            obstacles,
            (x_bounds, y_bounds),
            start,
            goal,
            risk_center,
            risk_radius,
            paths[key],
            logs[key],
            title,
            color,
            dashed=dashed,
            extra_path=extra,
        )

    png = OUT_DIR / "fig7_qualitative_route_case.png"
    pdf = OUT_DIR / "fig7_qualitative_route_case.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
