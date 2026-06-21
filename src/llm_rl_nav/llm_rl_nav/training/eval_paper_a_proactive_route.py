from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any
from urllib import error, request

from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES
from llm_rl_nav.training.eval_llm_route_planning import (
    _base_env,
    _candidate_route,
    _chat_url,
    _current_obs,
    _json_from_text,
    _predict_action,
    _sanitize_route,
    _set_goal,
)
from llm_rl_nav.training.eval_paper_a_cases import (
    _apply_case_reset,
    _float,
    _load_cases,
    _make_env,
)
from llm_rl_nav.training.train_multimap_ppo import parse_maps
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def _route_length(route: list[tuple[float, float]], start: tuple[float, float]) -> float:
    if not route:
        return 0.0
    total = 0.0
    cursor = start
    for point in route:
        total += math.hypot(point[0] - cursor[0], point[1] - cursor[1])
        cursor = point
    return total


def _trajectory_length(points: list[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]) for i in range(1, len(points)))


def _resample_by_distance(points: list[tuple[float, float]], spacing: float = 0.18) -> list[tuple[float, float]]:
    """Resample a dense robot trajectory into roughly equal-distance points.

    The robot moves only a few centimetres per simulator step, so counting turns
    directly from adjacent raw states is numerically unstable. The old turn
    counter ignored movements shorter than 0.25 m, which made every executed
    trajectory report zero turns. This resampling step preserves real bends while
    filtering tiny jitter.
    """
    if len(points) < 2:
        return points
    out = [points[0]]
    last = points[0]
    acc = 0.0
    cursor = points[0]
    for nxt in points[1:]:
        seg = math.hypot(nxt[0] - cursor[0], nxt[1] - cursor[1])
        if seg <= 1e-9:
            cursor = nxt
            continue
        acc += seg
        if acc >= spacing:
            out.append(nxt)
            last = nxt
            acc = 0.0
        cursor = nxt
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


def _turn_count(points: list[tuple[float, float]], min_angle_deg: float = 25.0, resample_spacing: float = 0.18) -> int:
    # Counts meaningful bends in either a route skeleton or an executed trajectory.
    pts = _resample_by_distance(points, spacing=resample_spacing)
    if len(pts) < 3:
        return 0
    count = 0
    min_angle = math.radians(min_angle_deg)
    prev_angle: float | None = None
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        if math.hypot(dx, dy) < max(0.02, resample_spacing * 0.4):
            continue
        angle = math.atan2(dy, dx)
        if prev_angle is not None:
            diff = abs((angle - prev_angle + math.pi) % (2 * math.pi) - math.pi)
            if diff >= min_angle:
                count += 1
        prev_angle = angle
    return count


def _base_clearance(base, x: float, y: float) -> float:
    # Same geometry idea as ShieldedActionEnv._clearance, kept local so the route scorer
    # works even when the safety wrapper is disabled.
    best = float("inf")
    obstacles = getattr(base, "collision_obstacles", getattr(base, "obstacles", []))
    for obs in obstacles:
        if obs.contains(x, y):
            return 0.0
        cx, cy = obs.center
        sx, sy = obs.size
        xmin = cx - sx / 2.0
        xmax = cx + sx / 2.0
        ymin = cy - sy / 2.0
        ymax = cy + sy / 2.0
        dx = max(xmin - x, 0.0, x - xmax)
        dy = max(ymin - y, 0.0, y - ymax)
        best = min(best, math.hypot(dx, dy))
    if hasattr(base, "x_bounds") and hasattr(base, "y_bounds"):
        best = min(best, x - base.x_bounds[0], base.x_bounds[1] - x)
        best = min(best, y - base.y_bounds[0], base.y_bounds[1] - y)
    if best == float("inf"):
        return 9.99
    return max(float(best), 0.0)


def _sample_polyline(points: list[tuple[float, float]], step: float = 0.5) -> list[tuple[float, float]]:
    if not points:
        return []
    samples = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(dist / step)))
        for i in range(1, n + 1):
            t = i / n
            samples.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    return samples


def _route_clearance_stats(base, route: list[tuple[float, float]], start: tuple[float, float], target_clearance: float = 0.75) -> tuple[float, float]:
    samples = _sample_polyline([start] + list(route), step=0.55)
    if not samples:
        return 9.99, 0.0
    clearances = [_base_clearance(base, x, y) for x, y in samples]
    min_clearance = min(clearances)
    penalty = sum(max(0.0, target_clearance - c) / max(target_clearance, 1e-6) for c in clearances) / len(clearances)
    return float(min_clearance), float(penalty)


def _semantic_cost(route: list[tuple[float, float]], risk_center: tuple[float, float], risk_radius: float) -> float:
    if risk_radius <= 0:
        return 0.0
    cost = 0.0
    for point in route:
        dist = math.hypot(point[0] - risk_center[0], point[1] - risk_center[1])
        if dist < risk_radius:
            cost += (risk_radius - dist) / max(risk_radius, 1e-6)
    return float(cost)


def _trajectory_semantic_cost(points: list[tuple[float, float]], risk_center: tuple[float, float], risk_radius: float) -> float:
    """Semantic/risk exposure of the *executed* path.

    Direct RL has no planned route, so route-level semantic cost incorrectly made
    it look perfectly constraint-compliant. For semantic-route tasks, compliance
    must be measured from the actual trajectory taken by the robot.
    """
    if risk_radius <= 0 or not points:
        return 0.0
    samples = _resample_by_distance(points, spacing=0.20)
    cost = 0.0
    for x, y in samples:
        dist = math.hypot(x - risk_center[0], y - risk_center[1])
        if dist < risk_radius:
            cost += (risk_radius - dist) / max(risk_radius, 1e-6)
    # Normalize by sampled trajectory length so long safe paths are not unfairly penalized.
    return float(cost / max(len(samples), 1))


def _line_route(start: tuple[float, float], goal: tuple[float, float], spacing: float = 2.8) -> list[tuple[float, float]]:
    dist = math.hypot(goal[0] - start[0], goal[1] - start[1])
    n = max(1, int(math.ceil(dist / spacing)))
    return [
        (start[0] + (goal[0] - start[0]) * i / n, start[1] + (goal[1] - start[1]) * i / n)
        for i in range(1, n + 1)
    ]


def _plan_between(base, start: tuple[float, float], goal: tuple[float, float], spacing: float, resolution: float) -> list[tuple[float, float]]:
    planner = WaypointGoalEnv(base, waypoint_spacing=spacing, grid_resolution=resolution)
    return [(round(float(x), 2), round(float(y), 2)) for x, y in planner._plan_waypoints(start, goal)]


def _safe_detour_points(
    base,
    start: tuple[float, float],
    goal: tuple[float, float],
    risk_center: tuple[float, float],
    risk_radius: float,
) -> list[tuple[float, float]]:
    sx, sy = start
    gx, gy = goal
    dx, dy = gx - sx, gy - sy
    norm = math.hypot(dx, dy) or 1.0
    px, py = -dy / norm, dx / norm
    candidates: list[tuple[float, float]] = []
    for sign in (1.0, -1.0):
        for scale in (1.4, 1.8, 2.4, 3.2, 4.2):
            point = (risk_center[0] + sign * px * risk_radius * scale, risk_center[1] + sign * py * risk_radius * scale)
            if not base._in_collision(point[0], point[1]):
                candidates.append((round(float(point[0]), 2), round(float(point[1]), 2)))
    # Add loose diagonal candidates around the risk zone for maps where perpendicular points collide.
    for ox in (-1.0, 1.0):
        for oy in (-1.0, 1.0):
            for scale in (2.0, 2.6, 3.4):
                point = (risk_center[0] + ox * risk_radius * scale, risk_center[1] + oy * risk_radius * scale)
                if not base._in_collision(point[0], point[1]):
                    candidates.append((round(float(point[0]), 2), round(float(point[1]), 2)))
    seen: set[tuple[float, float]] = set()
    out: list[tuple[float, float]] = []
    for point in candidates:
        if point not in seen:
            seen.add(point)
            out.append(point)
    return out[:10]


def _execution_score(option: dict[str, Any], args) -> float:
    return (
        float(option.get("approx_length", 9999.0))
        + float(args.turn_weight) * float(option.get("turn_count", 0.0))
        + float(args.clearance_weight) * float(option.get("clearance_penalty", 0.0))
        + float(args.waypoint_count_weight) * float(option.get("waypoint_count", 0.0))
        + float(args.semantic_cost_weight) * float(option.get("semantic_cost", 0.0))
    )


def _route_options(base, case: dict[str, Any], args) -> list[dict[str, Any]]:
    start = (_float(case, "start_x"), _float(case, "start_y"))
    goal = (_float(case, "goal_x"), _float(case, "goal_y"))
    risk_center = (_float(case, "risk_center_x"), _float(case, "risk_center_y"))
    risk_radius = _float(case, "risk_radius", 0.0)
    scenario = str(case.get("scenario") or "long_horizon")

    shortest = _candidate_route(base, goal, spacing=args.waypoint_spacing, resolution=args.waypoint_grid_resolution)
    shortest, shortest_geom = _sanitize_route(base, shortest, goal, max_segment=args.max_segment)
    direct = _line_route(start, goal, spacing=max(args.waypoint_spacing, 2.8))
    direct, direct_geom = _sanitize_route(base, direct, goal, max_segment=args.max_segment)

    options: list[dict[str, Any]] = []

    def add(option_id: str, desc: str, route: list[tuple[float, float]], geom_valid: bool, preference_note: str = ""):
        if not route:
            return
        route = route[:28]
        length = _route_length(route, start)
        cost = _semantic_cost(route, risk_center, risk_radius) if scenario == "semantic_constraint" else 0.0
        turns = _turn_count([start] + route)
        min_clearance, clearance_penalty = _route_clearance_stats(base, route, start, target_clearance=args.fast_safe_clearance)
        tmp = {
            "option_id": option_id,
            "description": desc,
            "preference_note": preference_note,
            "route": route,
            "waypoint_count": len(route),
            "approx_length": round(length, 2),
            "semantic_cost": round(cost, 3),
            "geometry_valid": bool(geom_valid),
            "turn_count": int(turns),
            "min_clearance": round(min_clearance, 3),
            "clearance_penalty": round(clearance_penalty, 3),
        }
        tmp["execution_score"] = round(_execution_score(tmp, args), 3)
        options.append(tmp)

    add("shortest", "Classical shortest/low-distance waypoint route.", shortest, shortest_geom, "Best distance baseline.")
    add("direct_split", "Straight-line split route, useful only in open layouts.", direct, direct_geom, "Naive direct decomposition baseline.")

    # Construct several safe detour candidates for semantic route preference and long-horizon decomposition.
    for idx, detour in enumerate(_safe_detour_points(base, start, goal, risk_center, max(risk_radius, 2.2))):
        part1 = _plan_between(base, start, detour, args.waypoint_spacing, args.waypoint_grid_resolution)
        part2 = _plan_between(base, detour, goal, args.waypoint_spacing, args.waypoint_grid_resolution)
        route = part1 + part2
        route, geom = _sanitize_route(base, route, goal, max_segment=args.max_segment)
        add(
            f"detour_{idx}",
            "Alternative topological detour through a different corridor/frontier.",
            route,
            geom,
            "Prefer this when it is faster to execute safely: wider clearance, fewer sharp turns, or lower semantic risk.",
        )

    # Deduplicate approximately identical routes.
    dedup: list[dict[str, Any]] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for option in options:
        sig = tuple((round(x * 2), round(y * 2)) for x, y in option["route"][:8])
        if sig not in signatures:
            signatures.add(sig)
            dedup.append(option)
    return dedup[:10]


def _select_rule_option(options: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    valid = [item for item in options if item.get("geometry_valid")]
    pool = valid or options
    if not pool:
        raise ValueError("No route options available")
    if scenario == "semantic_constraint":
        return min(pool, key=lambda item: (float(item.get("semantic_cost", 999.0)), float(item.get("approx_length", 9999.0))))
    return min(pool, key=lambda item: float(item.get("approx_length", 9999.0)))


def _select_fast_safe_option(options: list[dict[str, Any]], args) -> dict[str, Any]:
    valid = [item for item in options if item.get("geometry_valid")]
    pool = valid or options
    if not pool:
        raise ValueError("No route options available")
    return min(pool, key=lambda item: (float(item.get("execution_score", 999999.0)), float(item.get("approx_length", 9999.0))))


def _select_random_option(options: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    valid = [item for item in options if item.get("geometry_valid")]
    pool = valid or options
    if not pool:
        raise ValueError("No route options available")
    return rng.choice(pool)


def _request_llm_option(
    options: list[dict[str, Any]],
    case: dict[str, Any],
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    compact_options = []
    for option in options:
        compact_options.append(
            {
                "option_id": option["option_id"],
                "description": option["description"],
                "waypoint_count": option["waypoint_count"],
                "approx_length": option["approx_length"],
                "turn_count": option["turn_count"],
                "min_clearance": option["min_clearance"],
                "clearance_penalty": option["clearance_penalty"],
                "semantic_cost": option["semantic_cost"],
                "execution_score": option["execution_score"],
                "geometry_valid": option["geometry_valid"],
                "first_waypoints": [[round(x, 2), round(y, 2)] for x, y in option["route"][:5]],
            }
        )
    payload = {
        "scenario": case.get("scenario", "long_horizon"),
        "map_id": case.get("map_id", "reference_family_flat"),
        "start": [case.get("start_x"), case.get("start_y")],
        "goal": [case.get("goal_x"), case.get("goal_y")],
        "route_constraint": case.get("route_constraint", "Use the fastest collision-free route."),
        "selection_rule": "Choose the geometry_valid route with the lowest execution_score. execution_score already penalizes long routes, many turns, low clearance, too many subgoals, and semantic/risk cost.",
        "risk_zone": {
            "center": [case.get("risk_center_x"), case.get("risk_center_y")],
            "radius": case.get("risk_radius"),
            "meaning": "semantic/risk/narrow zone to avoid when the instruction asks for a safer route",
        },
        "route_options": compact_options,
        "instruction": "Choose exactly one option_id. Do not invent new coordinates or routes.",
    }
    prompt = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a pre-execution route planner for an indoor robot with an RL local controller.",
                        "Return compact JSON only. Do not use markdown.",
                        'Schema: {"selected_route":"shortest","reason":"one short sentence"}',
                        "Choose selected_route only from the provided route_options option_id values.",
                        "Primary objective: fastest collision-free execution, not merely shortest geometric distance.",
                        "Prefer low execution_score: shorter executed time, fewer turns, wider clearance, fewer subgoals, lower semantic/risk cost.",
                        "For semantic_constraint scenarios, the route must avoid semantic/risk zones when possible while staying fast.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.02,
        "max_tokens": max_tokens,
    }
    req = request.Request(
        _chat_url(base_url),
        data=json.dumps(prompt, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    parse_ok = True
    selected = ""
    try:
        parsed = _json_from_text(content)
        if isinstance(parsed, dict):
            selected = str(parsed.get("selected_route") or parsed.get("option_id") or parsed.get("route") or "")
        else:
            parse_ok = False
    except Exception:
        parse_ok = False
        m = re.search(r"(?:selected_route|option_id|route)\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+)", content)
        if m:
            selected = m.group(1)
        else:
            for option in options:
                if str(option["option_id"]) in content:
                    selected = str(option["option_id"])
                    break
    option_by_id = {str(item["option_id"]): item for item in options}
    meta = {
        "parse_ok": bool(parse_ok),
        "selected_route": selected,
        "raw_preview": content[:220].replace("\n", " "),
    }
    if selected in option_by_id:
        return option_by_id[selected], meta, content
    raise ValueError(f"LLM selected invalid route id: {selected!r}; raw={content[:200]!r}")



def _final_success_radius(base, args) -> float:
    """Use one consistent, explicit final-goal tolerance for all methods.

    The waypoint wrapper internally uses a tighter base.success_radius. For route-following
    experiments the robot often reaches the final waypoint within the subgoal radius but
    remains just outside the old 0.75m terminal threshold, which incorrectly labels
    near-arrivals as timeouts. This radius is applied equally to direct RL and all route
    methods for fair comparison.
    """
    configured = float(getattr(args, "final_success_radius", 0.0) or 0.0)
    if configured > 0:
        return configured
    return max(float(getattr(base, "success_radius", 0.75)), float(getattr(args, "subgoal_radius", 0.85)))


def _run_direct_metrics(env, model, obs, args) -> dict[str, Any]:
    episode_reward = 0.0
    last_info: dict[str, Any] = {}
    collided = False
    steps_used = 0
    positions: list[tuple[float, float]] = []
    for _ in range(args.max_steps):
        base = _base_env(env)
        positions.append((float(base.robot_x), float(base.robot_y)))
        action = _predict_action(model, obs)
        result = env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        episode_reward += float(reward)
        last_info = dict(info)
        steps_used += 1
        collided = bool(last_info.get("collided", False))
        if done:
            break
    base = _base_env(env)
    positions.append((float(base.robot_x), float(base.robot_y)))
    final_dist = math.hypot(base.robot_x - base.goal_x, base.robot_y - base.goal_y)
    reached = final_dist <= _final_success_radius(base, args) and not collided
    if reached:
        outcome = "success"
    elif collided:
        outcome = "collision"
    else:
        outcome = "timeout"
    return {
        "outcome": outcome,
        "reward": episode_reward,
        "final_dist": final_dist,
        "steps": steps_used,
        "collided": collided,
        "last_info": last_info,
        "trajectory_length": _trajectory_length(positions),
        "trajectory_turns": _turn_count(positions, min_angle_deg=25.0),
        "trajectory_positions": positions,
        "shield_interventions": int(last_info.get("shield_interventions", 0) or 0),
        "subgoals_reached": 0,
    }


def _execute_route_metrics(env, model, route: list[tuple[float, float]], final_goal: tuple[float, float], args) -> dict[str, Any]:
    episode_reward = 0.0
    last_info: dict[str, Any] = {}
    collided = False
    steps_used = 0
    subgoals_reached = 0
    if not route:
        route = [final_goal]
    positions: list[tuple[float, float]] = []
    for subgoal in route:
        _set_goal(env, subgoal)
        obs = _current_obs(env)
        while steps_used < args.max_steps:
            base = _base_env(env)
            positions.append((float(base.robot_x), float(base.robot_y)))
            action = _predict_action(model, obs)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            episode_reward += float(reward)
            last_info = dict(info)
            steps_used += 1
            base = _base_env(env)
            collided = bool(last_info.get("collided", False))
            if collided:
                break
            reach_radius = float(getattr(args, "subgoal_radius", base.success_radius))
            if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) <= reach_radius:
                subgoals_reached += 1
                break
            if done:
                break
        if collided or steps_used >= args.max_steps:
            break
        base = _base_env(env)
        reach_radius = float(getattr(args, "subgoal_radius", base.success_radius))
        if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) > reach_radius:
            break
    # Restore final goal for final-distance accounting and for any downstream wrappers.
    _set_goal(env, final_goal)
    base = _base_env(env)
    positions.append((float(base.robot_x), float(base.robot_y)))
    final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
    reached = final_dist <= _final_success_radius(base, args) and not collided
    if reached:
        outcome = "success"
    elif collided:
        outcome = "collision"
    else:
        outcome = "timeout"
    return {
        "outcome": outcome,
        "reward": episode_reward,
        "final_dist": final_dist,
        "steps": steps_used,
        "collided": collided,
        "last_info": last_info,
        "trajectory_length": _trajectory_length(positions),
        "trajectory_turns": _turn_count(positions, min_angle_deg=25.0),
        "trajectory_positions": positions,
        "shield_interventions": int(last_info.get("shield_interventions", 0) or 0),
        "subgoals_reached": subgoals_reached,
    }


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Paper A proactive fast-safe route decomposition evaluator.")
    parser.add_argument("--mode", choices=("direct", "classical_waypoint", "random_waypoint", "rule_semantic", "fast_safe_route", "llm_route"), required=True)
    parser.add_argument("--algo", choices=("ppo", "sac", "a2c", "dqn"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection", choices=MAP_SOURCE_CHOICES)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--subgoal-radius", type=float, default=0.85, help="Radius used to mark intermediate route waypoints as reached.")
    parser.add_argument("--final-success-radius", type=float, default=0.90, help="Final-goal success tolerance used consistently for direct and route methods.")
    parser.add_argument("--reward-profile", default="v8_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=2.0)
    parser.add_argument("--goal-max-distance", type=float, default=20.0)
    parser.add_argument("--goal-point-probability", type=float, default=0.95)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--shield-min-clearance", type=float, default=0.18)
    parser.add_argument("--shield-intervention-penalty", type=float, default=0.25)
    parser.add_argument("--waypoint-spacing", type=float, default=2.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.55)
    parser.add_argument("--max-segment", type=float, default=6.2)
    parser.add_argument("--lm-studio-url", default=os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lm-model", default=os.environ.get("LM_STUDIO_MODEL", "liquid/lfm2.5-1.2b"))
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=650)
    parser.add_argument("--repair-invalid-route", action="store_true")
    parser.add_argument("--execution-aware-rerank", action="store_true", help="After parsing the LLM response, choose the lowest execution-score route among valid candidates. This implements the fast-safe validator for Ours.")
    parser.add_argument("--fast-safe-clearance", type=float, default=0.75)
    parser.add_argument("--turn-weight", type=float, default=1.8)
    parser.add_argument("--clearance-weight", type=float, default=7.0)
    parser.add_argument("--waypoint-count-weight", type=float, default=0.35)
    parser.add_argument("--semantic-cost-weight", type=float, default=5.0)
    parser.add_argument("--safe-success-threshold", type=float, default=0.75)
    args = parser.parse_args()

    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    model_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}[args.algo]
    model = model_cls.load(args.model)
    cases = _load_cases(Path(args.cases))[: args.episodes]
    rng = random.Random(args.seed)

    success = collision = timeout = invalid_route = repaired_route = 0
    strict_plan_valid = repairable_plan = plan_calls = 0
    total_reward = 0.0
    total_route_len = 0
    total_route_distance = 0.0
    total_semantic_cost = 0.0  # executed-trajectory semantic exposure; kept for summary compatibility
    total_planned_semantic_cost = 0.0
    total_trajectory_semantic_cost = 0.0
    total_execution_score = 0.0
    total_route_turns = 0.0
    total_route_clearance_penalty = 0.0
    total_steps = 0
    total_final_dist = 0.0
    total_path_len = 0.0
    total_traj_turns = 0.0
    total_shield_interventions = 0
    total_subgoals_reached = 0
    semantic_cases = 0
    constraint_success = 0
    safe_success = 0
    success_steps: list[int] = []
    success_path_lengths: list[float] = []
    llm_overridden_count = 0
    llm_raw_used_count = 0

    print(f"Paper A proactive evaluation mode={args.mode} algo={args.algo}")
    print(f"model_file={args.model}")
    print(f"cases={args.cases} n={len(cases)}")
    print(
        f"fast_safe_params turn_weight={args.turn_weight} clearance_weight={args.clearance_weight} "
        f"waypoint_count_weight={args.waypoint_count_weight} semantic_cost_weight={args.semantic_cost_weight} "
        f"safe_success_threshold={args.safe_success_threshold} subgoal_radius={args.subgoal_radius} final_success_radius={args.final_success_radius}"
    )
    if args.mode == "llm_route":
        print(f"llm_model={args.lm_model} repair_invalid_route={int(args.repair_invalid_route)} execution_aware_rerank={int(args.execution_aware_rerank)}")

    for episode, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{episode:03d}")
        scenario = str(case.get("scenario") or "long_horizon")
        map_id = str(case.get("map_id") or args.maps[0])
        env = _make_env(args, map_id, args.seed + episode)
        obs, start, final_goal, map_id = _apply_case_reset(env, case, args.seed + episode)
        base = _base_env(env)
        route: list[tuple[float, float]] = []
        route_id = ""
        plan_valid = 1
        repaired = 0
        fallback = 0
        parse_ok = 1
        selected_valid = 1
        llm_raw_route_id = ""
        route_overridden_by_scorer = 0
        route_semantic_cost = 0.0
        trajectory_semantic_cost = 0.0
        route_distance = 0.0
        route_turns = 0
        route_clearance_penalty = 0.0
        route_execution_score = 0.0
        geometry_valid = True

        if args.mode == "direct":
            result = _run_direct_metrics(env, model, obs, args)
        else:
            selected: dict[str, Any] | None = None
            try:
                options = _route_options(base, case, args)
                if args.mode == "classical_waypoint":
                    selected = next((item for item in options if item["option_id"] == "shortest"), _select_rule_option(options, scenario))
                    route_id = str(selected["option_id"])
                elif args.mode == "random_waypoint":
                    selected = _select_random_option(options, rng)
                    route_id = str(selected["option_id"])
                elif args.mode == "rule_semantic":
                    selected = _select_rule_option(options, scenario)
                    route_id = str(selected["option_id"])
                elif args.mode == "fast_safe_route":
                    selected = _select_fast_safe_option(options, args)
                    route_id = str(selected["option_id"])
                else:
                    plan_calls += 1
                    try:
                        selected, meta, _raw = _request_llm_option(
                            options,
                            case,
                            args.lm_studio_url,
                            args.lm_model,
                            args.llm_timeout_s,
                            args.llm_max_tokens,
                        )
                        parse_ok = int(bool(meta.get("parse_ok", False)))
                        llm_raw_route_id = str(selected["option_id"])
                        route_id = llm_raw_route_id
                        if args.execution_aware_rerank:
                            # The fast-safe validator can override a noisy LLM choice with the
                            # lowest-score validated candidate. We log both the raw LLM choice
                            # and the final selected route so the LLM/scorer contribution is auditable.
                            selected_by_scorer = _select_fast_safe_option(options, args)
                            if str(selected_by_scorer["option_id"]) != llm_raw_route_id:
                                route_overridden_by_scorer = 1
                            selected = selected_by_scorer
                            route_id = str(selected["option_id"])
                    except Exception as exc:
                        selected_valid = 0
                        parse_ok = 0
                        fallback = 1
                        if args.repair_invalid_route:
                            selected = _select_fast_safe_option(options, args) if args.execution_aware_rerank else _select_rule_option(options, scenario)
                            route_id = str(selected["option_id"])
                            repaired = 1
                        else:
                            invalid_route += 1
                            result = {
                                "outcome": "invalid_route",
                                "reward": 0.0,
                                "final_dist": math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]),
                                "steps": 0,
                                "collided": False,
                                "trajectory_length": 0.0,
                                "trajectory_turns": 0,
                                "trajectory_positions": [],
                                "shield_interventions": 0,
                                "subgoals_reached": 0,
                            }
                            print(f"episode={episode:03d} case_id={case_id} llm_plan_failed={str(exc)[:220]}")
                    if selected_valid and parse_ok:
                        strict_plan_valid += 1
                    if selected_valid or args.repair_invalid_route:
                        repairable_plan += 1
                if args.mode == "llm_route" and selected_valid == 0 and not args.repair_invalid_route:
                    pass
                else:
                    if selected is None:
                        raise ValueError("No selected route to execute")
                    route = list(selected["route"])
                    route, geometry_valid = _sanitize_route(base, route, final_goal, max_segment=args.max_segment)
                    plan_valid = int(bool(geometry_valid and route))
                    if not plan_valid:
                        if args.mode == "llm_route" and args.repair_invalid_route:
                            repaired = 1
                            selected = _select_fast_safe_option(options, args) if args.execution_aware_rerank else _select_rule_option(options, scenario)
                            route_id = str(selected["option_id"])
                            route = list(selected["route"])
                            route, geometry_valid = _sanitize_route(base, route, final_goal, max_segment=args.max_segment)
                            plan_valid = int(bool(geometry_valid and route))
                        if not plan_valid:
                            invalid_route += 1
                            result = {
                                "outcome": "invalid_route",
                                "reward": 0.0,
                                "final_dist": math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]),
                                "steps": 0,
                                "collided": False,
                                "trajectory_length": 0.0,
                                "trajectory_turns": 0,
                                "trajectory_positions": [],
                                "shield_interventions": 0,
                                "subgoals_reached": 0,
                            }
                        else:
                            result = _execute_route_metrics(env, model, route, final_goal, args)
                    else:
                        if repaired:
                            repaired_route += 1
                        result = _execute_route_metrics(env, model, route, final_goal, args)
                    route_distance = _route_length(route, start)
                    risk_center = (_float(case, "risk_center_x"), _float(case, "risk_center_y"))
                    risk_radius = _float(case, "risk_radius", 0.0)
                    route_semantic_cost = _semantic_cost(route, risk_center, risk_radius) if scenario == "semantic_constraint" else 0.0
                    route_turns = _turn_count([start] + route)
                    _min_clearance, route_clearance_penalty = _route_clearance_stats(base, route, start, target_clearance=args.fast_safe_clearance)
                    route_execution_score = _execution_score(
                        {
                            "approx_length": route_distance,
                            "turn_count": route_turns,
                            "clearance_penalty": route_clearance_penalty,
                            "waypoint_count": len(route),
                            "semantic_cost": route_semantic_cost,
                        },
                        args,
                    )
            except Exception as exc:
                invalid_route += 1
                result = {
                    "outcome": "invalid_route",
                    "reward": 0.0,
                    "final_dist": math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]),
                    "steps": 0,
                    "collided": False,
                    "trajectory_length": 0.0,
                    "trajectory_turns": 0,
                    "trajectory_positions": [],
                    "shield_interventions": 0,
                    "subgoals_reached": 0,
                }
                plan_valid = 0
                parse_ok = 0
                fallback = 1
                print(f"episode={episode:03d} case_id={case_id} plan_failed={str(exc)[:220]}")

        outcome = str(result["outcome"])
        reward = float(result["reward"])
        final_dist = float(result["final_dist"])
        steps_used = int(result["steps"])
        collided = bool(result.get("collided", False))
        trajectory_length = float(result.get("trajectory_length", 0.0))
        trajectory_turns = int(result.get("trajectory_turns", 0) or 0)
        shield_interventions = int(result.get("shield_interventions", 0) or 0)
        subgoals_reached = int(result.get("subgoals_reached", 0) or 0)
        trajectory_positions = result.get("trajectory_positions", []) or []
        if scenario == "semantic_constraint":
            risk_center_for_traj = (_float(case, "risk_center_x"), _float(case, "risk_center_y"))
            risk_radius_for_traj = _float(case, "risk_radius", 0.0)
            trajectory_semantic_cost = _trajectory_semantic_cost(trajectory_positions, risk_center_for_traj, risk_radius_for_traj)

        if outcome == "success":
            success += 1
            success_steps.append(steps_used)
            success_path_lengths.append(trajectory_length)
        elif outcome == "collision":
            collision += 1
        elif outcome == "timeout":
            timeout += 1
        total_reward += float(reward)
        total_route_len += len(route)
        total_route_distance += float(route_distance)
        total_execution_score += float(route_execution_score)
        total_route_turns += float(route_turns)
        total_route_clearance_penalty += float(route_clearance_penalty)
        total_steps += steps_used
        total_final_dist += final_dist
        total_path_len += trajectory_length
        total_traj_turns += trajectory_turns
        total_shield_interventions += shield_interventions
        total_subgoals_reached += subgoals_reached
        if scenario == "semantic_constraint":
            semantic_cases += 1
            total_planned_semantic_cost += float(route_semantic_cost)
            total_trajectory_semantic_cost += float(trajectory_semantic_cost)
            total_semantic_cost += float(trajectory_semantic_cost)
            constraint_ok = trajectory_semantic_cost <= args.safe_success_threshold
            if constraint_ok:
                constraint_success += 1
            if outcome == "success" and constraint_ok:
                safe_success += 1
        if args.mode == "llm_route" and llm_raw_route_id:
            if route_overridden_by_scorer:
                llm_overridden_count += 1
            else:
                llm_raw_used_count += 1

        print(
            f"episode={episode:03d} case_id={case_id} scenario={scenario} map={map_id} mode={args.mode} algo={args.algo} "
            f"outcome={outcome:13s} reward={reward:8.2f} final_dist={final_dist:6.2f} steps={steps_used:04d} "
            f"path_len={trajectory_length:6.2f} traj_turns={trajectory_turns:03d} shield={shield_interventions:03d} "
            f"route_id={route_id or 'NA'} route_len={len(route):02d} route_distance={route_distance:6.2f} route_turns={route_turns:02d} "
            f"semantic_cost={trajectory_semantic_cost:5.3f} planned_semantic_cost={route_semantic_cost:5.3f} "
            f"exec_score={route_execution_score:7.2f} clearance_penalty={route_clearance_penalty:5.3f} "
            f"subgoals_reached={subgoals_reached:02d} plan_valid={plan_valid} repaired={repaired} "
            f"geometry_valid={int(geometry_valid)} fallback={fallback} parse_ok={parse_ok} selected_valid={selected_valid} "
            f"llm_raw_route={llm_raw_route_id or 'NA'} route_overridden={route_overridden_by_scorer}"
        )

    n = max(1, len(cases))
    mean_steps_success = statistics.mean(success_steps) if success_steps else 0.0
    median_steps_success = statistics.median(success_steps) if success_steps else 0.0
    mean_path_success = statistics.mean(success_path_lengths) if success_path_lengths else 0.0
    print("--- overall summary ---")
    print(f"episodes: {len(cases)}")
    print(f"success_rate: {success / n:.3f}")
    print(f"collision_rate: {collision / n:.3f}")
    print(f"timeout_rate: {timeout / n:.3f}")
    print(f"invalid_route_rate: {invalid_route / n:.3f}")
    print(f"repaired_route_rate: {repaired_route / n:.3f}")
    print(f"strict_llm_plan_valid_rate: {strict_plan_valid / max(plan_calls, 1):.3f}")
    print(f"repairable_llm_plan_rate: {repairable_plan / max(plan_calls, 1):.3f}")
    print(f"mean_reward: {total_reward / n:.3f}")
    print(f"mean_steps_all: {total_steps / n:.3f}")
    print(f"mean_steps_success: {mean_steps_success:.3f}")
    print(f"median_steps_success: {median_steps_success:.3f}")
    print(f"mean_final_distance: {total_final_dist / n:.3f}")
    print(f"mean_path_length_all: {total_path_len / n:.3f}")
    print(f"mean_path_length_success: {mean_path_success:.3f}")
    print(f"mean_trajectory_turns: {total_traj_turns / n:.3f}")
    print(f"mean_shield_interventions: {total_shield_interventions / n:.3f}")
    print(f"mean_subgoals_reached: {total_subgoals_reached / n:.3f}")
    print(f"mean_route_len: {total_route_len / n:.3f}")
    print(f"mean_route_distance: {total_route_distance / n:.3f}")
    print(f"mean_route_turns: {total_route_turns / n:.3f}")
    print(f"mean_route_clearance_penalty: {total_route_clearance_penalty / n:.3f}")
    print(f"mean_route_execution_score: {total_execution_score / n:.3f}")
    print(f"mean_planned_semantic_cost: {total_planned_semantic_cost / max(semantic_cases, 1):.3f}")
    print(f"mean_trajectory_semantic_cost: {total_trajectory_semantic_cost / max(semantic_cases, 1):.3f}")
    print(f"mean_semantic_cost: {total_semantic_cost / max(semantic_cases, 1):.3f}")
    print(f"llm_raw_used_rate: {llm_raw_used_count / max(plan_calls, 1):.3f}")
    print(f"llm_overridden_rate: {llm_overridden_count / max(plan_calls, 1):.3f}")
    print(f"constraint_success_rate: {constraint_success / max(semantic_cases, 1):.3f}")
    print(f"safe_success_rate: {safe_success / max(semantic_cases, 1):.3f}")


if __name__ == "__main__":
    main()
