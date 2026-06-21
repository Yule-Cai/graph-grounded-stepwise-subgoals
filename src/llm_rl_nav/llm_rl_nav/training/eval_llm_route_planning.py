from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np

from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, build_nav_env
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.train_multimap_ppo import parse_maps
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def _json_from_text(text: str) -> Any:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start_obj = cleaned.find("{")
    end_obj = cleaned.rfind("}")
    start_arr = cleaned.find("[")
    end_arr = cleaned.rfind("]")
    if start_obj >= 0 and end_obj > start_obj and (start_arr < 0 or start_obj < start_arr):
        snippet = cleaned[start_obj : end_obj + 1]
    elif start_arr >= 0 and end_arr > start_arr:
        snippet = cleaned[start_arr : end_arr + 1]
    else:
        raise ValueError("LLM did not return JSON")
    snippet = re.sub(r",\s*([}\]])", r"\1", snippet)
    return json.loads(snippet)


def _normalize_waypoints(parsed: Any) -> list[tuple[float, float]]:
    if isinstance(parsed, dict):
        raw = parsed.get("waypoints") or parsed.get("route") or parsed.get("path") or []
    elif isinstance(parsed, list):
        raw = parsed
    else:
        raw = []
    waypoints: list[tuple[float, float]] = []
    for item in raw[:18]:
        if isinstance(item, dict):
            x, y = item.get("x"), item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            continue
        try:
            waypoints.append((float(x), float(y)))
        except (TypeError, ValueError):
            continue
    return waypoints


def _candidate_route(env, final_goal: tuple[float, float], spacing: float = 2.2, resolution: float = 0.55) -> list[tuple[float, float]]:
    planner = WaypointGoalEnv(env, waypoint_spacing=spacing, grid_resolution=resolution)
    start = (float(env.robot_x), float(env.robot_y))
    route = planner._plan_waypoints(start, final_goal)
    return [(round(float(x), 2), round(float(y), 2)) for x, y in route[:28]]


def _normalize_indices(parsed: Any) -> list[int]:
    raw: Any
    if isinstance(parsed, dict):
        raw = (
            parsed.get("route")
            or parsed.get("node_route")
            or parsed.get("indices")
            or parsed.get("route_indices")
            or parsed.get("selected_indices")
            or []
        )
    elif isinstance(parsed, list):
        raw = parsed
    else:
        raw = []
    indices: list[int] = []
    if isinstance(raw, list):
        for item in raw:
            try:
                indices.append(int(item))
            except (TypeError, ValueError):
                continue
    return indices


def _expand_route_indices(indices: list[int], candidate_count: int) -> list[int]:
    if candidate_count <= 0:
        return []
    cleaned: list[int] = []
    for index in indices:
        if 0 <= index < candidate_count and index not in cleaned:
            cleaned.append(index)
    if not cleaned:
        return []
    if cleaned[0] != 0:
        cleaned.insert(0, 0)
    if cleaned[-1] != candidate_count - 1:
        cleaned.append(candidate_count - 1)
    expanded: list[int] = []
    cursor = cleaned[0]
    expanded.append(cursor)
    for target in cleaned[1:]:
        step = 1 if target >= cursor else -1
        for value in range(cursor + step, target + step, step):
            if 0 <= value < candidate_count and value not in expanded:
                expanded.append(value)
        cursor = target
    return expanded


def _salvage_indices_from_text(text: str) -> list[int]:
    indices: list[int] = []
    for match in re.finditer(r"-?\d+", text):
        value = int(match.group(0))
        if value not in indices:
            indices.append(value)
        if len(indices) >= 18:
            break
    return indices


def _salvage_waypoints_from_text(text: str) -> list[tuple[float, float]]:
    waypoints: list[tuple[float, float]] = []
    patterns = [
        r'"x"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"y"\s*:\s*(-?\d+(?:\.\d+)?)',
        r"'x'\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*'y'\s*:\s*(-?\d+(?:\.\d+)?)",
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)",
    ]
    seen: set[tuple[float, float]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                point = (round(float(match.group(1)), 2), round(float(match.group(2)), 2))
            except (TypeError, ValueError):
                continue
            if point not in seen:
                seen.add(point)
                waypoints.append(point)
            if len(waypoints) >= 18:
                return waypoints
    return waypoints


def _line_safe(env, a: tuple[float, float], b: tuple[float, float], step: float = 0.25) -> bool:
    dist = math.hypot(b[0] - a[0], b[1] - a[1])
    steps = max(2, int(dist / step))
    for index in range(steps + 1):
        t = index / steps
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t
        if env._in_collision(x, y):
            return False
    return True


def _grid_payload(env, resolution: float = 1.2, max_free: int = 260, max_occ: int = 180) -> dict[str, Any]:
    free: list[list[float]] = []
    occupied: list[list[float]] = []
    xmin, xmax = env.x_bounds
    ymin, ymax = env.y_bounds
    xs = np.arange(xmin, xmax + 1e-6, resolution)
    ys = np.arange(ymin, ymax + 1e-6, resolution)
    for x in xs:
        for y in ys:
            point = [round(float(x), 2), round(float(y), 2)]
            if env._in_collision(float(x), float(y)):
                if len(occupied) < max_occ:
                    occupied.append(point)
            elif len(free) < max_free:
                free.append(point)
    return {
        "cell_size_m": resolution,
        "free_cells": free,
        "occupied_cells": occupied,
    }


def _semantic_entities(env, limit: int = 35) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for obs in env.obstacles[:limit]:
        entities.append(
            {
                "entity_id": obs.name,
                "name": obs.name.replace("_", " "),
                "type": "furniture_or_wall",
                "geometry": {
                    "type": "bbox",
                    "center": [round(obs.center[0], 2), round(obs.center[1], 2)],
                    "size": [round(obs.size[0], 2), round(obs.size[1], 2)],
                },
            }
        )
    return entities


def _nearby_entities(env, limit: int = 16) -> list[dict[str, Any]]:
    robot = (float(env.robot_x), float(env.robot_y))
    ranked = sorted(
        env.obstacles,
        key=lambda obs: math.hypot(obs.center[0] - robot[0], obs.center[1] - robot[1]),
    )
    entities: list[dict[str, Any]] = []
    for obs in ranked[:limit]:
        entities.append(
            {
                "name": obs.name,
                "center": [round(obs.center[0], 2), round(obs.center[1], 2)],
                "size": [round(obs.size[0], 2), round(obs.size[1], 2)],
            }
        )
    return entities


def _request_llm_route(
    env,
    base_url: str,
    model: str,
    timeout_s: float,
    max_tokens: int,
    grid_resolution: float,
    max_free_cells: int,
    max_occupied_cells: int,
) -> tuple[list[tuple[float, float]], str, dict[str, Any]]:
    robot = {"x": round(env.robot_x, 2), "y": round(env.robot_y, 2), "yaw": round(env.robot_yaw, 3)}
    goal = {"x": round(env.goal_x, 2), "y": round(env.goal_y, 2)}
    final_goal = (float(env.goal_x), float(env.goal_y))
    candidates = _candidate_route(env, final_goal, spacing=2.2, resolution=0.55)
    route_distance = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
    required_waypoints = max(3, min(len(candidates), int(route_distance / 3.2) + 2)) if candidates else 0
    use_compact_payload = max_free_cells <= 0 or max_occupied_cells <= 0
    nodes = [
        {"id": index, "x": point[0], "y": point[1]}
        for index, point in enumerate(candidates)
    ]
    edges = [[index, index + 1] for index in range(max(0, len(candidates) - 1))]
    payload = {
        "map_id": env.active_map_id,
        "robot_pose": robot,
        "final_goal": goal,
        "route_distance_estimate_m": round(route_distance, 2),
        "required_waypoints": required_waypoints,
        "start_node_id": 0 if nodes else None,
        "final_node_id": len(nodes) - 1 if nodes else None,
        "navigation_graph": {"nodes": nodes, "edges": edges},
        "nearby_obstacles": _nearby_entities(env),
        "instruction": (
            "Select a complete route through navigation_graph node ids. "
            "Do not invent coordinates. Preserve edge order. "
            "The route must start with start_node_id and end with final_node_id."
        ),
    }
    if not use_compact_payload:
        payload["belief_map"] = _grid_payload(
            env,
            resolution=grid_resolution,
            max_free=max_free_cells,
            max_occ=max_occupied_cells,
        )
    prompt = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "\n".join(
                    [
                        "You are a high-level route planner for an indoor robot.",
                        "Return compact JSON only. Do not use markdown.",
                        'Schema: {"route":[0,1,2],"warnings":[]}',
                        "Choose only node ids from navigation_graph.nodes.",
                        "Follow navigation_graph.edges.",
                        "The first route id must be start_node_id. The last route id must be final_node_id.",
                        "A partial prefix is invalid.",
                    ]
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "temperature": 0.05,
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
    fallback_used = False
    try:
        parsed = _json_from_text(content)
        indices = _normalize_indices(parsed)
    except Exception:
        parse_ok = False
        indices = _salvage_indices_from_text(content)
    selected_indices: list[int] = []
    for index in indices:
        if 0 <= index < len(candidates) and index not in selected_indices:
            selected_indices.append(index)
    model_selected_indices = list(selected_indices)
    final_index = len(candidates) - 1
    model_included_final = final_index in model_selected_indices if final_index >= 0 else False
    if not selected_indices:
        fallback_used = True
        selected_indices = list(range(len(candidates)))
    expanded_indices = _expand_route_indices(selected_indices, len(candidates))
    waypoints: list[tuple[float, float]] = []
    for index in expanded_indices:
        if 0 <= index < len(candidates):
            point = candidates[index]
            if point not in waypoints:
                waypoints.append(point)
    if candidates and (not waypoints or waypoints[-1] != candidates[-1]):
        waypoints.append(candidates[-1])
    if not waypoints:
        salvaged = _salvage_waypoints_from_text(content)
        if salvaged:
            fallback_used = True
        waypoints = salvaged or candidates
    if not waypoints:
        raise ValueError(f"LLM returned no usable route choice: {content[:300]}")
    meta = {
        "parse_ok": parse_ok,
        "fallback_used": fallback_used,
        "candidate_count": len(candidates),
        "model_selected_indices": model_selected_indices[:28],
        "selected_indices": selected_indices[:28],
        "expanded_indices": expanded_indices[:28],
        "model_selected_count": len(model_selected_indices),
        "selected_count": len(selected_indices),
        "expanded_count": len(expanded_indices),
        "model_included_final": model_included_final,
        "raw_preview": content[:220].replace("\n", " "),
    }
    return waypoints, content, meta


def _sanitize_route(
    env,
    raw_waypoints: list[tuple[float, float]],
    final_goal: tuple[float, float],
    max_segment: float = 5.8,
) -> tuple[list[tuple[float, float]], bool]:
    route: list[tuple[float, float]] = []
    cursor = (float(env.robot_x), float(env.robot_y))
    valid = True
    for point in raw_waypoints:
        if env._in_collision(point[0], point[1]):
            valid = False
            continue
        if math.hypot(point[0] - cursor[0], point[1] - cursor[1]) > max_segment:
            valid = False
        route.append(point)
        cursor = point
    if not route or math.hypot(route[-1][0] - final_goal[0], route[-1][1] - final_goal[1]) > 1.0:
        if not env._in_collision(final_goal[0], final_goal[1]) and _line_safe(env, cursor, final_goal):
            route.append(final_goal)
        else:
            valid = False
    return route[:28], valid


def _base_env(env):
    """Return the true underlying navigation env, not a gym wrapper.

    The previous version used ``hasattr(current, "_in_collision")`` to decide
    when to stop unwrapping. That is unsafe for wrappers such as
    ``ShieldedActionEnv`` because ``__getattr__`` delegates attributes to the
    wrapped env, making the wrapper appear to own ``_in_collision``. As a result,
    helper functions such as ``_set_goal`` wrote ``goal_x/goal_y`` onto the
    wrapper instead of the real env. Route/subgoal execution then silently used
    the original final goal, which made direct RL, classical waypoints, and LLM
    routes produce identical trajectories.

    Always unwrap through the explicit ``.env`` chain until there is no inner env.
    """
    current = env
    seen: set[int] = set()
    while True:
        obj_id = id(current)
        if obj_id in seen:
            break
        seen.add(obj_id)
        inner = getattr(current, "env", None)
        if inner is None or inner is current:
            break
        current = inner
    return current


def _set_goal(env, goal: tuple[float, float]) -> None:
    base = _base_env(env)
    base.goal_x = float(goal[0])
    base.goal_y = float(goal[1])
    base.prev_dist = base._distance_to_goal()
    base.best_dist = base.prev_dist
    base.no_progress_steps = 0
    base.near_wall_steps = 0


def _current_obs(env):
    return _base_env(env)._observation()


def _predict_action(model, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Evaluate SAC/PPO with LLM ahead route planning.")
    parser.add_argument("--algo", choices=("ppo", "sac"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection", choices=MAP_SOURCE_CHOICES)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--reward-profile", default="v8_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=2.0)
    parser.add_argument("--goal-max-distance", type=float, default=12.0)
    parser.add_argument("--goal-point-probability", type=float, default=0.95)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--shield-min-clearance", type=float, default=0.18)
    parser.add_argument("--shield-intervention-penalty", type=float, default=0.25)
    parser.add_argument("--lm-studio-url", default=os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lm-model", default=os.environ.get("LM_STUDIO_MODEL", "google/gemma-3-1b"))
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=650)
    parser.add_argument("--llm-grid-resolution", type=float, default=1.2)
    parser.add_argument("--llm-max-free-cells", type=int, default=260)
    parser.add_argument("--llm-max-occupied-cells", type=int, default=180)
    parser.add_argument(
        "--execute-invalid-route",
        action="store_true",
        help="Execute fallback/completed routes even when the LLM route is invalid. Use only for debugging.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    model_cls = PPO if args.algo == "ppo" else SAC
    model = model_cls.load(args.model)
    print(f"LLM route evaluation model={args.lm_model}")
    print(f"RL backend algo={args.algo} model_file={args.model}")

    total_successes = 0
    total_collisions = 0
    total_timeouts = 0
    total_invalid_routes = 0
    total_plan_valid = 0
    total_plan_calls = 0
    total_episodes = 0

    for map_index, map_id in enumerate(args.maps):
        successes = 0
        collisions = 0
        timeouts = 0
        invalid_routes = 0
        rewards = 0.0
        plan_valid = 0
        plan_calls = 0
        print(f"=== map={map_id} algo={args.algo}+llm ===")
        for episode in range(args.episodes):
            env = build_nav_env(
                args.map_source,
                seed=args.seed + map_index,
                map_id=map_id,
                max_steps=args.max_steps,
                reward_profile=args.reward_profile,
                goal_min_distance=args.goal_min_distance,
                goal_max_distance=args.goal_max_distance,
                goal_point_probability=args.goal_point_probability,
            )
            if args.safety_shield:
                env = ShieldedActionEnv(
                    env,
                    min_clearance=args.shield_min_clearance,
                    intervention_penalty=args.shield_intervention_penalty,
                )
            reset_result = env.reset(seed=args.seed + map_index * 1000 + episode, options={"map_id": map_id})
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            base = _base_env(env)
            final_goal = (float(base.goal_x), float(base.goal_y))
            route_meta: dict[str, Any] = {}
            try:
                raw_route, _raw, route_meta = _request_llm_route(
                    base,
                    args.lm_studio_url,
                    args.lm_model,
                    args.llm_timeout_s,
                    args.llm_max_tokens,
                    args.llm_grid_resolution,
                    args.llm_max_free_cells,
                    args.llm_max_occupied_cells,
                )
                plan_calls += 1
                route, geometry_valid = _sanitize_route(base, raw_route, final_goal)
                valid = (
                    geometry_valid
                    and bool(route_meta.get("parse_ok", False))
                    and not bool(route_meta.get("fallback_used", False))
                    and bool(route_meta.get("selected_indices"))
                    and bool(route_meta.get("model_included_final", False))
                )
                if valid:
                    plan_valid += 1
                if not route:
                    route = [final_goal]
            except Exception as exc:
                raw_route = []
                route = [final_goal]
                valid = False
                route_meta = {"error": str(exc), "fallback_used": True}
                print(f"episode={episode:03d} llm_plan_failed={exc}")

            if not valid and not args.execute_invalid_route:
                invalid_routes += 1
                print(
                    f"episode={episode:03d} outcome=invalid_route "
                    f"reward={0.0:8.2f} final_dist={math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]):6.2f} "
                    f"route_len={len(route):02d} raw_route_len={len(raw_route):02d} plan_valid=0 "
                    f"candidates={int(route_meta.get('candidate_count', 0)):02d} "
                    f"model_selected={route_meta.get('model_selected_indices', [])} "
                    f"exec_selected=[] expanded_count=0 "
                    f"model_final={int(bool(route_meta.get('model_included_final', False)))} "
                    f"fallback={int(bool(route_meta.get('fallback_used', False)))} "
                    f"parse_ok={int(bool(route_meta.get('parse_ok', False)))}"
                )
                continue

            episode_reward = 0.0
            last_info: dict[str, Any] = {}
            collided = False
            steps_used = 0
            reached_all = False
            for subgoal_index, subgoal in enumerate(route):
                _set_goal(env, subgoal)
                obs = _current_obs(env)
                while steps_used < args.max_steps:
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
                    if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) <= base.success_radius:
                        break
                    if done:
                        break
                if collided or steps_used >= args.max_steps:
                    break
                if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) > base.success_radius:
                    break
            base = _base_env(env)
            final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
            reached_all = final_dist <= base.success_radius and not collided
            if reached_all:
                successes += 1
                outcome = "success"
            elif collided:
                collisions += 1
                outcome = "collision"
            else:
                timeouts += 1
                outcome = "timeout"
            rewards += episode_reward
            print(
                f"episode={episode:03d} outcome={outcome:9s} "
                f"reward={episode_reward:8.2f} final_dist={final_dist:6.2f} "
                f"route_len={len(route):02d} raw_route_len={len(raw_route):02d} plan_valid={int(valid)} "
                f"candidates={int(route_meta.get('candidate_count', 0)):02d} "
                f"model_selected={route_meta.get('model_selected_indices', [])} "
                f"exec_selected={route_meta.get('selected_indices', [])} "
                f"expanded_count={int(route_meta.get('expanded_count', 0))} "
                f"model_final={int(bool(route_meta.get('model_included_final', False)))} "
                f"fallback={int(bool(route_meta.get('fallback_used', False)))} "
                f"parse_ok={int(bool(route_meta.get('parse_ok', False)))}"
            )
        total_successes += successes
        total_collisions += collisions
        total_timeouts += timeouts
        total_invalid_routes += invalid_routes
        total_plan_valid += plan_valid
        total_plan_calls += plan_calls
        total_episodes += args.episodes
        print(
            f"map_summary success={successes / args.episodes:.3f} "
            f"collision={collisions / args.episodes:.3f} "
            f"timeout={timeouts / args.episodes:.3f} "
            f"invalid_route={invalid_routes / args.episodes:.3f} "
            f"plan_valid={plan_valid / max(plan_calls, 1):.3f} "
            f"mean_reward={rewards / args.episodes:.3f}"
        )

    print("--- overall summary ---")
    print(f"episodes: {total_episodes}")
    print(f"success_rate: {total_successes / total_episodes:.3f}")
    print(f"collision_rate: {total_collisions / total_episodes:.3f}")
    print(f"timeout_rate: {total_timeouts / total_episodes:.3f}")
    print(f"invalid_route_rate: {total_invalid_routes / total_episodes:.3f}")
    print(f"llm_plan_valid_rate: {total_plan_valid / max(total_plan_calls, 1):.3f}")


if __name__ == "__main__":
    main()
