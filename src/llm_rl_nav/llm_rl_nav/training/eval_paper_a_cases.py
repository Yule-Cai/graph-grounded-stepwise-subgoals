from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

from llm_rl_nav.envs.action_wrappers import DiscreteActionWrapper
from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, build_nav_env
from llm_rl_nav.training.eval_llm_route_planning import (
    _base_env,
    _candidate_route,
    _current_obs,
    _predict_action,
    _request_llm_route,
    _sanitize_route,
    _set_goal,
)
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.train_multimap_ppo import parse_maps
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cases.append(row)
    if not cases:
        raise SystemExit(f"No cases found in {path}")
    return cases


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_case_reset(env, case: dict[str, Any], seed: int):
    start = (_float(case, "start_x"), _float(case, "start_y"))
    goal = (_float(case, "goal_x"), _float(case, "goal_y"))
    yaw = _float(case, "yaw", 0.0)
    map_id = str(case.get("map_id") or "reference_family_flat")
    reset_result = env.reset(seed=seed, options={"map_id": map_id, "start": start, "goal": goal, "yaw": yaw})
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    return obs, start, goal, map_id


def _make_env(args, map_id: str, seed: int):
    env = build_nav_env(
        args.map_source,
        seed=seed,
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
    if args.algo == "dqn":
        env = DiscreteActionWrapper(env)
    return env


def _run_direct(env, model, obs, args) -> tuple[str, float, float, int, bool, dict[str, Any]]:
    episode_reward = 0.0
    last_info: dict[str, Any] = {}
    collided = False
    steps_used = 0
    for _ in range(args.max_steps):
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
    final_dist = math.hypot(base.robot_x - base.goal_x, base.robot_y - base.goal_y)
    reached = final_dist <= base.success_radius and not collided
    if reached:
        outcome = "success"
    elif collided:
        outcome = "collision"
    else:
        outcome = "timeout"
    return outcome, episode_reward, final_dist, steps_used, collided, last_info


def _execute_route(env, model, route: list[tuple[float, float]], final_goal: tuple[float, float], args) -> tuple[str, float, float, int, bool, dict[str, Any]]:
    episode_reward = 0.0
    last_info: dict[str, Any] = {}
    collided = False
    steps_used = 0
    if not route:
        route = [final_goal]
    for subgoal in route:
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
        base = _base_env(env)
        if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) > base.success_radius:
            break
    base = _base_env(env)
    final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
    reached = final_dist <= base.success_radius and not collided
    if reached:
        outcome = "success"
    elif collided:
        outcome = "collision"
    else:
        outcome = "timeout"
    return outcome, episode_reward, final_dist, steps_used, collided, last_info


def _strict_llm_valid(meta: dict[str, Any], geometry_valid: bool) -> bool:
    return (
        geometry_valid
        and bool(meta.get("parse_ok", False))
        and not bool(meta.get("fallback_used", False))
        and bool(meta.get("selected_indices"))
        and bool(meta.get("model_included_final", False))
    )


def _repair_llm_valid(meta: dict[str, Any], geometry_valid: bool, allow_fallback: bool) -> bool:
    if not geometry_valid:
        return False
    if bool(meta.get("fallback_used", False)) and not allow_fallback:
        return False
    # A route can be repaired when the model selected at least one in-graph node and we can
    # complete missing intermediate/final nodes using the candidate graph. This is the key
    # route-validity repair ablation for Paper A.
    return bool(meta.get("selected_indices")) or (allow_fallback and int(meta.get("candidate_count", 0)) > 0)


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Paper A fixed-case evaluator: direct RL, classical waypoint, or LLM route with repair accounting.")
    parser.add_argument("--mode", choices=("direct", "classical_waypoint", "llm_route"), required=True)
    parser.add_argument("--algo", choices=("ppo", "sac", "a2c", "dqn"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", required=True, help="CSV generated by generate_paper_a_hard_cases.py")
    parser.add_argument("--episodes", type=int, default=100, help="Maximum number of cases to evaluate.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection", choices=MAP_SOURCE_CHOICES)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--reward-profile", default="v8_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=2.0)
    parser.add_argument("--goal-max-distance", type=float, default=16.0)
    parser.add_argument("--goal-point-probability", type=float, default=0.95)
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--shield-min-clearance", type=float, default=0.18)
    parser.add_argument("--shield-intervention-penalty", type=float, default=0.25)
    parser.add_argument("--waypoint-spacing", type=float, default=2.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.55)
    parser.add_argument("--lm-studio-url", default=os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lm-model", default=os.environ.get("LM_STUDIO_MODEL", "liquid/lfm2.5-1.2b"))
    parser.add_argument("--llm-timeout-s", type=float, default=60.0)
    parser.add_argument("--llm-max-tokens", type=int, default=650)
    parser.add_argument("--llm-grid-resolution", type=float, default=1.2)
    parser.add_argument("--llm-max-free-cells", type=int, default=0)
    parser.add_argument("--llm-max-occupied-cells", type=int, default=0)
    parser.add_argument("--repair-invalid-route", action="store_true", help="Execute graph-completed LLM route when strict LLM output is incomplete but repairable.")
    parser.add_argument("--allow-fallback-repair", action="store_true", help="Also execute full candidate route when the LLM output is unusable. Keep off for main paper validity accounting.")
    args = parser.parse_args()

    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    model_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}[args.algo]
    model = model_cls.load(args.model)
    cases = _load_cases(Path(args.cases))[: args.episodes]

    success = collision = timeout = invalid_route = repaired_route = 0
    strict_plan_valid = repairable_plan = plan_calls = 0
    total_reward = 0.0

    print(f"Paper A fixed-case evaluation mode={args.mode} algo={args.algo}")
    print(f"model_file={args.model}")
    print(f"cases={args.cases} n={len(cases)}")
    if args.mode == "llm_route":
        print(f"llm_model={args.lm_model} repair_invalid_route={int(args.repair_invalid_route)} allow_fallback_repair={int(args.allow_fallback_repair)}")

    for episode, case in enumerate(cases):
        case_id = str(case.get("case_id") or f"case_{episode:03d}")
        map_id = str(case.get("map_id") or args.maps[0])
        env = _make_env(args, map_id, args.seed + episode)
        obs, start, final_goal, map_id = _apply_case_reset(env, case, args.seed + episode)
        base = _base_env(env)
        route: list[tuple[float, float]] = []
        route_len = 0
        raw_route_len = 0
        plan_valid = 1
        repaired = 0
        fallback = 0
        parse_ok = 1
        model_final = 1
        selected_count = 0
        candidate_count = 0
        geometry_valid = True

        if args.mode == "direct":
            outcome, reward, final_dist, steps_used, collided, _info = _run_direct(env, model, obs, args)
        elif args.mode == "classical_waypoint":
            candidate = _candidate_route(base, final_goal, spacing=args.waypoint_spacing, resolution=args.waypoint_grid_resolution)
            route, geometry_valid = _sanitize_route(base, candidate, final_goal)
            candidate_count = len(candidate)
            route_len = len(route)
            plan_valid = int(geometry_valid and bool(route))
            if not route:
                invalid_route += 1
                outcome, reward, final_dist, steps_used = "invalid_route", 0.0, math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]), 0
            else:
                outcome, reward, final_dist, steps_used, collided, _info = _execute_route(env, model, route, final_goal, args)
        else:
            try:
                raw_route, _raw, meta = _request_llm_route(
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
                strict_valid = _strict_llm_valid(meta, geometry_valid)
                repair_valid = _repair_llm_valid(meta, geometry_valid, args.allow_fallback_repair)
                if strict_valid:
                    strict_plan_valid += 1
                if repair_valid:
                    repairable_plan += 1
                repaired = int((not strict_valid) and repair_valid)
                fallback = int(bool(meta.get("fallback_used", False)))
                parse_ok = int(bool(meta.get("parse_ok", False)))
                model_final = int(bool(meta.get("model_included_final", False)))
                selected_count = int(meta.get("selected_count", 0) or 0)
                candidate_count = int(meta.get("candidate_count", 0) or 0)
                route_len = len(route)
                raw_route_len = len(raw_route)
                plan_valid = int(strict_valid)
                if strict_valid or (args.repair_invalid_route and repair_valid):
                    if repaired:
                        repaired_route += 1
                    outcome, reward, final_dist, steps_used, collided, _info = _execute_route(env, model, route, final_goal, args)
                else:
                    invalid_route += 1
                    outcome = "invalid_route"
                    reward = 0.0
                    final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
                    steps_used = 0
            except Exception as exc:
                invalid_route += 1
                outcome = "invalid_route"
                reward = 0.0
                final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
                steps_used = 0
                plan_valid = 0
                parse_ok = 0
                fallback = 1
                print(f"episode={episode:03d} case_id={case_id} llm_plan_failed={str(exc)[:220]}")

        if outcome == "success":
            success += 1
        elif outcome == "collision":
            collision += 1
        elif outcome == "timeout":
            timeout += 1
        # invalid_route already counted where applicable; direct/classical invalids rare.
        total_reward += float(reward)

        print(
            f"episode={episode:03d} case_id={case_id} map={map_id} mode={args.mode} algo={args.algo} "
            f"outcome={outcome:13s} reward={reward:8.2f} final_dist={final_dist:6.2f} steps={steps_used:04d} "
            f"start=({start[0]:.2f},{start[1]:.2f}) goal=({final_goal[0]:.2f},{final_goal[1]:.2f}) "
            f"route_len={route_len:02d} raw_route_len={raw_route_len:02d} candidates={candidate_count:02d} "
            f"plan_valid={plan_valid} repaired={repaired} geometry_valid={int(geometry_valid)} "
            f"fallback={fallback} parse_ok={parse_ok} model_final={model_final} selected_count={selected_count}"
        )

    n = max(1, len(cases))
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


if __name__ == "__main__":
    main()
