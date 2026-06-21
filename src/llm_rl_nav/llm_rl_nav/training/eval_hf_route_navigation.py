from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Any

from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, build_nav_env
from llm_rl_nav.training.eval_hf_route_compilers import (
    _config,
    _generate,
    _is_generative,
    _model_dirs,
    _params_label,
    _parse_route,
    _prompt,
    _safe_name,
)
from llm_rl_nav.training.eval_llm_route_planning import (
    _base_env,
    _candidate_route,
    _current_obs,
    _expand_route_indices,
    _predict_action,
    _sanitize_route,
    _set_goal,
)
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.train_multimap_ppo import parse_maps


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def _indices_to_waypoints(candidates: list[tuple[float, float]], indices: list[int]) -> list[tuple[float, float]]:
    waypoints: list[tuple[float, float]] = []
    for index in indices:
        if 0 <= index < len(candidates):
            point = candidates[index]
            if point not in waypoints:
                waypoints.append(point)
    if candidates and (not waypoints or waypoints[-1] != candidates[-1]):
        waypoints.append(candidates[-1])
    return waypoints


def _load_policy(algo: str, model_path: str):
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc
    return (PPO if algo == "ppo" else SAC).load(model_path)


def main() -> None:
    root = project_root()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Evaluate local HF route compilers with PPO/SAC execution.")
    parser.add_argument("--algo", choices=("ppo", "sac"), required=True)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--maps", type=parse_maps, default=("reference_family_flat",))
    parser.add_argument("--map-source", default="gazebo_3d_projection", choices=MAP_SOURCE_CHOICES)
    parser.add_argument("--models", default=None, help="Comma-separated external_models/hf directory names.")
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--safety-shield", action="store_true")
    parser.add_argument("--shield-min-clearance", type=float, default=0.18)
    parser.add_argument("--shield-intervention-penalty", type=float, default=0.25)
    parser.add_argument(
        "--execute-invalid-route",
        action="store_true",
        help="Execute fallback/completed routes even when the HF model route is invalid. Use only for debugging.",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("transformers and torch are required for HF navigation evaluation.") from exc

    if args.device == "mps":
        device = "mps"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    policy = _load_policy(args.algo, args.policy_model)
    out_dir = root / "logs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"hf_route_navigation_{args.algo}_{args.episodes}ep_{stamp}.csv"
    log_path = out_dir / f"hf_route_navigation_{args.algo}_{args.episodes}ep_{stamp}.log"
    rows: list[dict[str, Any]] = []

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"RL backend algo={args.algo} model_file={args.policy_model}\n")
        for model_dir in _model_dirs(root, args.models):
            if not (model_dir / "config.json").exists():
                continue
            config = _config(model_dir)
            model_name = _safe_name(model_dir)
            model_type = config.get("model_type", "unknown")
            params = _params_label(config, model_dir)
            generative = _is_generative(config)
            log.write(f"=== model={model_name} type={model_type} params={params} generative={int(generative)} ===\n")
            log.flush()

            hf_model = None
            tokenizer = None
            load_error = ""
            if generative:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        local_files_only=True,
                        torch_dtype=torch.float16 if device == "mps" else torch.float32,
                        low_cpu_mem_usage=True,
                    )
                    hf_model.to(device)
                    hf_model.eval()
                except Exception as exc:
                    load_error = str(exc)
                    generative = False
                    log.write(f"load_error={load_error}\n")

            successes = 0
            collisions = 0
            timeouts = 0
            invalid_routes = 0
            plan_valid = 0
            parse_ok = 0
            final_ok = 0
            fallback_count = 0
            rewards = 0.0
            latency_total = 0.0
            episodes_run = 0

            for map_index, map_id in enumerate(args.maps):
                for episode in range(args.episodes):
                    env = build_nav_env(
                        args.map_source,
                        seed=args.seed + map_index,
                        map_id=map_id,
                        max_steps=args.max_steps,
                        reward_profile="v8_goal",
                        goal_min_distance=2.0,
                        goal_max_distance=12.0,
                        goal_point_probability=0.95,
                    )
                    if args.safety_shield:
                        env = ShieldedActionEnv(
                            env,
                            min_clearance=args.shield_min_clearance,
                            intervention_penalty=args.shield_intervention_penalty,
                        )
                    reset = env.reset(seed=args.seed + map_index * 1000 + episode, options={"map_id": map_id})
                    obs = reset[0] if isinstance(reset, tuple) else reset
                    base = _base_env(env)
                    final_goal = (float(base.goal_x), float(base.goal_y))
                    candidates = _candidate_route(base, final_goal, spacing=2.2, resolution=0.55)
                    if len(candidates) < 2:
                        continue
                    episodes_run += 1

                    selected: list[int] = []
                    parsed = False
                    reached_final = False
                    valid_plan = False
                    fallback = False
                    latency = 0.0
                    if generative and hf_model is not None and tokenizer is not None:
                        prompt = _prompt(
                            map_id,
                            (float(base.robot_x), float(base.robot_y)),
                            final_goal,
                            candidates,
                        )
                        started = time.time()
                        try:
                            text = _generate(hf_model, tokenizer, prompt, args.max_new_tokens)
                            latency = time.time() - started
                            selected, parsed, reached_final, valid_plan = _parse_route(text, len(candidates))
                        except Exception as exc:
                            text = f"generation_error={exc}"
                            latency = time.time() - started
                    else:
                        text = "skipped_non_generative"

                    if not selected:
                        fallback = True
                        exec_indices = list(range(len(candidates)))
                    else:
                        exec_indices = _expand_route_indices(selected, len(candidates))
                        if not exec_indices:
                            fallback = True
                            exec_indices = list(range(len(candidates)))
                    raw_route = _indices_to_waypoints(candidates, exec_indices)
                    route, geometry_valid = _sanitize_route(base, raw_route, final_goal)
                    counted_valid = bool(valid_plan and geometry_valid and not fallback)
                    plan_valid += int(counted_valid)
                    parse_ok += int(parsed)
                    final_ok += int(reached_final)
                    fallback_count += int(fallback)
                    latency_total += latency

                    if not counted_valid and not args.execute_invalid_route:
                        invalid_routes += 1
                        log.write(
                            f"episode={episode:03d} outcome=invalid_route reward={0.0:8.2f} "
                            f"final_dist={math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1]):6.2f} "
                            f"parse_ok={int(parsed)} final_ok={int(reached_final)} plan_valid=0 "
                            f"fallback={int(fallback)} selected={selected[:24]} route_len=00 "
                            f"candidates={len(candidates):02d} latency={latency:.2f}s "
                            f"raw={text[:160].replace(chr(10), ' ')}\n"
                        )
                        log.flush()
                        continue

                    episode_reward = 0.0
                    steps_used = 0
                    collided = False
                    for subgoal in route or [final_goal]:
                        _set_goal(env, subgoal)
                        obs = _current_obs(env)
                        while steps_used < args.max_steps:
                            action = _predict_action(policy, obs)
                            result = env.step(action)
                            if len(result) == 5:
                                obs, reward, terminated, truncated, info = result
                                done = terminated or truncated
                            else:
                                obs, reward, done, info = result
                            episode_reward += float(reward)
                            steps_used += 1
                            base = _base_env(env)
                            collided = bool(dict(info).get("collided", False))
                            if collided:
                                break
                            if math.hypot(base.robot_x - subgoal[0], base.robot_y - subgoal[1]) <= base.success_radius:
                                break
                            if done:
                                break
                        if collided or steps_used >= args.max_steps:
                            break
                    base = _base_env(env)
                    final_dist = math.hypot(base.robot_x - final_goal[0], base.robot_y - final_goal[1])
                    if final_dist <= base.success_radius and not collided:
                        successes += 1
                        outcome = "success"
                    elif collided:
                        collisions += 1
                        outcome = "collision"
                    else:
                        timeouts += 1
                        outcome = "timeout"
                    rewards += episode_reward
                    log.write(
                        f"episode={episode:03d} outcome={outcome:9s} reward={episode_reward:8.2f} "
                        f"final_dist={final_dist:6.2f} parse_ok={int(parsed)} final_ok={int(reached_final)} "
                        f"plan_valid={int(counted_valid)} fallback={int(fallback)} selected={selected[:24]} "
                        f"route_len={len(route):02d} candidates={len(candidates):02d} latency={latency:.2f}s "
                        f"raw={text[:160].replace(chr(10), ' ')}\n"
                    )
                    log.flush()

            if hf_model is not None:
                del hf_model
            if tokenizer is not None:
                del tokenizer
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

            denom = max(episodes_run, 1)
            row = {
                "model": model_name,
                "source": "HF",
                "model_type": model_type,
                "params": params,
                "generative": int(generative),
                "algo": args.algo,
                "episodes": episodes_run,
                "success_rate": successes / denom,
                "collision_rate": collisions / denom,
                "timeout_rate": timeouts / denom,
                "invalid_route_rate": invalid_routes / denom,
                "json_parse_rate": parse_ok / denom,
                "final_node_rate": final_ok / denom,
                "complete_route_rate": plan_valid / denom,
                "fallback_rate": fallback_count / denom,
                "mean_reward": rewards / denom,
                "avg_latency_s": latency_total / denom,
                "load_error": load_error[:180],
            }
            rows.append(row)
            print(
                f"{args.algo}+{model_name}: success={row['success_rate']:.3f} "
                f"collision={row['collision_rate']:.3f} invalid={row['invalid_route_rate']:.3f} "
                f"complete={row['complete_route_rate']:.3f} "
                f"fallback={row['fallback_rate']:.3f} latency={row['avg_latency_s']:.2f}s"
            )

    fieldnames = [
        "model",
        "source",
        "model_type",
        "params",
        "generative",
        "algo",
        "episodes",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "invalid_route_rate",
        "json_parse_rate",
        "final_node_rate",
        "complete_route_rate",
        "fallback_rate",
        "mean_reward",
        "avg_latency_s",
        "load_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
