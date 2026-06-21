from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_rl_nav.envs.action_wrappers import DiscreteActionWrapper
from llm_rl_nav.envs.hospital_2d_env import ALL_SEMANTIC_MAP_IDS
from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, build_nav_env
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.train_multimap_ppo import parse_maps
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv
from llm_rl_nav.utils import latest_successful_ppo_path


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Evaluate a PPO/SAC/A2C/DQN policy across semantic indoor maps.")
    parser.add_argument("--algo", choices=("ppo", "sac", "a2c", "dqn"), default="ppo")
    parser.add_argument(
        "--model",
        default=str(latest_successful_ppo_path(root)),
        help="Path to model zip.",
    )
    parser.add_argument("--episodes", type=int, default=12, help="Episodes per map.")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--maps", type=parse_maps, default=ALL_SEMANTIC_MAP_IDS, help="Comma-separated map ids or all.")
    parser.add_argument(
        "--map-source",
        default=os.environ.get("MAP_SOURCE", "gazebo_3d_projection"),
        choices=MAP_SOURCE_CHOICES,
        help="semantic_2d uses legacy hand projection; gazebo_3d_projection loads geometry from generated Gazebo worlds.",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--reward-profile", default="v7_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=8.0)
    parser.add_argument("--goal-max-distance", type=float, default=24.0)
    parser.add_argument("--goal-point-probability", type=float, default=0.90)
    parser.add_argument("--safety-shield", action="store_true", help="Evaluate the deployed policy with hard geometry shield.")
    parser.add_argument("--shield-min-clearance", type=float, default=0.21)
    parser.add_argument("--shield-intervention-penalty", type=float, default=1.2)
    parser.add_argument("--waypoint-goals", action="store_true", help="Evaluate with high-level waypoint subgoals.")
    parser.add_argument("--waypoint-spacing", type=float, default=3.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.8)
    parser.add_argument("--waypoint-reward", type=float, default=24.0)
    parser.add_argument("--waypoint-failed-plan-penalty", type=float, default=60.0)
    parser.add_argument("--waypoint-final-distance-penalty", type=float, default=14.0)
    parser.add_argument("--waypoint-incomplete-penalty", type=float, default=22.0)
    args = parser.parse_args()

    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    model_cls = {"ppo": PPO, "sac": SAC, "a2c": A2C, "dqn": DQN}[args.algo]
    model = model_cls.load(args.model)
    total_successes = 0
    total_collisions = 0
    total_timeouts = 0
    total_episodes = 0

    for map_index, map_id in enumerate(args.maps):
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
        if args.waypoint_goals:
            env = WaypointGoalEnv(
                env,
                waypoint_spacing=args.waypoint_spacing,
                grid_resolution=args.waypoint_grid_resolution,
                waypoint_reward=args.waypoint_reward,
                failed_plan_penalty=args.waypoint_failed_plan_penalty,
                final_distance_penalty=args.waypoint_final_distance_penalty,
                incomplete_waypoint_penalty=args.waypoint_incomplete_penalty,
            )
        if args.algo == "dqn":
            env = DiscreteActionWrapper(env)
        successes = 0
        collisions = 0
        timeouts = 0
        rewards = 0.0

        print(f"=== map={map_id} ===")
        for episode in range(args.episodes):
            reset_result = env.reset(seed=args.seed + map_index * 1000 + episode, options={"map_id": map_id})
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            episode_reward = 0.0
            last_info = {}
            last_terminated = False
            last_truncated = False

            for _ in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                    last_terminated = bool(terminated)
                    last_truncated = bool(truncated)
                else:
                    obs, reward, done, info = result
                    last_terminated = bool(done)
                    last_truncated = False
                episode_reward += reward
                last_info = info
                if done:
                    break

            dist = float(
                last_info.get(
                    "distance_to_final_goal" if args.waypoint_goals else "distance_to_goal",
                    999.0,
                )
            )
            collided = bool(last_info.get("collided", False))
            success_radius = float(last_info.get("success_radius", 0.75))
            reached_goal = dist <= success_radius
            if reached_goal and not collided:
                successes += 1
                outcome = "success"
            elif collided:
                collisions += 1
                outcome = "collision"
            else:
                timeouts += 1
                outcome = "timeout" if last_truncated or not last_terminated else "stalled"
            rewards += episode_reward
            print(
                f"episode={episode:03d} outcome={outcome:9s} "
                f"reward={episode_reward:8.2f} final_dist={dist:6.2f} "
                f"success_radius={success_radius:4.2f}"
            )

        total_successes += successes
        total_collisions += collisions
        total_timeouts += timeouts
        total_episodes += args.episodes
        print(
            f"map_summary success={successes / args.episodes:.3f} "
            f"collision={collisions / args.episodes:.3f} "
            f"timeout={timeouts / args.episodes:.3f} "
            f"mean_reward={rewards / args.episodes:.3f}"
        )

    print("--- overall summary ---")
    print(f"episodes: {total_episodes}")
    print(f"success_rate: {total_successes / total_episodes:.3f}")
    print(f"collision_rate: {total_collisions / total_episodes:.3f}")
    print(f"timeout_rate: {total_timeouts / total_episodes:.3f}")


if __name__ == "__main__":
    main()
