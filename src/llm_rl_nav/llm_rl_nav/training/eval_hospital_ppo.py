from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_rl_nav.envs.hospital_2d_env import Hospital2DNavEnv
from llm_rl_nav.utils import latest_successful_ppo_path


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Evaluate a PPO full-map hospital policy.")
    parser.add_argument(
        "--model",
        default=str(latest_successful_ppo_path(root)),
        help="Path to PPO model zip.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for evaluation.") from exc

    env = Hospital2DNavEnv(seed=args.seed)
    model = PPO.load(args.model)

    successes = 0
    collisions = 0
    timeouts = 0
    total_reward = 0.0

    for episode in range(args.episodes):
        reset_result = env.reset(seed=args.seed + episode)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0.0
        last_info = {}

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            episode_reward += reward
            last_info = info
            if done:
                break

        total_reward += episode_reward
        dist = float(last_info.get("distance_to_goal", 999.0))
        collided = bool(last_info.get("collided", False))
        success_radius = float(last_info.get("success_radius", env.success_radius))
        if dist <= success_radius and not collided:
            successes += 1
            outcome = "success"
        elif collided:
            collisions += 1
            outcome = "collision"
        else:
            timeouts += 1
            outcome = "timeout"

        print(
            f"episode={episode:03d} outcome={outcome:9s} "
            f"reward={episode_reward:8.2f} final_dist={dist:6.2f} "
            f"success_radius={success_radius:4.2f}"
        )

    print("--- summary ---")
    print(f"episodes: {args.episodes}")
    print(f"success_rate: {successes / args.episodes:.3f}")
    print(f"collision_rate: {collisions / args.episodes:.3f}")
    print(f"timeout_rate: {timeouts / args.episodes:.3f}")
    print(f"mean_reward: {total_reward / args.episodes:.3f}")


if __name__ == "__main__":
    main()
