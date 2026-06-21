from __future__ import annotations

import argparse
import os

import numpy as np

from llm_rl_nav.envs.hospital_2d_env import ALL_SEMANTIC_MAP_IDS
from llm_rl_nav.envs.semantic_world_source import MAP_SOURCE_CHOICES, MAP_SOURCE_SEMANTIC_2D, build_nav_env
from llm_rl_nav.training.train_multimap_ppo import parse_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the generic multi-map RL env.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--maps", type=parse_maps, default=ALL_SEMANTIC_MAP_IDS, help="Comma-separated map ids or all.")
    parser.add_argument(
        "--map-source",
        default=os.environ.get("MAP_SOURCE", MAP_SOURCE_SEMANTIC_2D),
        choices=MAP_SOURCE_CHOICES,
    )
    args = parser.parse_args()

    for map_index, map_id in enumerate(args.maps):
        env = build_nav_env(args.map_source, seed=args.seed + map_index, map_id=map_id)
        reset_result = env.reset(seed=args.seed + map_index, options={"map_id": map_id})
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        print(f"=== map={map_id} observation_shape={obs.shape} info={env._info()} ===")
        total_reward = 0.0
        for step in range(args.steps):
            action = np.array([0.12, 0.4 * np.sin(step / 3.0)], dtype=np.float32)
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            total_reward += reward
            print(
                f"step={step:03d} reward={reward:7.3f} "
                f"dist={info['distance_to_goal']:6.2f} "
                f"min_lidar={info['min_lidar']:5.2f} done={done}"
            )
            if done:
                break
        print(f"map={map_id} total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
