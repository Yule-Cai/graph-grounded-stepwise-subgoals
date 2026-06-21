from __future__ import annotations

import argparse

import numpy as np

from llm_rl_nav.envs.hospital_2d_env import Hospital2DNavEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the full-map hospital RL env.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = Hospital2DNavEnv(seed=args.seed)
    reset_result = env.reset(seed=args.seed)
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {env._info()}")

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

    print(f"Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
