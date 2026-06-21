from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_rl_nav.envs.hospital_2d_env import Hospital2DNavEnv


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Train PPO on the full primitive hospital map.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-name", default="ppo_hospital_full_map")
    parser.add_argument(
        "--load-model",
        default=None,
        help="Optional existing PPO zip to continue training from.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in this environment. "
            "Install requirements first, then rerun training."
        ) from exc

    log_dir = root / "logs" / "training" / "hospital_ppo"
    models_dir = root / "models" / "PPO"
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(Hospital2DNavEnv(seed=args.seed), filename=str(log_dir / "monitor.csv"))

    if args.load_model:
        print(f"Continuing PPO training from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            custom_objects={"action_space": env.action_space},
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            ent_coef=0.01,
            seed=args.seed,
        )

    print("Training PPO on the full hospital map.")
    print("No forbidden-zone constraints are active during this baseline stage.")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    save_path = models_dir / args.save_name
    model.save(str(save_path))
    print(f"Saved full-map baseline model to: {save_path}.zip")


if __name__ == "__main__":
    main()
