from __future__ import annotations

import argparse
import os
from pathlib import Path

from llm_rl_nav.envs.action_wrappers import DiscreteActionWrapper
from llm_rl_nav.envs.hospital_2d_env import ALL_SEMANTIC_MAP_IDS
from llm_rl_nav.envs.semantic_world_source import (
    MAP_SOURCE_CHOICES,
    MAP_SOURCE_GAZEBO_3D,
    MAP_SOURCE_SEMANTIC_2D,
    build_nav_env,
)
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv
from llm_rl_nav.training.train_multimap_ppo import (
    CURRICULUM_STAGES,
    GOAL_DISTANCE_CURRICULUM,
    parse_maps,
)


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def build_model(algo: str, env, log_dir: Path, seed: int, load_model: str | None = None):
    try:
        from stable_baselines3 import A2C, DQN, SAC
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in this environment. "
            "Install requirements first, then rerun training."
        ) from exc

    algo = algo.lower()
    cls_by_algo = {"sac": SAC, "a2c": A2C, "dqn": DQN}
    cls = cls_by_algo[algo]
    if load_model:
        print(f"Continuing {algo.upper()} training from: {load_model}")
        custom_objects = None if algo == "dqn" else {"action_space": env.action_space}
        return cls.load(
            load_model,
            env=env,
            tensorboard_log=str(log_dir),
            seed=seed,
            custom_objects=custom_objects,
        )

    if algo == "sac":
        return SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            learning_rate=3e-4,
            buffer_size=160_000,
            batch_size=256,
            learning_starts=6_000,
            train_freq=8,
            gradient_steps=8,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed,
        )
    if algo == "a2c":
        return A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            learning_rate=7e-4,
            n_steps=16,
            gamma=0.99,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed,
        )
    return DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir),
        learning_rate=2.5e-4,
        buffer_size=120_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=2_000,
        exploration_fraction=0.65,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.08,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
    )


def _build_env(algo: str, args, maps: tuple[str, ...], log_dir: Path, stage_name: str):
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in this environment. "
            "Install requirements first, then rerun training."
        ) from exc

    stage_goal_min, stage_goal_max = _goal_distance_for_stage(args, stage_name)
    env = build_nav_env(
        args.map_source,
        seed=args.seed,
        map_ids=maps,
        max_steps=args.max_steps,
        reward_profile=args.reward_profile,
        goal_min_distance=stage_goal_min,
        goal_max_distance=stage_goal_max,
        goal_point_probability=args.goal_point_probability,
    )
    info_keywords = ("map_id",)
    if args.safety_shield:
        env = ShieldedActionEnv(
            env,
            min_clearance=args.shield_min_clearance,
            intervention_penalty=args.shield_intervention_penalty,
        )
        info_keywords = (
            "map_id",
            "shield_interventions",
            "shield_blocks",
            "shield_last_intervention",
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
        info_keywords = tuple(
            dict.fromkeys(
                (
                    *info_keywords,
                    "waypoint_index",
                    "waypoint_count",
                    "waypoint_plan_failed",
                )
            )
        )
    if algo == "dqn":
        env = DiscreteActionWrapper(env)
    return Monitor(env, filename=str(log_dir / f"monitor_{stage_name}.csv"), info_keywords=info_keywords)


def _curriculum_for_maps(maps: tuple[str, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    selected = set(maps)
    stages: list[tuple[str, tuple[str, ...]]] = []
    for name, stage_maps in CURRICULUM_STAGES:
        filtered = tuple(item for item in stage_maps if item in selected)
        if filtered:
            stages.append((name, filtered))
    return tuple(stages) or (("selected", maps),)


def _goal_distance_for_stage(args, stage_name: str) -> tuple[float, float | None]:
    if args.goal_min_distance is not None or args.goal_max_distance is not None:
        return (
            args.goal_min_distance if args.goal_min_distance is not None else 8.0,
            args.goal_max_distance,
        )
    return GOAL_DISTANCE_CURRICULUM.get(stage_name, GOAL_DISTANCE_CURRICULUM["selected"])


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(description="Train SAC/A2C/DQN baselines over the semantic indoor map set.")
    parser.add_argument("--algo", choices=("sac", "a2c", "dqn"), required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--maps", type=parse_maps, default=ALL_SEMANTIC_MAP_IDS, help="Comma-separated map ids or all.")
    parser.add_argument(
        "--map-source",
        default=os.environ.get("MAP_SOURCE", MAP_SOURCE_GAZEBO_3D),
        choices=MAP_SOURCE_CHOICES,
        help="semantic_2d uses legacy hand projection; gazebo_3d_projection loads geometry from generated Gazebo worlds.",
    )
    parser.add_argument("--load-model", default=None, help="Optional existing model zip to continue training from.")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--reward-profile", default="v7_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=None)
    parser.add_argument("--goal-max-distance", type=float, default=None)
    parser.add_argument("--goal-point-probability", type=float, default=0.90)
    parser.add_argument("--waypoint-goals", action="store_true")
    parser.add_argument("--waypoint-spacing", type=float, default=3.2)
    parser.add_argument("--waypoint-grid-resolution", type=float, default=0.8)
    parser.add_argument("--waypoint-reward", type=float, default=24.0)
    parser.add_argument("--waypoint-failed-plan-penalty", type=float, default=60.0)
    parser.add_argument("--waypoint-final-distance-penalty", type=float, default=14.0)
    parser.add_argument("--waypoint-incomplete-penalty", type=float, default=22.0)
    parser.add_argument("--curriculum", action="store_true", help="Train easy -> structured -> hard -> all map stages.")
    parser.add_argument("--safety-shield", action="store_true", help="Block collision-risky actions during training.")
    parser.add_argument("--shield-min-clearance", type=float, default=0.21)
    parser.add_argument("--shield-intervention-penalty", type=float, default=1.2)
    args = parser.parse_args()

    algo = args.algo.lower()
    log_dir = root / "logs" / "training" / f"multimap_{algo}"
    models_dir = root / "models" / algo.upper()
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    stages = _curriculum_for_maps(args.maps) if args.curriculum else (("selected", args.maps),)
    env = _build_env(algo, args, stages[0][1], log_dir, stages[0][0])

    save_name = args.save_name or f"{algo}_indoor_nav_12map_v3_baseline"
    model = build_model(algo, env, log_dir, args.seed, args.load_model)

    print(f"Training {algo.upper()} baseline over semantic indoor maps:")
    print(", ".join(args.maps))
    print(f"Map source: {args.map_source}")
    if args.map_source != MAP_SOURCE_SEMANTIC_2D:
        print("3D Gazebo world files are the geometry source; 2D is only the fast navigation projection.")
    print("LLM guidance: OFF")
    if args.safety_shield:
        print("Hard geometry safety shield: ON")
        print(f"Shield min clearance: {args.shield_min_clearance}")
        print(f"Shield intervention penalty: {args.shield_intervention_penalty}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Reward profile: {args.reward_profile}")
    if args.waypoint_goals:
        print("Waypoint subgoal decomposition: ON")
    if args.curriculum:
        print("Curriculum training is ON:")
        for stage_name, stage_maps in stages:
            goal_min, goal_max = _goal_distance_for_stage(args, stage_name)
            max_text = "unbounded" if goal_max is None else f"{goal_max:.1f}m"
            print(f"  {stage_name}: {', '.join(stage_maps)} | goal distance {goal_min:.1f}m - {max_text}")
    print(f"Save name: {save_name}")

    remaining = args.timesteps
    for index, (stage_name, stage_maps) in enumerate(stages):
        stage_steps = remaining if index == len(stages) - 1 else max(min(args.timesteps // len(stages), remaining), 1)
        remaining -= stage_steps
        if stage_steps <= 0:
            break
        if index > 0:
            env = _build_env(algo, args, stage_maps, log_dir, stage_name)
            model.set_env(env)
        print(f"=== {algo.upper()} stage {index + 1}/{len(stages)}: {stage_name}, steps={stage_steps}, maps={', '.join(stage_maps)} ===")
        model.learn(total_timesteps=stage_steps, reset_num_timesteps=(index == 0), progress_bar=False)

    save_path = models_dir / save_name
    model.save(str(save_path))
    print(f"Saved {algo.upper()} baseline model to: {save_path}.zip")


if __name__ == "__main__":
    main()
