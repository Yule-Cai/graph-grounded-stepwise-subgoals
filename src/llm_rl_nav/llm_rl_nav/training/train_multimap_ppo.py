from __future__ import annotations

import argparse
import os
from pathlib import Path

# macOS/conda can load more than one OpenMP runtime when Stable-Baselines,
# PyTorch, NumPy, and subprocess vector envs meet. Keep each worker single
# threaded and allow duplicate libomp so parallel PPO can start reliably.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from llm_rl_nav.envs.hospital_2d_env import ALL_SEMANTIC_MAP_IDS
from llm_rl_nav.envs.semantic_world_source import (
    MAP_SOURCE_CHOICES,
    MAP_SOURCE_GAZEBO_3D,
    MAP_SOURCE_SEMANTIC_2D,
    build_nav_env,
    ensure_semantic_3d_worlds,
)
from llm_rl_nav.training.llm_guided_env import LLMGuidedRecoveryEnv
from llm_rl_nav.training.shielded_env import ShieldedActionEnv
from llm_rl_nav.training.waypoint_env import WaypointGoalEnv

CURRICULUM_STAGES = (
    ("easy", ("reference_family_flat", "studio_apartment", "two_bedroom_apartment", "open_plan_house")),
    ("structured", ("bungalow_house", "courtyard_house", "duplex_family", "narrow_lot_house")),
    ("hard", ("suburban_villa", "townhouse_long", "luxury_villa")),
    ("all", ALL_SEMANTIC_MAP_IDS),
)

GOAL_DISTANCE_CURRICULUM = {
    "easy": (3.0, 10.0),
    "structured": (5.0, 18.0),
    "hard": (7.0, 24.0),
    "all": (8.0, 24.0),
    "selected": (8.0, 24.0),
}


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def parse_maps(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return ALL_SEMANTIC_MAP_IDS
    maps = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = [item for item in maps if item not in ALL_SEMANTIC_MAP_IDS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown map id(s): {', '.join(unknown)}")
    if not maps:
        raise argparse.ArgumentTypeError("at least one map id is required")
    return maps


def _build_single_env(args, maps: tuple[str, ...], log_dir: Path, stage_name: str, env_index: int = 0):
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
        seed=args.seed + env_index * 1009,
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
    if args.llm_guide:
        env = LLMGuidedRecoveryEnv(
            env,
            base_url=args.lm_studio_url,
            model=args.lm_model,
            trigger_clearance=args.llm_trigger_clearance,
            max_calls_per_episode=args.llm_max_calls_per_episode,
            teacher_force_prob=args.llm_teacher_force_prob,
            cooldown_steps=args.llm_cooldown_steps,
            timeout_s=args.llm_timeout_s,
        )
        info_keywords = tuple(
            dict.fromkeys(
                (
                    *info_keywords,
                    "llm_guidance_calls",
                    "llm_guidance_failures",
                    "llm_guidance_active",
                )
            )
        )
    return Monitor(
        env,
        filename=str(log_dir / f"monitor_{stage_name}_{env_index}.csv"),
        info_keywords=info_keywords,
    )


def _build_env(args, maps: tuple[str, ...], log_dir: Path, stage_name: str):
    if args.n_envs <= 1:
        return _build_single_env(args, maps, log_dir, stage_name, 0)

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in this environment. "
            "Install requirements first, then rerun training."
        ) from exc

    def make_env(env_index: int):
        def _factory():
            return _build_single_env(args, maps, log_dir, stage_name, env_index)

        return _factory

    env_fns = [make_env(index) for index in range(args.n_envs)]
    if args.vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=args.subproc_start_method)


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

    parser = argparse.ArgumentParser(description="Train one generic PPO policy over all semantic indoor maps.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-name", default="ppo_indoor_nav_multimap")
    parser.add_argument("--maps", type=parse_maps, default=ALL_SEMANTIC_MAP_IDS, help="Comma-separated map ids or all.")
    parser.add_argument(
        "--map-source",
        default=os.environ.get("MAP_SOURCE", MAP_SOURCE_GAZEBO_3D),
        choices=MAP_SOURCE_CHOICES,
        help="semantic_2d uses legacy hand projection; gazebo_3d_projection loads geometry from generated Gazebo worlds.",
    )
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--reward-profile", default="v7_goal", choices=("v8_goal", "v7_goal", "v6_safe"))
    parser.add_argument("--goal-min-distance", type=float, default=None)
    parser.add_argument("--goal-max-distance", type=float, default=None)
    parser.add_argument("--goal-point-probability", type=float, default=0.90)
    parser.add_argument("--waypoint-goals", action="store_true", help="Train on short waypoint subgoals while preserving final task goals.")
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
    parser.add_argument("--llm-guide", action="store_true", help="Use LM Studio as a low-frequency recovery teacher.")
    parser.add_argument("--lm-studio-url", default=os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234"))
    parser.add_argument("--lm-model", default=os.environ.get("LM_STUDIO_MODEL", "local-model"))
    parser.add_argument("--llm-max-calls-per-episode", type=int, default=3)
    parser.add_argument("--llm-trigger-clearance", type=float, default=0.72)
    parser.add_argument("--llm-teacher-force-prob", type=float, default=0.18)
    parser.add_argument("--llm-cooldown-steps", type=int, default=95)
    parser.add_argument("--llm-timeout-s", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.0012)
    parser.add_argument("--n-envs", type=int, default=int(os.environ.get("N_ENVS", "1")))
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default=os.environ.get("VEC_ENV", "subproc"))
    parser.add_argument("--subproc-start-method", default=os.environ.get("SUBPROC_START_METHOD", "forkserver"))
    parser.add_argument("--ppo-n-steps", type=int, default=int(os.environ.get("PPO_N_STEPS", "1024")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PPO_BATCH_SIZE", "1024")))
    parser.add_argument("--checkpoint-freq", type=int, default=int(os.environ.get("PPO_CHECKPOINT_FREQ", "1000000")))
    parser.add_argument(
        "--reset-action-std",
        type=float,
        default=None,
        help="Optional std reset for loaded continuous PPO policies, e.g. 0.35.",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        help="Optional existing PPO zip to continue training from.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.utils import get_schedule_fn
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in this environment. "
            "Install requirements first, then rerun training."
        ) from exc

    log_dir = root / "logs" / "training" / "multimap_ppo"
    models_dir = root / "models" / "PPO"
    log_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = None
    if args.checkpoint_freq > 0:
        checkpoint_dir = log_dir / "checkpoints" / args.save_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // max(args.n_envs, 1), 1),
            save_path=str(checkpoint_dir),
            name_prefix=args.save_name,
        )

    stages = _curriculum_for_maps(args.maps) if args.curriculum else (("selected", args.maps),)
    if args.map_source == MAP_SOURCE_GAZEBO_3D:
        ensure_semantic_3d_worlds(root, map_ids=args.maps, overwrite=True)
    env = _build_env(args, stages[0][1], log_dir, stages[0][0])

    if args.load_model:
        print(f"Continuing generic PPO training from: {args.load_model}")
        model = PPO.load(
            args.load_model,
            env=env,
            tensorboard_log=str(log_dir),
            seed=args.seed,
            custom_objects={"action_space": env.action_space},
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.ent_coef = args.ent_coef
        if args.reset_action_std is not None and hasattr(model.policy, "log_std"):
            import math as _math

            model.policy.log_std.data.fill_(_math.log(max(args.reset_action_std, 1e-4)))
            print(f"Reset loaded PPO action std to: {args.reset_action_std}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(log_dir),
            learning_rate=args.learning_rate,
            n_steps=args.ppo_n_steps,
            batch_size=args.batch_size,
            gamma=0.99,
            ent_coef=args.ent_coef,
            seed=args.seed,
        )

    print("Training generic PPO over semantic indoor maps:")
    print(", ".join(args.maps))
    print(f"Map source: {args.map_source}")
    if args.map_source != MAP_SOURCE_SEMANTIC_2D:
        print("3D Gazebo world files are the geometry source; 2D is only the fast navigation projection.")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Reward profile: {args.reward_profile}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Entropy coefficient: {args.ent_coef}")
    print(f"Parallel envs: {args.n_envs}")
    print(f"Vec env: {args.vec_env}")
    print(f"PPO n_steps per env: {args.ppo_n_steps}")
    print(f"PPO batch size: {args.batch_size}")
    if args.checkpoint_freq > 0:
        print(f"Checkpoint every approx {args.checkpoint_freq} timesteps.")
    if args.waypoint_goals:
        print("Waypoint subgoal decomposition is ON.")
        print(f"Waypoint spacing: {args.waypoint_spacing}")
        print(f"Waypoint grid resolution: {args.waypoint_grid_resolution}")
        print(f"Waypoint reward: {args.waypoint_reward}")
        print(f"Waypoint final distance penalty: {args.waypoint_final_distance_penalty}")
        print(f"Waypoint incomplete penalty: {args.waypoint_incomplete_penalty}")
    if args.safety_shield:
        print("Hard geometry safety shield is ON.")
        print(f"Shield min clearance: {args.shield_min_clearance}")
        print(f"Shield intervention penalty: {args.shield_intervention_penalty}")
    if args.curriculum:
        print("Curriculum training is ON:")
        for stage_name, stage_maps in stages:
            goal_min, goal_max = _goal_distance_for_stage(args, stage_name)
            max_text = "unbounded" if goal_max is None else f"{goal_max:.1f}m"
            print(f"  {stage_name}: {', '.join(stage_maps)} | goal distance {goal_min:.1f}m - {max_text}")
    if args.llm_guide:
        print("LLM-guided training is ON.")
        print(f"LM Studio URL: {args.lm_studio_url}")
        print(f"LM model: {args.lm_model}")
        print(
            "The LLM is queried only for hard recovery states; "
            "forbidden-zone constraints are still inactive during this RL stage."
        )
    else:
        print("No forbidden-zone constraints or LLM teacher are active during this baseline RL stage.")
    remaining = args.timesteps
    for index, (stage_name, stage_maps) in enumerate(stages):
        stage_steps = remaining if index == len(stages) - 1 else max(min(args.timesteps // len(stages), remaining), 1)
        remaining -= stage_steps
        if stage_steps <= 0:
            break
        if index > 0:
            env = _build_env(args, stage_maps, log_dir, stage_name)
            model.set_env(env)
        print(f"=== PPO stage {index + 1}/{len(stages)}: {stage_name}, steps={stage_steps}, maps={', '.join(stage_maps)} ===")
        model.learn(
            total_timesteps=stage_steps,
            reset_num_timesteps=(index == 0),
            progress_bar=False,
            callback=checkpoint_callback,
        )

    save_path = models_dir / args.save_name
    model.save(str(save_path))
    print(f"Saved generic multi-map model to: {save_path}.zip")


if __name__ == "__main__":
    main()
