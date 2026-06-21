from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

from llm_rl_nav.constraints import (
    LMStudioConstraintCompiler,
    NaturalLanguageConstraintCompiler,
    SemanticMap,
)
from llm_rl_nav.constraints.shield import ConstraintShield
from llm_rl_nav.constraints.validator import ConstraintValidator
from llm_rl_nav.envs.hospital_2d_env import Hospital2DNavEnv
from llm_rl_nav.utils import latest_successful_ppo_path


def project_root() -> Path:
    return Path(os.environ.get("LLM_RL_NAV_HOME", Path.cwd())).resolve()


def default_semantic_map_path(root: Path) -> Path:
    return root / "src" / "llm_rl_nav" / "config" / "semantic_maps" / "hospital_semantic.yaml"


@dataclass
class EvalStats:
    episodes: int = 0
    successes: int = 0
    collisions: int = 0
    timeouts: int = 0
    total_reward: float = 0.0
    forbidden_violations: int = 0
    distance_violations: int = 0
    interventions: int = 0

    def as_rates(self) -> dict[str, float]:
        return {
            "success_rate": self.successes / self.episodes,
            "collision_rate": self.collisions / self.episodes,
            "timeout_rate": self.timeouts / self.episodes,
            "mean_reward": self.total_reward / self.episodes,
            "forbidden_violations_per_episode": self.forbidden_violations / self.episodes,
            "distance_violations_per_episode": self.distance_violations / self.episodes,
            "interventions_per_episode": self.interventions / self.episodes,
        }


def main() -> None:
    root = project_root()
    os.environ.setdefault("MPLCONFIGDIR", str(root / "log" / "matplotlib"))

    parser = argparse.ArgumentParser(
        description="Compare PPO navigation with and without symbolic constraint shielding."
    )
    parser.add_argument(
        "--model",
        default=str(latest_successful_ppo_path(root)),
        help="Path to PPO model zip.",
    )
    parser.add_argument(
        "--rule",
        default="不要进红色厨房区，离蓝色花瓶远一点。",
        help="Natural-language rule compiled into symbolic constraints.",
    )
    parser.add_argument(
        "--semantic-map",
        default=str(default_semantic_map_path(root)),
        help="Path to semantic map YAML/JSON.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--compiler",
        choices=["local", "lmstudio"],
        default="local",
        help="Natural-language compiler backend.",
    )
    parser.add_argument("--lm-studio-url", default="http://localhost:1234/v1")
    parser.add_argument("--llm-model", default=os.environ.get("LM_STUDIO_MODEL", "local-model"))
    parser.add_argument(
        "--scenario",
        choices=["random", "red_zone", "vase"],
        default="random",
        help="Evaluation scenario. random uses random starts/goals; red_zone and vase are constraint stress tests.",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("stable-baselines3 is required for shield evaluation.") from exc

    semantic_map = SemanticMap.from_file(args.semantic_map)
    if args.compiler == "lmstudio":
        compiler = LMStudioConstraintCompiler(
            semantic_map,
            base_url=args.lm_studio_url,
            model=args.llm_model,
        )
    else:
        compiler = NaturalLanguageConstraintCompiler(semantic_map)
    validator = ConstraintValidator(semantic_map)
    constraint_set = validator.validate(compiler.compile(args.rule))

    print("Compiled constraints:")
    for constraint in constraint_set.constraints:
        print(f"  - {constraint.to_dict()}")
    if constraint_set.unknown_phrases:
        print(f"Unknown phrases: {constraint_set.unknown_phrases}")

    model = PPO.load(args.model)

    baseline = evaluate(
        model=model,
        semantic_map=semantic_map,
        constraints=constraint_set.constraints,
        episodes=args.episodes,
        seed=args.seed,
        scenario=args.scenario,
        shield_enabled=False,
    )
    shielded = evaluate(
        model=model,
        semantic_map=semantic_map,
        constraints=constraint_set.constraints,
        episodes=args.episodes,
        seed=args.seed,
        scenario=args.scenario,
        shield_enabled=True,
    )

    print_summary("baseline", baseline)
    print_summary("shielded", shielded)


def evaluate(
    model,
    semantic_map: SemanticMap,
    constraints,
    episodes: int,
    seed: int,
    scenario: str,
    shield_enabled: bool,
) -> EvalStats:
    stats = EvalStats(episodes=episodes)
    env = Hospital2DNavEnv(seed=seed)
    shield = ConstraintShield(semantic_map, constraints)

    for episode in range(episodes):
        reset_options = scenario_options(scenario, episode)
        reset_result = env.reset(seed=seed + episode, options=reset_options)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        episode_reward = 0.0
        last_info = {}

        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            if shield_enabled:
                decision = shield.filter_action(env, action)
                action = decision.action
                if decision.intervened:
                    stats.interventions += 1

            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result

            counts = shield.count_state_violations(env)
            stats.forbidden_violations += counts["forbidden_zone"]
            stats.distance_violations += counts["min_distance"]

            episode_reward += reward
            last_info = info
            if done:
                break

        stats.total_reward += episode_reward
        dist = float(last_info.get("distance_to_goal", 999.0))
        collided = bool(last_info.get("collided", False))
        success_radius = float(last_info.get("success_radius", env.success_radius))
        if dist <= success_radius and not collided:
            stats.successes += 1
        elif collided:
            stats.collisions += 1
        else:
            stats.timeouts += 1

    return stats


def scenario_options(scenario: str, episode: int) -> dict[str, object]:
    if scenario == "red_zone":
        starts = [(-13.0, -26.0), (-11.0, -28.3), (-13.0, -25.4), (-11.7, -28.3)]
        goals = [(-9.0, -26.0), (-11.0, -23.7), (-9.0, -25.4), (-10.2, -23.7)]
        start = starts[episode % len(starts)]
        goal = goals[episode % len(goals)]
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        return {"start": start, "goal": goal, "yaw": yaw}

    if scenario == "vase":
        starts = [(9.0, 1.2), (12.5, 1.2), (9.0, 5.2), (12.5, 5.2)]
        goals = [(12.5, 5.2), (9.0, 5.2), (12.5, 1.2), (9.0, 1.2)]
        start = starts[episode % len(starts)]
        goal = goals[episode % len(goals)]
        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        return {"start": start, "goal": goal, "yaw": yaw}

    return {}


def print_summary(name: str, stats: EvalStats) -> None:
    rates = stats.as_rates()
    print(f"--- {name} ---")
    print(f"episodes: {stats.episodes}")
    print(f"success_rate: {rates['success_rate']:.3f}")
    print(f"collision_rate: {rates['collision_rate']:.3f}")
    print(f"timeout_rate: {rates['timeout_rate']:.3f}")
    print(f"mean_reward: {rates['mean_reward']:.3f}")
    print(f"forbidden_violations_per_episode: {rates['forbidden_violations_per_episode']:.3f}")
    print(f"distance_violations_per_episode: {rates['distance_violations_per_episode']:.3f}")
    print(f"interventions_per_episode: {rates['interventions_per_episode']:.3f}")


if __name__ == "__main__":
    main()
