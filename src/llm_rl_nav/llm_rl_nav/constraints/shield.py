from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from llm_rl_nav.constraints.schema import Constraint
from llm_rl_nav.constraints.semantic_map import SemanticMap
from llm_rl_nav.envs.hospital_2d_env import Hospital2DNavEnv
from llm_rl_nav.utils import wrap_angle


@dataclass(frozen=True)
class ShieldDecision:
    action: np.ndarray
    intervened: bool
    reason: str | None = None


class ConstraintShield:
    """Action-level safety layer for symbolic navigation constraints."""

    def __init__(
        self,
        semantic_map: SemanticMap,
        constraints: list[Constraint],
        lookahead_steps: int = 8,
    ):
        self.semantic_map = semantic_map
        self.constraints = constraints
        self.lookahead_steps = lookahead_steps
        self.candidates = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.2], dtype=np.float32),
            np.array([0.0, -1.2], dtype=np.float32),
            np.array([-0.06, 0.0], dtype=np.float32),
            np.array([-0.05, 0.8], dtype=np.float32),
            np.array([-0.05, -0.8], dtype=np.float32),
            np.array([0.06, 1.0], dtype=np.float32),
            np.array([0.06, -1.0], dtype=np.float32),
            np.array([0.10, 0.0], dtype=np.float32),
        ]

    def filter_action(self, env: Hospital2DNavEnv, action: np.ndarray) -> ShieldDecision:
        action = np.asarray(action, dtype=np.float32)
        violation = self.first_violation(env, action)
        if violation is None:
            return ShieldDecision(action=action, intervened=False)

        safe_candidates: list[tuple[float, np.ndarray]] = []
        for candidate in self.candidates:
            candidate_violation = self.first_violation(env, candidate)
            if candidate_violation is not None:
                continue
            score = self._candidate_score(env, candidate)
            safe_candidates.append((score, candidate))

        if not safe_candidates:
            return ShieldDecision(
                action=np.array([0.0, 0.0], dtype=np.float32),
                intervened=True,
                reason=f"{violation}; no safe candidate found",
            )

        _, best = max(safe_candidates, key=lambda item: item[0])
        return ShieldDecision(action=best, intervened=True, reason=violation)

    def first_violation(self, env: Hospital2DNavEnv, action: np.ndarray) -> str | None:
        path = self._rollout_points(env, action)
        for x, y in path:
            for constraint in self.constraints:
                if constraint.type == "forbidden_zone":
                    if self.semantic_map.contains_point(constraint.target, x, y):
                        return f"would enter forbidden_zone:{constraint.target}"
                elif constraint.type == "min_distance":
                    min_distance = constraint.distance_m or 1.0
                    distance = self.semantic_map.distance_to_entity(constraint.target, x, y)
                    if distance < min_distance:
                        return (
                            f"would violate min_distance:{constraint.target} "
                            f"distance={distance:.2f} < {min_distance:.2f}"
                        )
        return None

    def count_state_violations(self, env: Hospital2DNavEnv) -> dict[str, int]:
        counts = {"forbidden_zone": 0, "min_distance": 0}
        x, y = env.robot_x, env.robot_y
        for constraint in self.constraints:
            if constraint.type == "forbidden_zone" and self.semantic_map.contains_point(
                constraint.target, x, y
            ):
                counts["forbidden_zone"] += 1
            elif constraint.type == "min_distance":
                min_distance = constraint.distance_m or 1.0
                if self.semantic_map.distance_to_entity(constraint.target, x, y) < min_distance:
                    counts["min_distance"] += 1
        return counts

    def _rollout_points(self, env: Hospital2DNavEnv, action: np.ndarray) -> list[tuple[float, float]]:
        v = float(np.clip(action[0], -0.12, 0.22))
        w = float(np.clip(action[1], -2.0, 2.0))
        x, y, yaw = env.robot_x, env.robot_y, env.robot_yaw
        points: list[tuple[float, float]] = []

        for _ in range(self.lookahead_steps):
            yaw = wrap_angle(yaw + w * env.dt)
            x += v * math.cos(yaw) * env.dt
            y += v * math.sin(yaw) * env.dt
            points.append((x, y))
        return points

    def _candidate_score(self, env: Hospital2DNavEnv, action: np.ndarray) -> float:
        x, y = self._rollout_points(env, action)[-1]
        distance_to_goal = math.hypot(env.goal_x - x, env.goal_y - y)
        turn_penalty = abs(float(action[1])) * 0.05
        speed_bonus = float(action[0]) * 0.1
        reverse_penalty = max(-float(action[0]), 0.0) * 0.04
        return -distance_to_goal - turn_penalty - reverse_penalty + speed_bonus
