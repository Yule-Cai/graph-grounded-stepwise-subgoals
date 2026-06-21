from __future__ import annotations

import math
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    try:
        import gym
    except ImportError:  # pragma: no cover
        gym = None

from llm_rl_nav.utils import wrap_angle


DEFAULT_SHIELD_CANDIDATES = np.array(
    [
        [0.00, 0.00],
        [-0.10, 0.00],
        [-0.09, 1.20],
        [-0.09, -1.20],
        [0.00, 1.50],
        [0.00, -1.50],
        [0.07, 0.00],
        [0.08, 0.85],
        [0.08, -0.85],
        [0.13, 0.00],
        [0.13, 1.15],
        [0.13, -1.15],
        [0.18, 0.00],
    ],
    dtype=np.float32,
)


class ShieldedActionEnv(gym.Wrapper if gym else object):
    """Hard geometry shield for RL training and deployment evaluation.

    This wrapper is deliberately independent from semantic constraints and LLM
    calls. It blocks actions that would collide with walls in short-horizon
    rollout, then substitutes the safest progress-making candidate. The policy
    still receives a penalty when the shield has to intervene, so long training
    should learn to avoid asking for unsafe actions.
    """

    def __init__(
        self,
        env,
        min_clearance: float = 0.24,
        lookahead_steps: int = 8,
        interpolation_steps: int = 4,
        intervention_penalty: float = 2.5,
        candidates: np.ndarray | None = None,
    ):
        if gym is None:
            raise RuntimeError("gymnasium or gym is required to use ShieldedActionEnv")
        super().__init__(env)
        self.min_clearance = float(min_clearance)
        self.lookahead_steps = int(lookahead_steps)
        self.interpolation_steps = int(interpolation_steps)
        self.intervention_penalty = float(intervention_penalty)
        self.candidates = np.asarray(
            candidates if candidates is not None else DEFAULT_SHIELD_CANDIDATES,
            dtype=np.float32,
        )
        self.shield_interventions = 0
        self.shield_blocks = 0
        self.last_shield_reason = ""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self.shield_interventions = 0
        self.shield_blocks = 0
        self.last_shield_reason = ""
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = np.asarray(action, dtype=np.float32)
        base_env = self._base_env()
        safe_action, intervened, reason = self._filter_action(base_env, raw_action)

        result = self.env.step(safe_action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done_result = (obs, reward, terminated, truncated, info)
        else:
            obs, reward, done, info = result
            done_result = (obs, reward, done, info)

        if intervened:
            self.shield_interventions += 1
            self.shield_blocks += 1
            self.last_shield_reason = reason
            reward = float(reward) - self.intervention_penalty

        info = dict(info)
        info.update(
            {
                "shield_interventions": self.shield_interventions,
                "shield_blocks": self.shield_blocks,
                "shield_last_intervention": bool(intervened),
                "shield_reason": reason if intervened else "",
                "shielded_action": tuple(float(v) for v in safe_action),
                "raw_action": tuple(float(v) for v in raw_action),
            }
        )

        if len(done_result) == 5:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def _base_env(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "_in_collision"):
            env = env.env
        return env

    def _filter_action(self, env, action: np.ndarray) -> tuple[np.ndarray, bool, str]:
        action = np.asarray(action, dtype=np.float32)
        unsafe, clearance = self._is_unsafe(env, action)
        if not unsafe and clearance >= self.min_clearance:
            return action, False, ""

        safe_candidates: list[tuple[float, np.ndarray]] = []
        for candidate in self.candidates:
            candidate_unsafe, candidate_clearance = self._is_unsafe(env, candidate)
            if candidate_unsafe:
                continue
            score = self._candidate_score(env, candidate, candidate_clearance)
            safe_candidates.append((score, candidate.copy()))

        if not safe_candidates:
            return np.array([0.0, 0.0], dtype=np.float32), True, "all shield candidates unsafe"

        _, best = max(safe_candidates, key=lambda item: item[0])
        reason = "collision lookahead" if unsafe else f"clearance {clearance:.2f} < {self.min_clearance:.2f}"
        return best, True, reason

    def _is_unsafe(self, env, action: np.ndarray) -> tuple[bool, float]:
        min_clearance = float("inf")
        points = self._rollout_points(env, action)
        for x, y, _yaw in points:
            if env._in_collision(x, y):
                return True, 0.0
            min_clearance = min(min_clearance, self._clearance(env, x, y))
        return False, min_clearance

    def _rollout_points(self, env, action: np.ndarray) -> list[tuple[float, float, float]]:
        v = float(np.clip(action[0], -0.12, 0.22))
        w = float(np.clip(action[1], -2.0, 2.0))
        x = float(env.robot_x)
        y = float(env.robot_y)
        yaw = float(env.robot_yaw)
        points: list[tuple[float, float, float]] = []

        for _ in range(self.lookahead_steps):
            next_yaw = wrap_angle(yaw + w * env.dt)
            next_x = x + v * math.cos(next_yaw) * env.dt
            next_y = y + v * math.sin(next_yaw) * env.dt
            for i in range(1, self.interpolation_steps + 1):
                ratio = i / self.interpolation_steps
                interp_x = x + (next_x - x) * ratio
                interp_y = y + (next_y - y) * ratio
                interp_yaw = wrap_angle(yaw + wrap_angle(next_yaw - yaw) * ratio)
                points.append((interp_x, interp_y, interp_yaw))
            x, y, yaw = next_x, next_y, next_yaw
        return points

    def _candidate_score(self, env, action: np.ndarray, clearance: float) -> float:
        rollout = self._rollout_points(env, action)
        x, y, yaw = rollout[-1]
        current_dist = math.hypot(env.goal_x - env.robot_x, env.goal_y - env.robot_y)
        next_dist = math.hypot(env.goal_x - x, env.goal_y - y)
        progress = current_dist - next_dist
        goal_angle = math.atan2(env.goal_y - y, env.goal_x - x)
        heading_bonus = math.cos(wrap_angle(goal_angle - yaw))
        v = float(action[0])
        w = float(action[1])
        reverse_penalty = max(-v, 0.0) * 0.45
        turn_penalty = abs(w) * 0.025
        wall_bonus = min(clearance, 1.2) * 0.35
        return progress * 8.0 + heading_bonus * 0.8 + wall_bonus + v * 0.8 - reverse_penalty - turn_penalty

    def _clearance(self, env, x: float, y: float) -> float:
        best = float("inf")
        for obs in env.collision_obstacles:
            if obs.contains(x, y):
                return 0.0
            cx, cy = obs.center
            sx, sy = obs.size
            xmin = cx - sx / 2.0
            xmax = cx + sx / 2.0
            ymin = cy - sy / 2.0
            ymax = cy + sy / 2.0
            dx = max(xmin - x, 0.0, x - xmax)
            dy = max(ymin - y, 0.0, y - ymax)
            best = min(best, math.hypot(dx, dy))
        best = min(best, x - env.x_bounds[0], env.x_bounds[1] - x)
        best = min(best, y - env.y_bounds[0], env.y_bounds[1] - y)
        return max(best, 0.0)
