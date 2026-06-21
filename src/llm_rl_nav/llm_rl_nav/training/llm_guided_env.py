from __future__ import annotations

import json
import math
from typing import Any
from urllib import error, request

import numpy as np

from llm_rl_nav.utils import wrap_angle

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    try:
        import gym
    except ImportError:  # pragma: no cover
        gym = None


class LLMGuidedRecoveryEnv(gym.Wrapper if gym else object):
    """Low-frequency LM Studio teacher for hard recovery states.

    PPO still trains the robot. The LLM is only queried when the robot is close
    to obstacles or has made little goal progress, then the chosen short
    waypoint becomes a temporary teacher signal.
    """

    def __init__(
        self,
        env,
        base_url: str,
        model: str,
        trigger_clearance: float = 0.72,
        max_calls_per_episode: int = 3,
        cooldown_steps: int = 95,
        teacher_force_prob: float = 0.18,
        timeout_s: float = 4.0,
    ):
        if gym:
            super().__init__(env)
        else:
            self.env = env
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.trigger_clearance = trigger_clearance
        self.max_calls_per_episode = max_calls_per_episode
        self.cooldown_steps = cooldown_steps
        self.teacher_force_prob = teacher_force_prob
        self.timeout_s = timeout_s
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.max_steps = env.max_steps
        self._rng = np.random.default_rng(17)
        self._guide_target: dict[str, float] | None = None
        self._guide_ttl = 0
        self._cooldown = 0
        self._calls = 0
        self._failures = 0
        self._no_progress_steps = 0
        self._last_goal_dist = math.inf

    def reset(self, **kwargs):
        self._guide_target = None
        self._guide_ttl = 0
        self._cooldown = 0
        self._calls = 0
        self._failures = 0
        self._no_progress_steps = 0
        result = self.env.reset(**kwargs)
        info = result[1] if isinstance(result, tuple) else self.env._info()
        self._last_goal_dist = float(info.get("distance_to_goal", math.inf))
        return result

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        guide_before = self._distance_to_guide()
        if self._guide_target and self._rng.random() < self.teacher_force_prob:
            action = self._teacher_action(self._guide_target)

        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            gymnasium_step = True
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
            gymnasium_step = False

        reward = float(reward)
        self._cooldown = max(0, self._cooldown - 1)
        self._guide_ttl = max(0, self._guide_ttl - 1)

        goal_dist = float(info.get("distance_to_goal", math.inf))
        if goal_dist < self._last_goal_dist - 0.015:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1
        self._last_goal_dist = goal_dist

        if self._guide_target:
            guide_after = self._distance_to_guide()
            if math.isfinite(guide_before) and math.isfinite(guide_after):
                reward += max(min((guide_before - guide_after) * 18.0, 2.0), -1.0)
            if guide_after < 0.45:
                reward += 4.0
                self._guide_target = None
                self._guide_ttl = 0
                self._no_progress_steps = 0
            elif self._guide_ttl <= 0:
                self._guide_target = None

        if not done and self._should_query_llm(info):
            waypoint = self._request_guide(info)
            if waypoint:
                self._guide_target = waypoint
                self._guide_ttl = 120
                self._cooldown = self.cooldown_steps
                reward += 0.5

        info = dict(info)
        info.update(
            {
                "llm_guidance_calls": self._calls,
                "llm_guidance_failures": self._failures,
                "llm_guidance_active": bool(self._guide_target),
                "llm_guide_target": None if not self._guide_target else (self._guide_target["x"], self._guide_target["y"]),
            }
        )

        if gymnasium_step:
            return obs, reward, terminated, truncated, info
        return obs, reward, done, info

    def _should_query_llm(self, info: dict[str, Any]) -> bool:
        if self._guide_target or self._cooldown > 0 or self._calls >= self.max_calls_per_episode:
            return False
        min_lidar = float(info.get("min_lidar", 999.0))
        near_wall_for_a_while = int(info.get("near_wall_steps", 0) or 0) > 16
        return min_lidar < self.trigger_clearance or self._no_progress_steps > 65 or near_wall_for_a_while

    def _request_guide(self, info: dict[str, Any]) -> dict[str, float] | None:
        candidates = self._candidate_waypoints()
        if not candidates:
            return None
        self._calls += 1
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an LLM teacher for PPO robot navigation training. "
                        "Choose one safe short recovery waypoint from candidate_waypoints. "
                        "Do not invent coordinates. Return JSON only: "
                        "{\"candidate_index\":0,\"rationale\":\"short reason\"}."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "map_id": info.get("map_id"),
                            "robot": {"x": self.env.robot_x, "y": self.env.robot_y, "yaw": self.env.robot_yaw},
                            "goal": {"x": self.env.goal_x, "y": self.env.goal_y},
                            "goal_vector": {
                                "dx": self.env.goal_x - self.env.robot_x,
                                "dy": self.env.goal_y - self.env.robot_y,
                                "distance": info.get("distance_to_goal"),
                            },
                            "min_lidar": info.get("min_lidar"),
                            "candidate_waypoints": candidates[:6],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 80,
        }
        try:
            req = request.Request(
                self._chat_url(),
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
            )
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = _json_from_text(content)
            index = int(parsed.get("candidate_index", parsed.get("selected_candidate", 0)))
            selected = candidates[index]
            return {"x": float(selected["x"]), "y": float(selected["y"])}
        except (error.URLError, TimeoutError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            self._failures += 1
            if candidates and self._failures <= self.max_calls_per_episode:
                selected = candidates[0]
                return {"x": float(selected["x"]), "y": float(selected["y"])}
            return None

    def _candidate_waypoints(self) -> list[dict[str, float]]:
        candidates: list[dict[str, float]] = []
        goal_angle = math.atan2(self.env.goal_y - self.env.robot_y, self.env.goal_x - self.env.robot_x)
        for radius in (1.8, 2.8, 4.0, 5.2):
            for offset in (0, -0.35, 0.35, -0.75, 0.75, -1.25, 1.25, math.pi / 2, -math.pi / 2, math.pi):
                angle = goal_angle + offset
                x = self.env.robot_x + math.cos(angle) * radius
                y = self.env.robot_y + math.sin(angle) * radius
                if self.env._in_collision(x, y) or not self._line_safe(x, y):
                    continue
                clearance = self._clearance_at(x, y)
                goal_dist = math.hypot(self.env.goal_x - x, self.env.goal_y - y)
                score = -goal_dist + clearance * 2.4 - abs(offset) * 0.25
                candidates.append({"x": round(x, 3), "y": round(y, 3), "clearance": round(clearance, 3), "score": round(score, 3)})
        candidates.sort(key=lambda item: item["score"], reverse=True)
        return candidates[:10]

    def _teacher_action(self, waypoint: dict[str, float]) -> np.ndarray:
        desired = math.atan2(waypoint["y"] - self.env.robot_y, waypoint["x"] - self.env.robot_x)
        error = wrap_angle(desired - self.env.robot_yaw)
        speed = 0.18 if abs(error) < 0.7 else 0.07
        turn = max(min(error * 2.1, 2.0), -2.0)
        return np.array([speed, turn], dtype=np.float32)

    def _distance_to_guide(self) -> float:
        if not self._guide_target:
            return math.inf
        return math.hypot(self._guide_target["x"] - self.env.robot_x, self._guide_target["y"] - self.env.robot_y)

    def _line_safe(self, x: float, y: float) -> bool:
        steps = max(4, int(math.hypot(x - self.env.robot_x, y - self.env.robot_y) / 0.18))
        for idx in range(1, steps + 1):
            t = idx / steps
            px = self.env.robot_x + (x - self.env.robot_x) * t
            py = self.env.robot_y + (y - self.env.robot_y) * t
            if self.env._in_collision(px, py):
                return False
        return True

    def _clearance_at(self, x: float, y: float) -> float:
        best = 999.0
        for obs in self.env.collision_obstacles:
            cx, cy = obs.center
            sx, sy = obs.size
            dx = max(abs(x - cx) - sx / 2, 0.0)
            dy = max(abs(y - cy) - sy / 2, 0.0)
            best = min(best, math.hypot(dx, dy))
        return min(best, self.env.lidar_range)

    def _chat_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"


def _json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM did not return JSON")
    return json.loads(cleaned[start : end + 1])
