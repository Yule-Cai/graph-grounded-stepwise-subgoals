from __future__ import annotations

import heapq
import math
from typing import Any

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover
    try:
        import gym
    except ImportError:  # pragma: no cover
        gym = None


class WaypointGoalEnv(gym.Wrapper if gym else object):
    """Split long navigation episodes into short, reachable waypoint goals."""

    def __init__(
        self,
        env,
        waypoint_spacing: float = 3.2,
        grid_resolution: float = 0.8,
        waypoint_reward: float = 80.0,
        failed_plan_penalty: float = 40.0,
        final_distance_penalty: float = 0.0,
        incomplete_waypoint_penalty: float = 0.0,
        max_astar_nodes: int = 18000,
    ):
        if gym is None:
            raise RuntimeError("gymnasium or gym is required to use WaypointGoalEnv")
        super().__init__(env)
        self.waypoint_spacing = float(waypoint_spacing)
        self.grid_resolution = float(grid_resolution)
        self.waypoint_reward = float(waypoint_reward)
        self.failed_plan_penalty = float(failed_plan_penalty)
        self.final_distance_penalty = float(final_distance_penalty)
        self.incomplete_waypoint_penalty = float(incomplete_waypoint_penalty)
        self.max_astar_nodes = int(max_astar_nodes)
        self.final_goal: tuple[float, float] | None = None
        self.waypoints: list[tuple[float, float]] = []
        self.waypoint_index = 0
        self.plan_failed = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        base = self._base_env()
        self.final_goal = (float(base.goal_x), float(base.goal_y))
        self.waypoints = self._plan_waypoints(
            (float(base.robot_x), float(base.robot_y)),
            self.final_goal,
        )
        self.waypoint_index = 0
        self.plan_failed = not self.waypoints
        if not self.waypoints:
            self.waypoints = [self.final_goal]
        self._set_active_waypoint(base, self.waypoints[0])
        obs = base._observation()
        info = base._info()
        info.update(self._waypoint_info(base))
        if isinstance(result, tuple) and len(result) == 2:
            return obs, info
        return obs

    def step(self, action):
        base = self._base_env()
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            gymnasium_api = True
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False
            gymnasium_api = False

        reward = float(reward)
        info = dict(info)
        reached_active = base._distance_to_goal() <= base.success_radius
        final_index = self.waypoint_index >= len(self.waypoints) - 1

        if reached_active and not final_index and not bool(info.get("collided", False)):
            self.waypoint_index += 1
            self._set_active_waypoint(base, self.waypoints[self.waypoint_index])
            obs = base._observation()
            reward -= self._base_terminal_success_bonus(base)
            reward += self.waypoint_reward
            info["intermediate_waypoint_reward"] = self.waypoint_reward
            terminated = False
            truncated = False
        elif self.plan_failed and (terminated or truncated):
            reward -= self.failed_plan_penalty

        collided = bool(info.get("collided", False))
        if final_index and reached_active and not collided:
            info["distance_to_final_goal"] = 0.0
        elif self.final_goal is not None:
            info["distance_to_final_goal"] = math.hypot(
                self.final_goal[0] - base.robot_x,
                self.final_goal[1] - base.robot_y,
            )

        final_dist = float(info.get("distance_to_final_goal", 0.0))
        final_reached = final_dist <= base.success_radius and not collided
        if (terminated or truncated) and not final_reached and self.final_goal is not None:
            remaining = max(len(self.waypoints) - self.waypoint_index - 1, 0)
            reward -= final_dist * self.final_distance_penalty
            reward -= remaining * self.incomplete_waypoint_penalty
            info["waypoint_final_debt_penalty"] = (
                final_dist * self.final_distance_penalty + remaining * self.incomplete_waypoint_penalty
            )
        else:
            info["waypoint_final_debt_penalty"] = 0.0

        info.update(self._waypoint_info(base))
        if gymnasium_api:
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info

    def _base_env(self):
        env = self.env
        while hasattr(env, "env") and not hasattr(env, "_in_collision"):
            env = env.env
        return env

    def _set_active_waypoint(self, base, waypoint: tuple[float, float]) -> None:
        base.goal_x, base.goal_y = float(waypoint[0]), float(waypoint[1])
        base.prev_dist = base._distance_to_goal()
        base.best_dist = base.prev_dist
        base.no_progress_steps = 0

    def _base_terminal_success_bonus(self, base) -> float:
        progress_factor = 1.0 - base.steps / max(base.max_steps, 1)
        if getattr(base, "reward_profile", "") == "v8_goal":
            return 1500.0 + 420.0 * progress_factor
        if getattr(base, "reward_profile", "") == "v7_goal":
            return 420.0 + 90.0 * progress_factor
        return 220.0 + 40.0 * progress_factor

    def _waypoint_info(self, base) -> dict[str, Any]:
        return {
            "waypoint_index": self.waypoint_index,
            "waypoint_count": len(self.waypoints),
            "active_waypoint": (float(base.goal_x), float(base.goal_y)),
            "final_goal": self.final_goal,
            "waypoint_plan_failed": self.plan_failed,
        }

    def _plan_waypoints(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> list[tuple[float, float]]:
        base = self._base_env()
        if self._line_is_free(base, start, goal):
            return self._split_segment(start, goal)[1:]
        path = self._astar_path(base, start, goal)
        if not path:
            return self._split_segment(start, goal)[1:]
        simplified = self._simplify_path(base, path)
        spaced: list[tuple[float, float]] = []
        current = start
        for point in simplified[1:]:
            segment = self._split_segment(current, point)[1:]
            spaced.extend(segment)
            current = point
        if not spaced or math.hypot(spaced[-1][0] - goal[0], spaced[-1][1] - goal[1]) > 0.5:
            spaced.append(goal)
        return spaced

    def _astar_path(self, base, start: tuple[float, float], goal: tuple[float, float]):
        resolution = self.grid_resolution
        x_min, x_max = base.x_bounds
        y_min, y_max = base.y_bounds
        width = int(math.ceil((x_max - x_min) / resolution)) + 1
        height = int(math.ceil((y_max - y_min) / resolution)) + 1

        def to_cell(point: tuple[float, float]) -> tuple[int, int]:
            x, y = point
            return (
                min(max(int(round((x - x_min) / resolution)), 0), width - 1),
                min(max(int(round((y - y_min) / resolution)), 0), height - 1),
            )

        def to_world(cell: tuple[int, int]) -> tuple[float, float]:
            return (x_min + cell[0] * resolution, y_min + cell[1] * resolution)

        def free(cell: tuple[int, int]) -> bool:
            x, y = to_world(cell)
            return not base._in_collision(x, y)

        start_cell = self._nearest_free_cell(to_cell(start), free, width, height)
        goal_cell = self._nearest_free_cell(to_cell(goal), free, width, height)
        if start_cell is None or goal_cell is None:
            return []

        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_cell))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start_cell: 0.0}
        visited = 0
        neighbors = (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        )

        while open_heap and visited < self.max_astar_nodes:
            _priority, current = heapq.heappop(open_heap)
            visited += 1
            if current == goal_cell:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                return [start, *[to_world(cell) for cell in cells[1:-1]], goal]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                    continue
                if not free(neighbor):
                    continue
                current_world = to_world(current)
                neighbor_world = to_world(neighbor)
                if not self._line_is_free(base, current_world, neighbor_world):
                    continue
                step_cost = math.hypot(dx, dy)
                tentative = g_score[current] + step_cost
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                heuristic = math.hypot(goal_cell[0] - neighbor[0], goal_cell[1] - neighbor[1])
                heapq.heappush(open_heap, (tentative + heuristic, neighbor))
        return []

    def _nearest_free_cell(self, cell, free, width: int, height: int):
        if free(cell):
            return cell
        for radius in range(1, 8):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (cell[0] + dx, cell[1] + dy)
                    if 0 <= candidate[0] < width and 0 <= candidate[1] < height and free(candidate):
                        candidates.append(candidate)
            if candidates:
                return min(candidates, key=lambda item: math.hypot(item[0] - cell[0], item[1] - cell[1]))
        return None

    def _simplify_path(self, base, path: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(path) <= 2:
            return path
        simplified = [path[0]]
        anchor = 0
        while anchor < len(path) - 1:
            next_index = len(path) - 1
            while next_index > anchor + 1 and not self._line_is_free(base, path[anchor], path[next_index]):
                next_index -= 1
            simplified.append(path[next_index])
            anchor = next_index
        return simplified

    def _split_segment(self, start: tuple[float, float], goal: tuple[float, float]) -> list[tuple[float, float]]:
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if distance <= self.waypoint_spacing:
            return [start, goal]
        steps = max(1, int(math.ceil(distance / self.waypoint_spacing)))
        return [
            (
                start[0] + (goal[0] - start[0]) * i / steps,
                start[1] + (goal[1] - start[1]) * i / steps,
            )
            for i in range(steps + 1)
        ]

    def _line_is_free(self, base, start: tuple[float, float], goal: tuple[float, float]) -> bool:
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        samples = max(2, int(math.ceil(distance / max(base.robot_radius, 0.18))))
        for i in range(samples + 1):
            ratio = i / samples
            x = start[0] + (goal[0] - start[0]) * ratio
            y = start[1] + (goal[1] - start[1]) * ratio
            if base._in_collision(x, y):
                return False
        return True
