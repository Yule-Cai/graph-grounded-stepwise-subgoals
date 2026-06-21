from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from llm_rl_nav.utils import wrap_angle

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - supports the older dependency file.
    try:
        import gym
        from gym import spaces
    except ImportError:  # Allows smoke tests before RL dependencies are installed.
        gym = None
        spaces = None


@dataclass(frozen=True)
class RectObstacle:
    name: str
    center: tuple[float, float]
    size: tuple[float, float]

    def expanded(self, margin: float) -> "RectObstacle":
        return RectObstacle(
            name=self.name,
            center=self.center,
            size=(self.size[0] + 2 * margin, self.size[1] + 2 * margin),
        )

    def contains(self, x: float, y: float) -> bool:
        cx, cy = self.center
        sx, sy = self.size
        return abs(x - cx) <= sx / 2 and abs(y - cy) <= sy / 2


@dataclass(frozen=True)
class SemanticMapSpec:
    map_id: str
    label: str
    spawn: tuple[float, float, float]
    obstacles: tuple[RectObstacle, ...]
    goal_points: tuple[tuple[float, float], ...]
    x_bounds: tuple[float, float] = (-13.2, 13.2)
    y_bounds: tuple[float, float] = (-29.2, 29.2)


ALL_SEMANTIC_MAP_IDS = (
    "reference_family_flat",
    "reference_villa_ground",
    "studio_apartment",
    "two_bedroom_apartment",
    "bungalow_house",
    "courtyard_house",
    "suburban_villa",
    "townhouse_long",
    "duplex_family",
    "open_plan_house",
    "narrow_lot_house",
    "luxury_villa",
)

TB3_GAZEBO_SHARE = Path("/opt/homebrew/Caskroom/miniforge/base/envs/ros_env/share/turtlebot3_gazebo")


class Hospital2DNavEnv(gym.Env if gym else object):
    """Fast 2D full-map RL environment matching hospital_primitives.world.

    This environment is intentionally constraint-free. It teaches baseline
    navigation over the whole map first; forbidden zones can be injected later
    as symbolic constraints or action shields.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 1200,
        lidar_bins: int = 60,
        lidar_range: float = 3.5,
        robot_radius: float = 0.18,
        success_radius: float = 0.75,
        dt: float = 0.15,
        seed: int | None = None,
        map_id: str = "hospital",
        map_ids: list[str] | tuple[str, ...] | None = None,
        reward_profile: str = "v7_goal",
        goal_min_distance: float = 8.0,
        goal_max_distance: float | None = None,
        goal_point_probability: float = 0.75,
        spawn_probability: float = 0.35,
        map_specs_override: dict[str, SemanticMapSpec] | None = None,
    ):
        self.max_steps = max_steps
        self.lidar_bins = lidar_bins
        self.lidar_range = lidar_range
        self.robot_radius = robot_radius
        self.success_radius = success_radius
        self.dt = dt
        self.reward_profile = reward_profile
        self.goal_min_distance = goal_min_distance
        self.goal_max_distance = goal_max_distance
        self.goal_point_probability = goal_point_probability
        self.spawn_probability = spawn_probability
        self.rng = np.random.default_rng(seed)
        self.map_specs = map_specs_override or semantic_map_specs()
        self.map_ids = tuple(map_ids or (map_id,))
        unknown = [item for item in self.map_ids if item not in self.map_specs]
        if unknown:
            raise ValueError(f"Unknown semantic map id(s): {', '.join(unknown)}")

        self.active_map_id = ""
        self.active_map_label = ""
        self.goal_points: tuple[tuple[float, float], ...] = ()
        self.reachability_resolution = 0.45
        self._reachability_components: dict[tuple[int, int], int] = {}
        self._reachability_free_cells: set[tuple[int, int]] = set()
        self._set_map(self.map_ids[0])

        self.action_space = _box([-0.12, -2.0], [0.22, 2.0], dtype=np.float32)
        self.observation_space = _box(
            [-1.0] * lidar_bins + [0.0, -math.pi],
            [1.0] * lidar_bins + [1.0, math.pi],
            dtype=np.float32,
        )

        self.robot_x = 0.0
        self.robot_y = -25.0
        self.robot_yaw = math.pi / 2
        self.goal_x = 0.0
        self.goal_y = 25.0
        self.prev_dist = 0.0
        self.best_dist = 0.0
        self.no_progress_steps = 0
        self.near_wall_steps = 0
        self.steps = 0

    def _set_map(self, map_id: str) -> None:
        spec = self.map_specs[map_id]
        self.active_map_id = spec.map_id
        self.active_map_label = spec.label
        self.x_bounds = spec.x_bounds
        self.y_bounds = spec.y_bounds
        self.obstacles = list(spec.obstacles)
        self.goal_points = spec.goal_points
        self.collision_obstacles = [obs.expanded(self.robot_radius) for obs in self.obstacles]
        self._build_reachability_index()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}
        requested_map = options.get("map_id")
        if requested_map is not None:
            self._set_map(str(requested_map))
        elif len(self.map_ids) > 1:
            self._set_map(str(self.rng.choice(self.map_ids)))

        start = options.get("start")
        goal = options.get("goal")
        yaw = options.get("yaw")

        if start is None:
            spawn = self.map_specs[self.active_map_id].spawn
            if self.rng.random() < self.spawn_probability and not self._in_collision(spawn[0], spawn[1]):
                self.robot_x, self.robot_y = spawn[0], spawn[1]
                yaw = spawn[2] if yaw is None else yaw
            else:
                self.robot_x, self.robot_y = self._sample_free_point()
        else:
            self.robot_x, self.robot_y = float(start[0]), float(start[1])

        if goal is None:
            self.goal_x, self.goal_y = self._sample_goal_far_from(self.robot_x, self.robot_y)
        else:
            self.goal_x, self.goal_y = float(goal[0]), float(goal[1])

        self.robot_yaw = float(yaw) if yaw is not None else self.rng.uniform(-math.pi, math.pi)
        self.prev_dist = self._distance_to_goal()
        self.best_dist = self.prev_dist
        self.no_progress_steps = 0
        self.near_wall_steps = 0
        self.steps = 0

        obs = self._observation()
        info = self._info()
        if gym and gym.__name__ == "gymnasium":
            return obs, info
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        v = float(np.clip(action[0], -0.12, 0.22))
        w = float(np.clip(action[1], -2.0, 2.0))

        old_x, old_y, old_yaw = self.robot_x, self.robot_y, self.robot_yaw
        self.robot_yaw = wrap_angle(self.robot_yaw + w * self.dt)
        self.robot_x += v * math.cos(self.robot_yaw) * self.dt
        self.robot_y += v * math.sin(self.robot_yaw) * self.dt
        self.steps += 1

        collided = self._in_collision(self.robot_x, self.robot_y)
        if collided:
            self.robot_x, self.robot_y, self.robot_yaw = old_x, old_y, old_yaw

        obs = self._observation()
        dist = self._distance_to_goal()
        heading_error = self._heading_error()
        min_lidar = float(np.min(obs[: self.lidar_bins]) * self.lidar_range)

        progress = self.prev_dist - dist
        if self.reward_profile == "v8_goal":
            reward = progress * 240.0
            reward -= abs(heading_error) * 0.012
            reward += max(v, 0.0) * 0.34
            reward -= max(-v, 0.0) * 0.03
            reward -= abs(w) * 0.006
            reward -= 0.03
            if abs(v) < 0.02 and abs(w) > 0.9:
                reward -= 0.045
        elif self.reward_profile == "v7_goal":
            reward = progress * 175.0
            reward -= abs(heading_error) * 0.018
            reward += max(v, 0.0) * 0.28
            reward -= max(-v, 0.0) * 0.04
            reward -= abs(w) * 0.009
            reward -= 0.018
            if abs(v) < 0.02 and abs(w) > 0.9:
                reward -= 0.025
        else:
            reward = progress * 120.0
            reward -= abs(heading_error) * 0.025
            reward += v * 0.18
            reward -= abs(w) * 0.012
            reward -= 0.025
            if abs(v) < 0.02 and abs(w) > 0.9:
                reward -= 0.035

        if dist < self.best_dist - 0.015:
            self.best_dist = dist
            self.no_progress_steps = 0
            reward += 0.24 if self.reward_profile == "v8_goal" else (0.18 if self.reward_profile == "v7_goal" else 0.08)
        else:
            self.no_progress_steps += 1
        if self.reward_profile == "v8_goal":
            if self.no_progress_steps > 18:
                reward -= min((self.no_progress_steps - 18) * 0.16, 12.0)
        elif self.reward_profile == "v7_goal":
            if self.no_progress_steps > 28:
                reward -= min((self.no_progress_steps - 28) * 0.085, 7.0)
        elif self.no_progress_steps > 45:
            reward -= min((self.no_progress_steps - 45) * 0.04, 4.0)
        self.prev_dist = dist

        terminated = False
        stalled = (
            self.reward_profile in ("v7_goal", "v8_goal")
            and self.no_progress_steps > (110 if self.reward_profile == "v8_goal" else 150)
            and dist > self.success_radius * 1.4
        )
        truncated = self.steps >= self.max_steps or stalled

        if self.reward_profile == "v8_goal":
            if min_lidar < 0.44:
                reward -= (0.44 - min_lidar) * 5.0
            if min_lidar < 0.29:
                reward -= (0.29 - min_lidar) * 12.0
        elif self.reward_profile == "v7_goal":
            if min_lidar < 0.48:
                reward -= (0.48 - min_lidar) * 6.0
            if min_lidar < 0.32:
                reward -= (0.32 - min_lidar) * 14.0
        else:
            if min_lidar < 0.55:
                reward -= (0.55 - min_lidar) * 8.0
            if min_lidar < 0.38:
                reward -= (0.38 - min_lidar) * 16.0
        if min_lidar < 0.5:
            self.near_wall_steps += 1
        else:
            self.near_wall_steps = max(0, self.near_wall_steps - 2)
        if self.near_wall_steps > 10:
            scale = 0.05 if self.reward_profile == "v8_goal" else (0.055 if self.reward_profile == "v7_goal" else 0.08)
            cap = 2.0 if self.reward_profile == "v8_goal" else (2.2 if self.reward_profile == "v7_goal" else 3.0)
            reward -= min((self.near_wall_steps - 10) * scale, cap)
        if collided:
            reward -= 260.0 if self.reward_profile == "v8_goal" else (170.0 if self.reward_profile == "v7_goal" else 130.0)
            terminated = True
        elif dist < self.success_radius:
            if self.reward_profile == "v8_goal":
                reward += 1500.0 + 420.0 * (1.0 - self.steps / max(self.max_steps, 1))
            elif self.reward_profile == "v7_goal":
                reward += 420.0 + 90.0 * (1.0 - self.steps / max(self.max_steps, 1))
            else:
                reward += 220.0 + 40.0 * (1.0 - self.steps / max(self.max_steps, 1))
            terminated = True
        elif stalled:
            if self.reward_profile == "v8_goal":
                reward -= 520.0 + dist * 42.0
            else:
                reward -= 150.0 + dist * 4.0
        elif truncated:
            if self.reward_profile == "v8_goal":
                reward -= 700.0 + dist * 34.0
            elif self.reward_profile == "v7_goal":
                reward -= 180.0 + dist * 5.0
            else:
                reward -= 90.0 + dist * 2.0

        info = self._info()
        info.update({"collided": collided, "min_lidar": min_lidar, "stalled": stalled})

        if gym and gym.__name__ == "gymnasium":
            return obs, reward, terminated, truncated, info
        return obs, reward, terminated or truncated, info

    def _observation(self) -> np.ndarray:
        lidar = self._simulate_lidar() / self.lidar_range
        dist = min(self._distance_to_goal() / 60.0, 1.0)
        heading = self._heading_error()
        return np.array([*lidar, dist, heading], dtype=np.float32)

    def _simulate_lidar(self) -> np.ndarray:
        readings = np.full(self.lidar_bins, self.lidar_range, dtype=np.float32)
        for i in range(self.lidar_bins):
            angle = self.robot_yaw + (2 * math.pi * i / self.lidar_bins)
            dx = math.cos(angle)
            dy = math.sin(angle)
            readings[i] = self._ray_distance(dx, dy)
        return readings

    def _ray_distance(self, dx: float, dy: float) -> float:
        best = self.lidar_range
        for obs in self.collision_obstacles:
            hit = _ray_rect_intersection(
                self.robot_x,
                self.robot_y,
                dx,
                dy,
                obs,
                self.lidar_range,
            )
            if hit is not None:
                best = min(best, hit)
        return best

    def _sample_free_point(self) -> tuple[float, float]:
        for _ in range(10000):
            x = self.rng.uniform(*self.x_bounds)
            y = self.rng.uniform(*self.y_bounds)
            if not self._in_collision(x, y):
                return x, y
        raise RuntimeError("Could not sample a free point in the hospital map.")

    def _sample_goal_far_from(self, x: float, y: float) -> tuple[float, float]:
        if self.goal_points and self.rng.random() < self.goal_point_probability:
            indices = self.rng.permutation(len(self.goal_points))
            for idx in indices:
                gx, gy = self.goal_points[int(idx)]
                distance = math.hypot(gx - x, gy - y)
                if (
                    not self._in_collision(gx, gy)
                    and self._same_reachable_component((x, y), (gx, gy))
                    and distance >= max(2.0, self.goal_min_distance * 0.65)
                    and (self.goal_max_distance is None or distance <= self.goal_max_distance)
                ):
                    return gx, gy
        for _ in range(10000):
            gx, gy = self._sample_free_point()
            distance = math.hypot(gx - x, gy - y)
            if (
                self._same_reachable_component((x, y), (gx, gy))
                and distance >= self.goal_min_distance
                and (self.goal_max_distance is None or distance <= self.goal_max_distance)
            ):
                return gx, gy
        for _ in range(10000):
            gx, gy = self._sample_free_point()
            if self._same_reachable_component((x, y), (gx, gy)):
                return gx, gy
        return self._sample_free_point()

    def _build_reachability_index(self) -> None:
        resolution = self.reachability_resolution
        xmin, xmax = self.x_bounds
        ymin, ymax = self.y_bounds
        nx = int(math.ceil((xmax - xmin) / resolution))
        ny = int(math.ceil((ymax - ymin) / resolution))
        free_cells: set[tuple[int, int]] = set()
        for ix in range(nx):
            for iy in range(ny):
                x, y = self._reachability_cell_center(ix, iy)
                if not self._in_collision(x, y):
                    free_cells.add((ix, iy))

        components: dict[tuple[int, int], int] = {}
        component_id = 0
        for cell in free_cells:
            if cell in components:
                continue
            queue: deque[tuple[int, int]] = deque([cell])
            components[cell] = component_id
            while queue:
                current = queue.popleft()
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor in free_cells and neighbor not in components:
                        components[neighbor] = component_id
                        queue.append(neighbor)
            component_id += 1

        self._reachability_free_cells = free_cells
        self._reachability_components = components

    def _reachability_cell_center(self, ix: int, iy: int) -> tuple[float, float]:
        resolution = self.reachability_resolution
        return (
            self.x_bounds[0] + (ix + 0.5) * resolution,
            self.y_bounds[0] + (iy + 0.5) * resolution,
        )

    def _nearest_reachability_cell(self, point: tuple[float, float]) -> tuple[int, int] | None:
        if not self._reachability_free_cells:
            return None
        resolution = self.reachability_resolution
        ix = int((point[0] - self.x_bounds[0]) / resolution)
        iy = int((point[1] - self.y_bounds[0]) / resolution)
        best = None
        best_distance = float("inf")
        for radius in range(0, 10):
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if radius and max(abs(dx), abs(dy)) != radius:
                        continue
                    cell = (ix + dx, iy + dy)
                    if cell in self._reachability_free_cells:
                        candidates.append(cell)
            if candidates:
                for cell in candidates:
                    x, y = self._reachability_cell_center(*cell)
                    distance = math.hypot(x - point[0], y - point[1])
                    if distance < best_distance:
                        best_distance = distance
                        best = cell
                return best
        return None

    def _same_reachable_component(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
    ) -> bool:
        start_cell = self._nearest_reachability_cell(start)
        goal_cell = self._nearest_reachability_cell(goal)
        if start_cell is None or goal_cell is None:
            return False
        return self._reachability_components.get(start_cell) == self._reachability_components.get(goal_cell)

    def _in_collision(self, x: float, y: float) -> bool:
        if not (self.x_bounds[0] <= x <= self.x_bounds[1] and self.y_bounds[0] <= y <= self.y_bounds[1]):
            return True
        return any(obs.contains(x, y) for obs in self.collision_obstacles)

    def _distance_to_goal(self) -> float:
        return math.hypot(self.goal_x - self.robot_x, self.goal_y - self.robot_y)

    def _heading_error(self) -> float:
        target_angle = math.atan2(self.goal_y - self.robot_y, self.goal_x - self.robot_x)
        return wrap_angle(target_angle - self.robot_yaw)

    def _info(self) -> dict[str, Any]:
        return {
            "robot_pose": (self.robot_x, self.robot_y, self.robot_yaw),
            "goal": (self.goal_x, self.goal_y),
            "distance_to_goal": self._distance_to_goal(),
            "success_radius": self.success_radius,
            "best_distance_to_goal": self.best_dist,
            "no_progress_steps": self.no_progress_steps,
            "near_wall_steps": self.near_wall_steps,
            "steps": self.steps,
            "map_id": self.active_map_id,
            "map_label": self.active_map_label,
        }


def semantic_map_specs() -> dict[str, SemanticMapSpec]:
    return {
        "reference_family_flat": SemanticMapSpec(
            "reference_family_flat",
            "reference Chinese family flat with living room, dining room, kitchen, bathroom, bedrooms, study and entrance hall",
            (8.5, -18.0, math.pi / 2),
            tuple(_reference_family_flat_obstacles()),
            (
                (-18.0, 14.0),
                (-21.5, 3.5),
                (-19.0, -8.0),
                (-8.5, 10.0),
                (3.5, 14.0),
                (17.0, 12.0),
                (19.0, 5.0),
                (18.0, -7.5),
                (8.5, -15.5),
                (-2.5, -6.0),
                (-7.0, -18.9),
            ),
            x_bounds=(-26.0, 26.0),
            y_bounds=(-20.0, 20.0),
        ),
        "reference_villa_ground": SemanticMapSpec(
            "reference_villa_ground",
            "villa ground floor with foyer, living room, dining room, kitchen, laundry, elder bedroom, bathrooms, craft room and stairs",
            (13.0, -22.0, math.pi / 2),
            tuple(_reference_villa_ground_obstacles()),
            (
                (15.0, -20.5),
                (16.8, -12.4),
                (15.4, -9.5),
                (13.8, 7.0),
                (17.0, 15.0),
                (-16.0, 16.0),
                (-12.0, 11.4),
                (-7.0, 10.2),
                (-14.2, 0.4),
                (-5.3, -5.0),
                (-4.8, -6.4),
                (-14.2, -11.2),
                (-8.5, -18.5),
                (-0.2, 4.6),
            ),
            x_bounds=(-21.0, 21.0),
            y_bounds=(-24.0, 24.0),
        ),
        "studio_apartment": SemanticMapSpec(
            "studio_apartment",
            "studio apartment with compact living, kitchen, bathroom and bedroom nook",
            (-10.0, -24.0, 0.35),
            tuple(_studio_apartment_obstacles()),
            ((-9.5, 24.0), (-6.0, -12.0), (8.0, -24.0), (-11.5, 16.0), (9.0, 6.5)),
        ),
        "two_bedroom_apartment": SemanticMapSpec(
            "two_bedroom_apartment",
            "two-bedroom apartment with living room, kitchen, bathroom and bedrooms",
            (-10.5, -24.5, 0.35),
            tuple(_two_bedroom_apartment_obstacles()),
            ((-10.0, 25.0), (-4.0, -13.0), (8.5, -24.0), (-10.5, 8.5), (6.0, 9.0), (10.0, 19.0)),
        ),
        "bungalow_house": SemanticMapSpec(
            "bungalow_house",
            "single-floor bungalow with central living room and side bedrooms",
            (-10.5, -25.0, 0.35),
            tuple(_bungalow_house_obstacles()),
            ((-9.5, 24.0), (0.0, -12.0), (8.0, -24.0), (-9.0, 9.0), (8.0, 9.0), (0.0, 20.0)),
        ),
        "courtyard_house": SemanticMapSpec(
            "courtyard_house",
            "single-floor courtyard house with rooms around a small inner court",
            (-10.5, -25.0, 0.35),
            tuple(_courtyard_house_obstacles()),
            ((-9.5, 24.0), (-9.0, -22.0), (9.0, -22.0), (-10.0, 18.0), (9.5, 18.0), (0.0, -12.0)),
        ),
        "suburban_villa": SemanticMapSpec(
            "suburban_villa",
            "villa ground floor with foyer, large living room, kitchen, bathrooms and bedrooms",
            (-16.0, -35.0, 0.35),
            tuple(_suburban_villa_obstacles()),
            ((-6.0, 36.0), (-16.0, -20.0), (16.0, -30.0), (-15.0, 10.0), (13.0, 10.0), (3.5, 23.0)),
            x_bounds=(-22.0, 22.0),
            y_bounds=(-40.0, 40.0),
        ),
        "townhouse_long": SemanticMapSpec(
            "townhouse_long",
            "long narrow townhouse with front living room, middle kitchen and rear bedrooms",
            (0.0, -36.0, math.pi / 2),
            tuple(_townhouse_long_obstacles()),
            ((-3.0, 36.0), (0.0, -25.0), (0.0, -8.0), (-5.5, 20.0), (5.5, 20.0), (6.0, 12.0)),
            x_bounds=(-10.0, 10.0),
            y_bounds=(-40.0, 40.0),
        ),
        "duplex_family": SemanticMapSpec(
            "duplex_family",
            "duplex family home with mirrored bedrooms and shared living/kitchen core",
            (-12.0, -28.0, 0.35),
            tuple(_duplex_family_obstacles()),
            ((-3.0, 26.0), (-8.8, -14.0), (10.8, -20.0), (-9.0, 9.0), (8.5, 9.0), (0.0, 1.0)),
        ),
        "open_plan_house": SemanticMapSpec(
            "open_plan_house",
            "open-plan family house with kitchen island, living area, bathroom and bedrooms",
            (-11.0, -25.0, 0.35),
            tuple(_open_plan_house_obstacles()),
            ((-3.0, 25.0), (-5.0, -14.5), (9.5, -24.0), (-9.5, 12.0), (8.5, 10.0), (0.0, 2.0)),
        ),
        "narrow_lot_house": SemanticMapSpec(
            "narrow_lot_house",
            "narrow-lot house with offset corridor, small kitchen, bathroom and bedrooms",
            (0.0, -36.0, math.pi / 2),
            tuple(_narrow_lot_house_obstacles()),
            ((-3.0, 36.0), (-3.0, -23.0), (3.8, -8.0), (-4.5, 13.0), (3.8, 18.0), (-6.2, 34.0)),
            x_bounds=(-9.0, 9.0),
            y_bounds=(-40.0, 40.0),
        ),
        "luxury_villa": SemanticMapSpec(
            "luxury_villa",
            "large villa with grand living room, kitchen wing, private bedrooms and bathrooms",
            (-17.5, -36.0, 0.35),
            tuple(_luxury_villa_obstacles()),
            ((0.0, 37.0), (-12.0, -18.0), (18.0, -34.0), (-17.0, 13.0), (14.0, 13.0), (4.0, 24.0)),
            x_bounds=(-24.0, 24.0),
            y_bounds=(-42.0, 42.0),
        ),
    }


def _rect(name: str, x: float, y: float, w: float, h: float) -> RectObstacle:
    return RectObstacle(name, (x, y), (w, h))


def _perimeter(width: float = 28, height: float = 60) -> list[RectObstacle]:
    return [
        _rect("outer_north", 0, height / 2, width, 0.35),
        _rect("outer_south", 0, -height / 2, width, 0.35),
        _rect("outer_west", -width / 2, 0, 0.35, height),
        _rect("outer_east", width / 2, 0, 0.35, height),
    ]


def _hwall(name: str, y: float, x1: float, x2: float) -> RectObstacle:
    return _rect(name, (x1 + x2) / 2.0, y, abs(x2 - x1), 0.28)


def _vwall(name: str, x: float, y1: float, y2: float) -> RectObstacle:
    return _rect(name, x, (y1 + y2) / 2.0, 0.28, abs(y2 - y1))


def _reference_family_flat_obstacles() -> list[RectObstacle]:
    """Apartment layout traced from the user-provided Chinese floor plan."""
    return [
        *_perimeter(52.0, 40.0),
        # Left service and guest-bedroom wing.
        _vwall("ref_left_wing_inner_wall_s", -14.5, -20.0, -3.2),
        _vwall("ref_left_wing_inner_wall_m", -14.5, 2.0, 7.5),
        _vwall("ref_left_wing_inner_wall_n", -14.5, 12.0, 20.0),
        _hwall("ref_kitchen_bath_wall", 7.5, -26.0, -14.5),
        _hwall("ref_bath_guest_wall_w", -3.2, -26.0, -18.2),
        _hwall("ref_bath_guest_wall_e", -3.2, -15.8, -14.5),
        # Top dining, study and second-bedroom band.
        _vwall("ref_dining_study_wall", -2.6, 7.5, 20.0),
        _vwall("ref_study_second_bed_wall", 10.8, 7.5, 20.0),
        _hwall("ref_study_bottom_wall_w", 7.5, -2.6, 1.2),
        _hwall("ref_study_bottom_wall_e", 7.5, 4.8, 10.8),
        _hwall("ref_second_bed_bottom_wall_w", 7.5, 10.8, 13.8),
        _hwall("ref_second_bed_bottom_wall_e", 7.5, 16.4, 26.0),
        # Right closet, entrance and master-bedroom wing.
        _hwall("ref_closet_top_wall", 7.5, 16.4, 26.0),
        _hwall("ref_closet_master_wall", -2.8, 16.4, 26.0),
        _vwall("ref_closet_living_wall_s", 16.4, -2.8, 0.2),
        _vwall("ref_closet_living_wall_n", 16.4, 3.2, 7.5),
        _vwall("ref_entry_left_wall_s", 5.0, -20.0, -9.0),
        _vwall("ref_entry_left_wall_n", 5.0, -5.6, -1.2),
        _vwall("ref_entry_right_wall_s", 14.5, -20.0, -12.0),
        _vwall("ref_entry_right_wall_n", 14.5, -7.8, -2.8),
        _hwall("ref_entry_top_wall_w", -5.6, 5.0, 8.0),
        _hwall("ref_entry_top_wall_e", -5.6, 11.0, 14.5),
        # Living room, corridor and balcony hints.
        _hwall("ref_balcony_top_wall_w", -15.0, -12.5, -8.4),
        _hwall("ref_balcony_top_wall_e", -15.0, -4.8, -1.6),
        _vwall("ref_balcony_left_wall", -12.5, -20.0, -15.0),
        _vwall("ref_balcony_right_wall", -1.6, -20.0, -15.0),
        _vwall("ref_living_entry_pier_n", 5.0, 0.4, 7.5),
        _vwall("ref_living_entry_pier_s", 5.0, -5.6, -2.8),
        # Furniture and room fixtures.
        _rect("ref_kitchen_counter", -23.2, 15.7, 4.5, 3.0),
        _rect("ref_bath_shower", -18.5, 3.5, 2.5, 2.5),
        _rect("ref_bath_toilet", -24.0, 2.6, 1.5, 1.8),
        _rect("ref_bath_sink", -22.5, -0.8, 2.6, 1.2),
        _rect("ref_bath_vanity", -18.0, -0.7, 2.1, 1.0),
        _rect("ref_guest_bed", -23.1, -12.5, 4.4, 5.6),
        _rect("ref_guest_desk", -18.0, -5.2, 4.0, 1.0),
        _rect("ref_dining_table", -8.2, 13.5, 4.5, 3.2),
        _rect("ref_study_desk", 7.2, 15.0, 2.8, 2.0),
        _rect("ref_study_bookshelf", 0.6, 15.2, 1.0, 5.0),
        _rect("ref_second_bed", 22.5, 14.0, 5.2, 4.8),
        _rect("ref_closet_storage", 21.0, 2.4, 4.5, 1.2),
        _rect("ref_master_bed", 22.5, -8.8, 5.0, 4.8),
        _rect("ref_master_cabinet", 20.0, -15.6, 5.2, 1.1),
        _rect("ref_living_sofa", -2.5, -4.2, 4.8, 1.5),
        _rect("ref_living_sofa_e", 2.8, -6.4, 1.7, 4.5),
        _rect("ref_living_chair_w", -6.3, -7.0, 2.0, 1.7),
        _rect("ref_living_chair_e", 1.4, -9.6, 2.0, 1.7),
        _rect("ref_living_table", -2.5, -8.2, 3.5, 2.2),
        _rect("ref_balcony_chair_w", -9.8, -17.5, 1.5, 1.2),
        _rect("ref_balcony_chair_e", -4.2, -17.5, 1.5, 1.2),
        _rect("ref_balcony_table", -7.0, -17.8, 1.3, 1.3),
        _rect("ref_entry_cabinet", 12.8, -10.4, 1.0, 5.2),
    ]


def _reference_villa_ground_obstacles() -> list[RectObstacle]:
    """Villa ground-floor layout traced from the second user-provided floor plan."""
    return [
        _hwall("villa_outer_north_left", 22.5, -21.0, -9.2),
        _hwall("villa_outer_north_right", 22.5, 5.2, 21.0),
        _vwall("villa_outer_west_upper", -21.0, -15.2, 22.5),
        _hwall("villa_outer_left_bottom", -15.2, -21.0, -12.2),
        _vwall("villa_outer_balcony_west", -12.2, -21.5, -15.2),
        _hwall("villa_outer_balcony_south", -21.5, -12.2, -3.6),
        _vwall("villa_outer_balcony_east", -3.6, -24.0, -16.8),
        _hwall("villa_outer_hall_south", -24.0, -3.6, 10.2),
        _vwall("villa_outer_foyer_west", 10.2, -24.0, -17.0),
        _hwall("villa_outer_foyer_south", -24.0, 10.2, 21.0),
        _vwall("villa_outer_east", 21.0, -24.0, 22.5),
        _rect("villa_exterior_void_sw", -16.6, -19.6, 8.8, 8.8),
        _rect("villa_exterior_void_s_center", -7.9, -22.75, 8.6, 2.5),
        _rect("villa_exterior_void_top_center", 0.0, 18.05, 10.4, 8.9),
        _hwall("villa_laundry_bottom_wall_w", 13.6, -21.0, -17.2),
        _hwall("villa_laundry_bottom_wall_e", 13.6, -13.8, -9.2),
        _vwall("villa_laundry_right_wall_s", -9.2, 13.6, 16.8),
        _vwall("villa_laundry_right_wall_n", -9.2, 19.8, 22.5),
        _hwall("villa_kitchen_bottom_wall_w", 13.6, 5.2, 8.4),
        _hwall("villa_kitchen_bottom_wall_e", 13.6, 13.2, 21.0),
        _vwall("villa_kitchen_left_wall", 5.2, 13.6, 22.5),
        _vwall("villa_hall_left_wall_s", -3.6, -24.0, -15.4),
        _vwall("villa_hall_left_wall_m", -3.6, -10.4, -3.2),
        _vwall("villa_hall_left_wall_n_lower", -3.6, 4.8, 6.8),
        _vwall("villa_hall_left_wall_n_upper", -3.6, 10.2, 22.5),
        _hwall("villa_craft_bottom_wall_w", 4.0, -21.0, -16.2),
        _hwall("villa_craft_bottom_wall_e", 4.0, -12.4, -3.6),
        _hwall("villa_craft_top_wall_w", 13.6, -21.0, -17.2),
        _hwall("villa_craft_top_wall_e", 13.6, -13.8, -9.2),
        _vwall("villa_craft_bath_wall_s", -8.4, 4.0, 7.6),
        _vwall("villa_craft_bath_wall_n", -8.4, 10.4, 13.6),
        _hwall("villa_bath_bottom_wall", 4.0, -8.4, -3.6),
        _hwall("villa_bath_top_wall", 12.2, -8.4, -3.6),
        _hwall("villa_cloak_elder_wall_w", -3.2, -21.0, -16.2),
        _hwall("villa_cloak_elder_wall_e", -3.2, -12.6, -3.6),
        _vwall("villa_cloak_right_wall_s", -11.2, -3.2, -0.7),
        _vwall("villa_cloak_right_wall_n", -11.2, 1.8, 4.0),
        _hwall("villa_elder_bed_bottom_w", -15.2, -12.2, -9.7),
        _hwall("villa_elder_bed_bottom_e", -15.2, -6.0, -3.6),
        _hwall("villa_elder_bed_top_w", -3.2, -11.0, -8.4),
        _hwall("villa_elder_bed_top_e", -3.2, -5.8, -3.6),
        _vwall("villa_elder_bath_right_wall_s", -11.2, -15.2, -12.3),
        _vwall("villa_elder_bath_right_wall_n", -11.2, -9.4, -3.2),
        _hwall("villa_balcony_top_w", -16.8, -12.2, -9.2),
        _hwall("villa_balcony_top_e", -16.8, -6.4, -3.6),
        _vwall("villa_balcony_left_wall", -12.2, -21.5, -16.8),
        _vwall("villa_balcony_right_wall", -3.6, -21.5, -16.8),
        _hwall("villa_living_top_w", 3.8, 4.8, 7.2),
        _hwall("villa_living_top_e", 3.8, 14.2, 21.0),
        _vwall("villa_living_left_wall_s", 4.8, -17.0, -12.2),
        _vwall("villa_living_left_wall_n", 4.8, -7.4, 3.8),
        _hwall("villa_foyer_top_w", -17.0, 10.2, 13.0),
        _hwall("villa_foyer_top_e", -17.0, 16.4, 21.0),
        _vwall("villa_foyer_left_wall", 10.2, -24.0, -17.0),
        _rect("villa_kitchen_counter", 12.6, 18.2, 6.4, 2.8),
        _rect("villa_kitchen_island", 12.6, 15.0, 3.2, 1.4),
        _rect("villa_washer", -18.0, 20.0, 1.3, 1.3),
        _rect("villa_laundry_sink", -14.6, 20.0, 2.8, 1.0),
        _rect("villa_dining_table", 10.6, 7.2, 4.2, 2.8),
        _rect("villa_stairs_block", 2.8, 4.8, 1.35, 8.8),
        _rect("villa_living_sofa", 13.2, -12.5, 5.6, 1.8),
        _rect("villa_living_table", 13.2, -9.3, 2.9, 1.7),
        _rect("villa_tv_console", 12.4, -3.2, 5.4, 0.9),
        _rect("villa_elder_bed", -7.2, -9.6, 4.8, 5.2),
        _rect("villa_elder_closet", -14.5, -5.8, 1.1, 4.4),
        _rect("villa_craft_table", -15.4, 9.2, 3.8, 2.3),
        _rect("villa_cloak_storage", -17.4, 0.2, 1.2, 5.3),
        _rect("villa_bath_toilet", -5.2, 10.2, 1.3, 1.5),
        _rect("villa_bath_sink", -6.6, 7.0, 1.8, 0.9),
        _rect("villa_elder_shower", -18.0, -10.8, 2.4, 2.4),
        _rect("villa_elder_toilet", -17.6, -13.8, 1.5, 1.9),
        _rect("villa_elder_sink", -14.3, -13.8, 2.0, 1.2),
        _rect("villa_balcony_chair", -9.7, -18.9, 1.4, 1.4),
        _rect("villa_foyer_console", 18.2, -20.6, 2.8, 1.0),
    ]


def _studio_apartment_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _hwall("studio_bed_wall_w", 1.0, -14.0, -4.0),
        _hwall("studio_bed_wall_e", 1.0, -0.5, 14.0),
        _vwall("studio_kitchen_wall", 4.0, -30.0, -12.0),
        _vwall("studio_bath_wall", 6.5, 1.0, 14.0),
        _hwall("studio_bath_front", 14.0, 6.5, 14.0),
        _rect("studio_sofa", -7.0, -15.0, 4.8, 1.8),
        _rect("studio_coffee_table", -3.0, -11.0, 2.4, 1.5),
        _rect("studio_kitchen_counter", 9.5, -19.5, 5.2, 1.2),
        _rect("studio_kitchen_island", 8.0, -13.0, 2.6, 1.8),
        _rect("studio_bed", -9.0, 15.0, 4.8, 5.2),
        _rect("studio_bath_fixture", 10.5, 8.0, 2.4, 3.0),
        _rect("studio_vase", -4.0, -7.5, 1.1, 1.1),
    ]


def _two_bedroom_apartment_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _hwall("apt2_private_wall_w", 2.0, -14.0, -4.2),
        _hwall("apt2_private_wall_e", 2.0, 3.8, 14.0),
        _vwall("apt2_bedroom_split", 0.0, 2.0, 30.0),
        _vwall("apt2_kitchen_wall", 4.8, -30.0, -10.0),
        _hwall("apt2_bath_wall", 9.0, 4.0, 14.0),
        _vwall("apt2_bath_split", 7.2, 2.0, 9.0),
        _rect("apt2_sofa", -7.0, -15.5, 5.0, 2.0),
        _rect("apt2_tv_console", 1.0, -18.0, 3.6, 0.8),
        _rect("apt2_table", -2.0, -8.5, 2.6, 1.6),
        _rect("apt2_counter", 10.0, -19.0, 4.8, 1.2),
        _rect("apt2_island", 8.2, -13.5, 2.8, 1.7),
        _rect("apt2_bed_w", -8.8, 14.5, 4.8, 5.4),
        _rect("apt2_bed_e", 7.8, 14.5, 4.5, 5.2),
        _rect("apt2_bath_fixture", 10.4, 5.8, 2.2, 2.8),
        _rect("apt2_fragile_lego", -5.0, -6.0, 1.5, 1.2),
    ]


def _bungalow_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _hwall("bungalow_sleep_wall_w", 3.0, -14.0, -5.0),
        _hwall("bungalow_sleep_wall_e", 3.0, 5.0, 14.0),
        _vwall("bungalow_west_bed_wall", -5.0, 3.0, 30.0),
        _vwall("bungalow_east_bed_wall", 5.0, 3.0, 30.0),
        _vwall("bungalow_kitchen_wall", 6.8, -30.0, -8.0),
        _hwall("bungalow_bath_wall", 14.5, 5.0, 14.0),
        _rect("bungalow_sofa", -5.0, -13.5, 5.2, 2.0),
        _rect("bungalow_dining_table", 0.0, -4.5, 3.2, 2.0),
        _rect("bungalow_kitchen_counter", 10.2, -18.0, 4.8, 1.2),
        _rect("bungalow_master_bed", -9.5, 15.0, 4.8, 5.5),
        _rect("bungalow_child_bed", 9.0, 15.0, 4.3, 4.8),
        _rect("bungalow_bath_fixture", 9.8, 22.0, 2.4, 3.0),
        _rect("bungalow_vase", -1.8, -10.0, 1.2, 1.2),
    ]


def _courtyard_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("courtyard_open_core", 0.0, 0.0, 5.8, 11.0),
        _hwall("courtyard_south_room_w", -12.0, -14.0, -4.0),
        _hwall("courtyard_south_room_e", -12.0, 4.0, 14.0),
        _hwall("courtyard_north_room_w", 12.0, -14.0, -4.0),
        _hwall("courtyard_north_room_e", 12.0, 4.0, 14.0),
        _vwall("courtyard_west_room_wall", -5.8, -30.0, -12.0),
        _vwall("courtyard_east_room_wall", 5.8, -30.0, -12.0),
        _vwall("courtyard_west_private_wall", -5.8, 12.0, 30.0),
        _vwall("courtyard_east_private_wall", 5.8, 12.0, 30.0),
        _rect("courtyard_sofa", -9.0, -18.0, 4.6, 2.0),
        _rect("courtyard_kitchen_counter", 9.5, -17.5, 4.8, 1.2),
        _rect("courtyard_master_bed", -9.5, 19.0, 4.8, 5.4),
        _rect("courtyard_child_bed", 9.4, 19.0, 4.2, 4.8),
        _rect("courtyard_bath_fixture", 8.8, 6.5, 2.2, 3.0),
        _rect("courtyard_fragile_planter", 0.0, -5.5, 1.3, 1.3),
    ]


def _suburban_villa_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(44, 80),
        _hwall("villa_public_private_wall_w", 3.0, -22.0, -5.5),
        _hwall("villa_public_private_wall_e", 3.0, 5.5, 22.0),
        _vwall("villa_west_wing_wall", -8.0, 3.0, 40.0),
        _vwall("villa_east_wing_wall", 8.0, 3.0, 40.0),
        _vwall("villa_kitchen_wall", 9.0, -40.0, -8.0),
        _hwall("villa_bath_wall_w", 18.0, -22.0, -8.0),
        _hwall("villa_bath_wall_e", 18.0, 8.0, 22.0),
        _rect("villa_living_sofa", -8.0, -17.0, 6.0, 2.4),
        _rect("villa_grand_table", 0.0, -7.0, 4.0, 2.4),
        _rect("villa_kitchen_counter", 15.0, -22.0, 6.0, 1.4),
        _rect("villa_kitchen_island", 13.0, -14.5, 3.2, 2.2),
        _rect("villa_master_bed", -15.0, 19.0, 5.4, 6.0),
        _rect("villa_guest_bed", 15.0, 18.0, 4.8, 5.2),
        _rect("villa_bath_fixture_w", -15.0, 28.5, 2.6, 3.2),
        _rect("villa_bath_fixture_e", 15.0, 28.5, 2.6, 3.2),
        _rect("villa_antique_vase", 1.0, -16.0, 1.3, 1.3),
    ]


def _townhouse_long_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(20, 80),
        _hwall("town_living_to_dining_w", -18.0, -10.0, -2.0),
        _hwall("town_living_to_dining_e", -18.0, 2.0, 10.0),
        _hwall("town_kitchen_wall_w", 0.0, -10.0, -2.0),
        _hwall("town_kitchen_wall_e", 0.0, 3.0, 10.0),
        _hwall("town_private_wall_w", 15.0, -10.0, -2.5),
        _hwall("town_private_wall_e", 15.0, 2.5, 10.0),
        _vwall("town_bed_split", 0.0, 15.0, 40.0),
        _vwall("town_bath_wall", 5.0, 0.0, 15.0),
        _rect("town_sofa", -4.0, -28.0, 4.8, 2.0),
        _rect("town_dining_table", -2.0, -8.5, 3.0, 2.0),
        _rect("town_kitchen_counter", 6.5, 5.0, 4.8, 1.2),
        _rect("town_master_bed", -5.5, 26.0, 4.2, 5.2),
        _rect("town_child_bed", 5.4, 27.0, 3.8, 4.8),
        _rect("town_bath_fixture", 7.2, 9.0, 1.8, 2.6),
        _rect("town_vase", -5.5, -14.0, 1.1, 1.1),
    ]


def _duplex_family_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _vwall("duplex_center_spine_s", 0.0, -30.0, -4.0),
        _vwall("duplex_center_spine_n", 0.0, 4.0, 30.0),
        _hwall("duplex_left_private_wall", 4.0, -14.0, -4.0),
        _hwall("duplex_right_private_wall", 4.0, 4.0, 14.0),
        _hwall("duplex_kitchen_wall", -11.0, -14.0, 14.0),
        _vwall("duplex_bath_left_wall", -7.0, 4.0, 16.0),
        _vwall("duplex_bath_right_wall", 7.0, 4.0, 16.0),
        _rect("duplex_sofa_left", -8.0, -18.0, 4.0, 1.8),
        _rect("duplex_sofa_right", 8.0, -18.0, 4.0, 1.8),
        _rect("duplex_shared_table", 0.0, -6.5, 3.4, 2.0),
        _rect("duplex_kitchen_counter", 0.0, -18.5, 6.0, 1.2),
        _rect("duplex_bed_left", -9.0, 16.5, 4.4, 5.0),
        _rect("duplex_bed_right", 9.0, 16.5, 4.4, 5.0),
        _rect("duplex_bath_left", -10.0, 7.8, 2.0, 2.8),
        _rect("duplex_bath_right", 10.0, 7.8, 2.0, 2.8),
    ]


def _open_plan_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _hwall("open_private_wall_w", 6.0, -14.0, -4.0),
        _hwall("open_private_wall_e", 6.0, 4.0, 14.0),
        _vwall("open_bedroom_split", 0.0, 6.0, 30.0),
        _vwall("open_bath_wall", 7.0, 6.0, 18.0),
        _rect("open_kitchen_counter", 9.5, -18.0, 5.2, 1.2),
        _rect("open_kitchen_island", 6.5, -10.5, 3.6, 2.0),
        _rect("open_sectional_sofa", -6.5, -11.0, 5.5, 2.2),
        _rect("open_coffee_table", -1.5, -8.0, 2.6, 1.6),
        _rect("open_dining_table", 1.0, -1.0, 3.4, 2.0),
        _rect("open_master_bed", -8.8, 17.0, 4.8, 5.4),
        _rect("open_child_bed", 8.5, 17.0, 4.2, 4.8),
        _rect("open_bath_fixture", 10.2, 12.0, 2.2, 2.8),
        _rect("open_lego_project", -3.5, -4.0, 1.8, 1.4),
    ]


def _narrow_lot_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(18, 80),
        _vwall("narrow_corridor_w1", -3.6, -40.0, -18.0),
        _vwall("narrow_corridor_w2", 3.6, -18.0, 2.0),
        _vwall("narrow_corridor_w3", -3.6, 2.0, 22.0),
        _hwall("narrow_kitchen_wall", -9.0, -9.0, 9.0),
        _hwall("narrow_bath_wall", 8.0, -9.0, -3.6),
        _hwall("narrow_bed_wall", 22.0, -9.0, 9.0),
        _vwall("narrow_bed_split", 0.0, 22.0, 40.0),
        _rect("narrow_sofa", -5.6, -26.0, 3.2, 1.8),
        _rect("narrow_tv", 5.6, -28.0, 2.6, 0.8),
        _rect("narrow_kitchen_counter", 5.5, -4.0, 3.2, 1.1),
        _rect("narrow_dining_table", -4.5, -2.0, 2.8, 1.6),
        _rect("narrow_bath_fixture", -6.0, 12.0, 1.8, 2.6),
        _rect("narrow_master_bed", -4.8, 30.0, 3.5, 4.6),
        _rect("narrow_child_bed", 4.8, 30.0, 3.5, 4.4),
        _rect("narrow_vase", 0.0, -14.0, 1.0, 1.0),
    ]


def _luxury_villa_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(48, 84),
        _rect("luxury_central_stair_core", 0.0, 1.0, 5.5, 12.0),
        _hwall("luxury_public_private_w", 10.0, -24.0, -5.0),
        _hwall("luxury_public_private_e", 10.0, 5.0, 24.0),
        _vwall("luxury_west_suite_wall", -9.0, 10.0, 42.0),
        _vwall("luxury_east_suite_wall", 9.0, 10.0, 42.0),
        _vwall("luxury_kitchen_wall", 11.0, -42.0, -10.0),
        _hwall("luxury_bath_wing_wall", 26.0, -24.0, 24.0),
        _rect("luxury_sofa_a", -11.0, -18.0, 6.0, 2.2),
        _rect("luxury_sofa_b", -4.0, -12.5, 4.2, 1.8),
        _rect("luxury_dining_table", 0.0, -25.0, 5.0, 2.6),
        _rect("luxury_kitchen_counter", 17.0, -22.0, 6.0, 1.4),
        _rect("luxury_kitchen_island", 15.0, -13.5, 3.4, 2.2),
        _rect("luxury_master_bed", -16.0, 20.0, 5.8, 6.2),
        _rect("luxury_guest_bed", 16.0, 19.5, 5.0, 5.4),
        _rect("luxury_bath_fixture_w", -16.0, 32.0, 2.8, 3.4),
        _rect("luxury_bath_fixture_e", 16.0, 32.0, 2.8, 3.4),
        _rect("luxury_gallery_vase", -2.0, -7.0, 1.4, 1.4),
    ]


def _pose6(text: str | None) -> tuple[float, float, float, float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    values = [float(item) for item in text.split()]
    values += [0.0] * (6 - len(values))
    return tuple(values[:6])  # type: ignore[return-value]


def _combine_pose(
    base: tuple[float, float, float, float, float, float],
    local: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    bx, by, bz, br, bp, byaw = base
    lx, ly, lz, lr, lp, lyaw = local
    cos_yaw = math.cos(byaw)
    sin_yaw = math.sin(byaw)
    return (
        bx + cos_yaw * lx - sin_yaw * ly,
        by + sin_yaw * lx + cos_yaw * ly,
        bz + lz,
        br + lr,
        bp + lp,
        byaw + lyaw,
    )


@lru_cache(maxsize=8)
def _extract_sdf_collision_rects(
    sdf_path: str,
    scale: float = 1.0,
    structural_only: bool = False,
) -> tuple[RectObstacle, ...]:
    """Project Gazebo collision geometry into conservative 2D obstacles."""

    path = Path(sdf_path)
    if not path.exists():
        return ()

    root = ET.parse(path).getroot()
    obstacles: list[RectObstacle] = []
    for link in root.findall(".//link"):
        link_name = link.attrib.get("name", "link")
        link_pose = _pose6(link.findtext("pose"))
        for collision in link.findall("collision"):
            collision_name = collision.attrib.get("name", "collision")
            if structural_only and not (
                link_name.lower().startswith("wall") or collision_name.lower().startswith("wall")
            ):
                continue

            pose = _combine_pose(link_pose, _pose6(collision.findtext("pose")))
            box_size = collision.findtext("./geometry/box/size")
            cylinder = collision.find("./geometry/cylinder")
            if box_size:
                sx, sy, sz = [float(value) for value in box_size.split()]
                if sz < 0.2:
                    continue
                if pose[2] - sz / 2.0 > 0.8:
                    continue
                abs_cos = abs(math.cos(pose[5]))
                abs_sin = abs(math.sin(pose[5]))
                width = (abs_cos * sx + abs_sin * sy) * scale
                height = (abs_sin * sx + abs_cos * sy) * scale
            elif cylinder is not None:
                radius = float(cylinder.findtext("radius") or 0.0)
                length = float(cylinder.findtext("length") or 0.0)
                if radius <= 0 or length < 0.2:
                    continue
                if pose[2] - length / 2.0 > 0.8:
                    continue
                width = height = 2.0 * radius * scale
            else:
                continue

            if max(width, height) < 0.25:
                continue
            obstacles.append(
                _rect(
                    f"{link_name}_{collision_name}",
                    pose[0] * scale,
                    pose[1] * scale,
                    width,
                    height,
                )
            )
    return tuple(obstacles)


def _turtlebot3_house_obstacles() -> list[RectObstacle]:
    return list(
        _extract_sdf_collision_rects(
            str(TB3_GAZEBO_SHARE / "models/turtlebot3_house/model.sdf"),
            scale=3.0,
            structural_only=True,
        )
    )


def _turtlebot3_world_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(30.0, 30.0),
        *_extract_sdf_collision_rects(
            str(TB3_GAZEBO_SHARE / "models/turtlebot3_world/model.sdf"),
            scale=8.0,
        ),
    ]


def _hospital_ui_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("west_room_wall_north_a", -8.2, 5.8, 0.28, 2.4),
        _rect("west_room_wall_north_b", -8.2, 10.6, 0.28, 2.4),
        _rect("west_room_wall_north_c", -8.2, 15.4, 0.28, 2.4),
        _rect("west_room_wall_north_d", -8.2, 20.2, 0.28, 2.4),
        _rect("west_room_wall_north_e", -8.2, 25.0, 0.28, 2.0),
        _rect("east_room_wall_north_a", 8.2, 5.8, 0.28, 2.4),
        _rect("east_room_wall_north_b", 8.2, 10.6, 0.28, 2.4),
        _rect("east_room_wall_north_c", 8.2, 15.4, 0.28, 2.4),
        _rect("east_room_wall_north_d", 8.2, 20.2, 0.28, 2.4),
        _rect("east_room_wall_north_e", 8.2, 25.0, 0.28, 2.0),
        _rect("west_room_wall_south_a", -8.2, -6.8, 0.28, 1.0),
        _rect("west_room_wall_south_b", -8.2, -10.5, 0.28, 2.2),
        _rect("west_room_wall_south_c", -8.2, -15.5, 0.28, 2.2),
        _rect("west_room_wall_south_d", -8.2, -20.5, 0.28, 2.2),
        _rect("west_room_wall_south_e", -8.2, -25.0, 0.28, 2.0),
        _rect("east_room_wall_south_a", 8.2, -6.8, 0.28, 1.0),
        _rect("east_room_wall_south_b", 8.2, -10.5, 0.28, 2.2),
        _rect("east_room_wall_south_c", 8.2, -15.5, 0.28, 2.2),
        _rect("east_room_wall_south_d", 8.2, -20.5, 0.28, 2.2),
        _rect("east_room_wall_south_e", 8.2, -25.0, 0.28, 2.0),
        _rect("north_cross_wall_west", -4.2, 8.4, 5.2, 0.28),
        _rect("north_cross_wall_east", 4.2, 8.4, 5.2, 0.28),
        _rect("south_cross_wall_west", -4.2, -8.4, 5.2, 0.28),
        _rect("south_cross_wall_east", 4.2, -8.4, 5.2, 0.28),
        _rect("nurse_station_block", 0, 3.2, 3.6, 2.0),
        _rect("central_exam_block", 0, -4.0, 3.7, 3.0),
        _rect("central_storage_block", 0, -15.2, 3.8, 3.6),
        _rect("west_divider_1", -11.1, 20, 5.6, 0.22),
        _rect("west_divider_2", -11.1, 12, 5.6, 0.22),
        _rect("west_divider_3", -11.1, -2, 5.6, 0.22),
        _rect("west_divider_4", -11.1, -12, 5.6, 0.22),
        _rect("west_divider_5", -11.1, -22, 5.6, 0.22),
        _rect("east_divider_1", 11.1, 20, 5.6, 0.22),
        _rect("east_divider_2", 11.1, 12, 5.6, 0.22),
        _rect("east_divider_3", 11.1, -2, 5.6, 0.22),
        _rect("east_divider_4", 11.1, -12, 5.6, 0.22),
        _rect("east_divider_5", 11.1, -22, 5.6, 0.22),
    ]


def _apartment_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("apt_living_bedroom_wall", -2.0, 9.5, 0.28, 20.0),
        _rect("apt_living_kitchen_wall", 4.5, -13.0, 0.28, 8.0),
        _rect("apt_hall_wall_left", -5.8, -2.0, 8.0, 0.28),
        _rect("apt_hall_wall_right", 4.0, -2.0, 8.0, 0.28),
        _rect("apt_bath_wall", 7.0, 7.0, 0.28, 8.0),
        _rect("apt_study_wall", 2.8, 13.0, 0.28, 12.0),
        _rect("apt_bedroom_divider", -8.0, 3.0, 8.0, 0.28),
        _rect("apt_kitchen_counter", 10.0, -18.8, 4.8, 1.2),
        _rect("apt_island", 8.0, -14.2, 2.8, 1.8),
        _rect("apt_sofa", -6.2, -14.2, 5.0, 2.0),
        _rect("apt_tv_console", 1.5, -16.8, 3.5, 0.8),
        _rect("apt_coffee_table", -2.0, -10.0, 2.4, 1.7),
        _rect("apt_bed_main", -9.2, 14.8, 4.8, 5.5),
        _rect("apt_bed_child", -9.4, 5.2, 4.4, 3.2),
        _rect("apt_desk", 4.5, 16.5, 3.2, 1.8),
        _rect("apt_bath_fixture", 11.0, 7.5, 2.0, 3.2),
        _rect("apt_closet", -4.0, 18.5, 2.0, 4.0),
    ]


def _school_corridor_obstacles() -> list[RectObstacle]:
    corridor_walls: list[RectObstacle] = []
    # Long classroom walls with repeated door gaps, like a real school corridor.
    for side, x in (("west", -6.5), ("east", 6.5)):
        for idx, y in enumerate((-25.0, -15.0, -5.0, 5.0, 15.0, 25.0), start=1):
            corridor_walls.append(_rect(f"school_{side}_corridor_wall_{idx}", x, y, 0.28, 6.3))

    room_dividers = [
        _rect(f"school_west_room_divider_{i}", -10.2, y, 7.2, 0.24)
        for i, y in enumerate((-20.0, -10.0, 0.0, 10.0, 20.0), start=1)
    ] + [
        _rect(f"school_east_room_divider_{i}", 10.2, y, 7.2, 0.24)
        for i, y in enumerate((-20.0, -10.0, 0.0, 10.0, 20.0), start=1)
    ]
    light_furniture = [
        _rect("school_library_stack_w", -2.2, 23.0, 1.0, 5.2),
        _rect("school_library_stack_e", 2.2, 23.0, 1.0, 5.2),
        _rect("school_cafe_table_w", -2.1, -24.0, 2.4, 1.2),
        _rect("school_cafe_table_e", 2.1, -22.0, 2.4, 1.2),
        _rect("school_west_teacher_desk_a", -10.2, 14.0, 2.8, 0.9),
        _rect("school_west_teacher_desk_b", -10.2, -6.0, 2.8, 0.9),
        _rect("school_east_teacher_desk_a", 10.2, 14.0, 2.8, 0.9),
        _rect("school_east_teacher_desk_b", 10.2, -6.0, 2.8, 0.9),
    ]
    return [
        *_perimeter(),
        *corridor_walls,
        *room_dividers,
        # Central service core creates a loop corridor instead of one empty strip.
        _rect("school_central_service_core", 0.0, 1.5, 4.8, 16.0),
        _rect("school_service_wall_south_w", -3.7, -8.2, 5.2, 0.28),
        _rect("school_service_wall_south_e", 3.7, -8.2, 5.2, 0.28),
        _rect("school_service_wall_north_w", -3.7, 11.2, 5.2, 0.28),
        _rect("school_service_wall_north_e", 3.7, 11.2, 5.2, 0.28),
        *light_furniture,
    ]


def _office_suite_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("office_north_wall_left", -3.0, 15.0, 13.0, 0.28),
        _rect("office_north_wall_right", 6.0, 15.0, 8.0, 0.28),
        _rect("office_server_wall", 4.8, 21, 0.28, 11.0),
        _rect("office_meeting_wall", -4.0, 21, 0.28, 11.0),
        *[
            _rect(f"office_desk_{x}_{y}", x, y, 2.4, 1.6)
            for x in (-7.5, -3.5, 3.5, 7.5)
            for y in (-14, -8, -2, 4, 10)
        ],
        _rect("office_meeting_table", -8.5, 21.0, 4.8, 2.0),
        _rect("office_server_rack_a", 8.8, 18.5, 1.2, 4.0),
        _rect("office_server_rack_b", 10.6, 22.5, 1.2, 4.0),
        _rect("office_reception_counter", 0.0, 24.0, 5.0, 1.0),
        _rect("office_printer_island", -10.0, -4.0, 1.8, 1.4),
    ]


def _warehouse_aisle_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        *[
            item
            for x in (-9, -5, -1, 3, 7)
            for item in (
                _rect(f"warehouse_shelf_{x}_a", x, -8, 1.2, 24),
                _rect(f"warehouse_shelf_{x}_b", x, 13, 1.2, 13),
            )
        ],
        _rect("warehouse_loading_counter", 0, -26.2, 10.0, 1.0),
        _rect("warehouse_pallet_stack_a", -11.3, 20.0, 2.0, 3.5),
        _rect("warehouse_pallet_stack_b", 10.8, 13.0, 2.0, 3.5),
        _rect("warehouse_forklift_body", 5.5, -24.0, 2.2, 1.6),
        _rect("warehouse_hazmat_cage_wall", 9.5, 21.0, 5.0, 0.28),
    ]


def _museum_gallery_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("museum_partition_left", -4.2, -3, 0.28, 18),
        _rect("museum_partition_right", 5.4, 5, 0.28, 16),
        _rect("museum_vault_wall", 6.2, 20, 0.28, 10),
        _rect("museum_case_1", -2.5, 5, 3.0, 2.0),
        _rect("museum_case_2", 3.5, -10, 3.2, 2.0),
        _rect("museum_case_3", -8.5, 13, 2.2, 2.2),
        _rect("museum_case_4", 0.5, 17, 3.4, 2.0),
        _rect("museum_case_5", -8.5, -12.0, 2.4, 2.4),
        _rect("museum_case_6", 8.0, -2.0, 2.4, 2.4),
        _rect("museum_ticket_counter", 0.0, -25.0, 4.0, 1.0),
    ]


def _maze_clinic_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("maze_wall_1", -6.8, -14.0, 0.28, 18.0),
        _rect("maze_wall_2", -1.5, -21.0, 11.0, 0.28),
        _rect("maze_wall_3", 4.2, -9.8, 0.28, 19.0),
        _rect("maze_wall_4", -4.3, -3.0, 12.0, 0.28),
        _rect("maze_wall_5", -10.2, 4.5, 0.28, 18.0),
        _rect("maze_wall_6", -2.0, 9.5, 13.5, 0.28),
        _rect("maze_wall_7", 5.8, 14.5, 0.28, 15.0),
        _rect("maze_wall_8", 0.8, 21.0, 10.5, 0.28),
        _rect("maze_bed_1", -9.5, 18.0, 3.2, 2.2),
        _rect("maze_bed_2", -9.5, 12.8, 3.2, 2.2),
        _rect("maze_exam_bed_a", 8.8, 15.8, 3.0, 1.6),
        _rect("maze_exam_bed_b", 8.8, 21.0, 3.0, 1.6),
        _rect("maze_counter", 7.8, -16.5, 4.2, 2.0),
        _rect("maze_waiting_chair_a", -8.8, -23.5, 1.1, 1.1),
        _rect("maze_waiting_chair_b", -6.4, -23.5, 1.1, 1.1),
        _rect("maze_pharmacy_shelf_a", 9.8, -13.0, 1.0, 5.0),
        _rect("maze_pharmacy_shelf_b", 6.2, -20.0, 1.0, 5.0),
    ]


def _loop_mall_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("mall_atrium_block", 0, 0, 6.8, 19.0),
        _rect("mall_shop_wall_w1", -5.4, -18, 0.28, 10),
        _rect("mall_shop_wall_w2", -5.4, 18, 0.28, 10),
        _rect("mall_shop_wall_e1", 5.4, -18, 0.28, 10),
        _rect("mall_shop_wall_e2", 5.4, 18, 0.28, 10),
        _rect("mall_bridge_s", 0, -11.8, 8.0, 0.28),
        _rect("mall_bridge_n", 0, 11.8, 8.0, 0.28),
        _rect("mall_kiosk_1", -9.0, -5.0, 2.0, 2.0),
        _rect("mall_kiosk_2", 9.0, 5.0, 2.0, 2.0),
        _rect("mall_food_table_a", 9.0, -20.0, 2.0, 1.4),
        _rect("mall_food_table_b", 9.0, -15.5, 2.0, 1.4),
        _rect("mall_jewelry_counter", -9.0, 18.0, 3.5, 1.2),
        _rect("mall_kids_play_block", 9.0, 18.0, 2.4, 2.4),
    ]


def _conflict_trap_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(44, 88),
        _rect("trap_loop_core", 0.5, -1.0, 6.6, 24.0),
        _rect("trap_wall_w_l1", -7.2, -30.0, 0.3, 13.0),
        _rect("trap_wall_w_l2", -7.2, -9.0, 0.3, 14.0),
        _rect("trap_wall_w_l3", -7.2, 15.0, 0.3, 22.0),
        _rect("trap_wall_e_l1", 7.4, -31.0, 0.3, 12.0),
        _rect("trap_wall_e_l2", 7.4, -10.0, 0.3, 13.0),
        _rect("trap_wall_e_l3", 7.4, 15.0, 0.3, 20.0),
        _rect("trap_cross_s_w", -3.0, -24.0, 8.0, 0.3),
        _rect("trap_cross_s_e", 8.5, -24.0, 6.8, 0.3),
        _rect("trap_cross_m_w", -12.0, -3.0, 7.0, 0.3),
        _rect("trap_cross_m_e", 12.5, 2.5, 7.0, 0.3),
        _rect("trap_cross_n_w", -6.8, 26.0, 9.0, 0.3),
        _rect("trap_cross_n_e", 11.5, 22.0, 8.0, 0.3),
        _rect("trap_store_wall", 9.0, 31.5, 0.3, 12.0),
        _rect("trap_dead_end_a", -17.0, -16.0, 7.0, 0.3),
        _rect("trap_dead_end_b", 17.0, 6.0, 6.8, 0.3),
        _rect("trap_table_1", -2.5, 18.0, 3.2, 2.0),
        _rect("trap_table_2", 2.5, 23.5, 3.0, 2.0),
        _rect("trap_bed_1", -15.5, -4.0, 3.5, 2.2),
        _rect("trap_bed_2", -15.5, 10.5, 3.5, 2.2),
        _rect("trap_bed_3", -15.5, 24.5, 3.5, 2.2),
        _rect("trap_lab_bench", 15.0, -15.0, 5.0, 1.9),
    ]


def _research_lab_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("lab_spine_wall", -1.3, 0, 0.28, 54.0),
        _rect("lab_door_block_1", 1.8, -7.0, 6.0, 0.28),
        _rect("lab_door_block_2", 1.8, 14.0, 6.0, 0.28),
        _rect("lab_chem_wall", 3.2, 7, 0.28, 16.0),
        _rect("lab_storage_wall", 3.2, 22, 0.28, 7.0),
        _rect("lab_bench_1", 7.2, -19.0, 5.5, 1.7),
        _rect("lab_bench_2", 7.2, -13.0, 5.5, 1.7),
        _rect("lab_bench_3", 8.5, 3.0, 5.0, 1.6),
        _rect("lab_bench_4", 8.5, 10.5, 5.0, 1.6),
        _rect("lab_robot_cage", 9.0, -8.0, 4.0, 2.0),
        _rect("lab_storage_shelf_a", 7.0, 22.0, 1.0, 5.0),
        _rect("lab_storage_shelf_b", 10.0, 22.0, 1.0, 5.0),
        _rect("lab_airlock_counter", -4.8, 24.0, 3.5, 1.0),
    ]


def _care_home_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _rect("care_west_spine", -4.7, 7, 0.28, 36),
        _rect("care_east_spine", 4.7, 7, 0.28, 36),
        _rect("care_west_room_divider_a", -8.8, -2.0, 7.0, 0.24),
        _rect("care_west_room_divider_b", -8.8, 8.0, 7.0, 0.24),
        _rect("care_west_room_divider_c", -8.8, 18.0, 7.0, 0.24),
        _rect("care_east_room_divider_a", 8.8, -2.0, 7.0, 0.24),
        _rect("care_east_room_divider_b", 8.8, 8.0, 7.0, 0.24),
        _rect("care_east_room_divider_c", 8.8, 18.0, 7.0, 0.24),
        _rect("care_lobby_wall_l", -7.2, -12, 0.28, 7.0),
        _rect("care_lobby_wall_r", 7.2, -12, 0.28, 7.0),
        _rect("care_dining_table", 0, 5.2, 4.2, 2.0),
        _rect("care_sofa_1", -4.8, -20.0, 3.2, 1.8),
        _rect("care_sofa_2", 4.8, -20.0, 3.2, 1.8),
        _rect("care_med_wall", 0, 18.5, 7.8, 0.28),
        _rect("care_bed_w1", -8.8, 3, 3.0, 2.0),
        _rect("care_bed_w2", -8.8, 14, 3.0, 2.0),
        _rect("care_bed_w3", -8.8, 24, 3.0, 2.0),
        _rect("care_bed_e1", 8.8, 3, 3.0, 2.0),
        _rect("care_bed_e2", 8.8, 14, 3.0, 2.0),
        _rect("care_bed_e3", 8.8, 24, 3.0, 2.0),
        _rect("care_nurse_cart", 0.0, -6.0, 1.8, 1.2),
    ]


def _connected_home_core(width: float = 28, height: float = 60) -> list[RectObstacle]:
    """Shared connected room skeleton with wide door gaps for household maps."""
    left = -width / 2 + 3.2
    right = width / 2 - 3.2
    bottom = -height / 2 + 4.0
    top = height / 2 - 4.0
    return [
        *_perimeter(width, height),
        _hwall("home_public_private_wall_w", 4.0, left, -5.2),
        _hwall("home_public_private_wall_e", 4.0, 5.2, right),
        _vwall("home_left_private_wall", -5.4, 7.0, top),
        _vwall("home_right_private_wall", 5.4, 7.0, top),
        _vwall("home_kitchen_side_wall", min(width / 2 - 5.0, 7.4), bottom, -12.0),
        _hwall("home_kitchen_dining_wall_w", -9.5, left, -4.0),
        _hwall("home_kitchen_dining_wall_e", -9.5, 4.0, right),
        _hwall("home_bath_wall_w", 17.0, left, -5.4),
        _hwall("home_bath_wall_e", 17.0, 5.4, right),
    ]


def _household_furniture(prefix: str, wide: bool = False) -> list[RectObstacle]:
    scale = 1.2 if wide else 1.0
    return [
        _rect(f"{prefix}_sofa", -7.0 * scale, -17.5, 4.4 * scale, 1.6),
        _rect(f"{prefix}_coffee_table", -2.0 * scale, -13.0, 2.2, 1.3),
        _rect(f"{prefix}_dining_table", 0.0, -3.0, 2.8, 1.6),
        _rect(f"{prefix}_kitchen_counter", 9.8 * scale, -21.0, 4.0, 1.0),
        _rect(f"{prefix}_kitchen_island", 8.0 * scale, -15.0, 2.3, 1.4),
        _rect(f"{prefix}_bed_left", -9.5 * scale, 12.0, 3.8, 4.2),
        _rect(f"{prefix}_bed_right", 9.0 * scale, 12.0, 3.6, 4.0),
        _rect(f"{prefix}_bath_fixture", 0.0, 22.5, 2.1, 2.6),
        _rect(f"{prefix}_fragile_object", -2.5 * scale, -6.5, 1.0, 1.0),
    ]


def _studio_apartment_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(),
        *_household_furniture("studio"),
        _hwall("studio_bed_nook_screen", 12.0, -14.0, -9.0),
    ]


def _two_bedroom_apartment_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(),
        *_household_furniture("apt2"),
        _vwall("apt2_bedroom_split_upper", 0.0, 17.0, 30.0),
    ]


def _bungalow_house_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(),
        *_household_furniture("bungalow"),
        _hwall("bungalow_entry_partition_w", -22.0, -14.0, -6.0),
        _hwall("bungalow_entry_partition_e", -22.0, 6.0, 14.0),
    ]


def _courtyard_house_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(),
        _rect("courtyard_low_planter_core", 0.0, -1.5, 4.8, 8.0),
        *_household_furniture("courtyard"),
        _hwall("courtyard_west_room_screen", -16.0, -14.0, -8.0),
        _hwall("courtyard_east_room_screen", -16.0, 8.0, 14.0),
    ]


def _suburban_villa_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(44, 80),
        *_household_furniture("villa", wide=True),
        _rect("villa_grand_stair_low_core", 0.0, 2.0, 4.5, 8.0),
        _hwall("villa_foyer_partition_w", -28.0, -22.0, -6.0),
        _hwall("villa_foyer_partition_e", -28.0, 6.0, 22.0),
    ]


def _townhouse_long_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(20, 80),
        _hwall("town_living_kitchen_wall_w", -16.0, -10.0, -2.5),
        _hwall("town_living_kitchen_wall_e", -16.0, 2.5, 10.0),
        _hwall("town_kitchen_private_wall_w", 8.0, -10.0, -2.5),
        _hwall("town_kitchen_private_wall_e", 8.0, 2.5, 10.0),
        _vwall("town_bedroom_split", 0.0, 20.0, 40.0),
        _rect("town_sofa", -4.8, -28.0, 3.8, 1.6),
        _rect("town_table", 2.5, -24.0, 2.2, 1.3),
        _rect("town_counter", 6.0, -4.0, 3.2, 1.0),
        _rect("town_island", 0.0, -2.0, 2.2, 1.4),
        _rect("town_bed_left", -5.2, 28.0, 3.2, 4.0),
        _rect("town_bed_right", 5.2, 28.0, 3.2, 4.0),
        _rect("town_bath_fixture", 6.2, 14.0, 1.8, 2.4),
    ]


def _duplex_family_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(),
        *_household_furniture("duplex"),
        _vwall("duplex_half_height_spine_s", 0.0, -30.0, -12.0),
        _vwall("duplex_half_height_spine_n", 0.0, 18.0, 30.0),
    ]


def _open_plan_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(),
        _hwall("open_private_wall_w", 8.0, -14.0, -4.5),
        _hwall("open_private_wall_e", 8.0, 4.5, 14.0),
        _vwall("open_bedroom_split", 0.0, 8.0, 30.0),
        _vwall("open_kitchen_hint_wall", 7.8, -30.0, -18.0),
        _rect("open_sofa", -7.0, -13.0, 5.0, 1.8),
        _rect("open_coffee_table", -2.5, -9.5, 2.4, 1.4),
        _rect("open_dining_table", 0.5, -2.0, 3.0, 1.7),
        _rect("open_kitchen_counter", 10.0, -22.0, 4.2, 1.0),
        _rect("open_kitchen_island", 7.2, -14.5, 2.6, 1.6),
        _rect("open_bed_left", -9.2, 16.0, 3.8, 4.2),
        _rect("open_bed_right", 8.8, 16.0, 3.6, 4.0),
        _rect("open_bath_fixture", 0.0, 22.5, 2.0, 2.6),
    ]


def _narrow_lot_house_obstacles() -> list[RectObstacle]:
    return [
        *_perimeter(18, 80),
        _hwall("narrow_living_kitchen_wall_w", -16.0, -9.0, -2.4),
        _hwall("narrow_living_kitchen_wall_e", -16.0, 2.4, 9.0),
        _hwall("narrow_kitchen_bath_wall_w", 5.0, -9.0, -2.4),
        _hwall("narrow_kitchen_bath_wall_e", 5.0, 2.4, 9.0),
        _hwall("narrow_private_wall_w", 20.0, -9.0, -2.4),
        _hwall("narrow_private_wall_e", 20.0, 2.4, 9.0),
        _vwall("narrow_bedroom_split", 0.0, 20.0, 40.0),
        _rect("narrow_sofa", -4.8, -27.0, 3.2, 1.5),
        _rect("narrow_table", 3.5, -23.0, 2.2, 1.2),
        _rect("narrow_counter", 5.5, -6.0, 3.0, 1.0),
        _rect("narrow_bath_fixture", -5.8, 11.0, 1.8, 2.4),
        _rect("narrow_bed_left", -4.8, 30.0, 3.0, 3.8),
        _rect("narrow_bed_right", 4.8, 30.0, 3.0, 3.8),
    ]


def _luxury_villa_obstacles() -> list[RectObstacle]:
    return [
        *_connected_home_core(48, 84),
        *_household_furniture("luxury", wide=True),
        _rect("luxury_low_stair_core", 0.0, 2.0, 5.0, 10.0),
        _hwall("luxury_gallery_screen_w", -30.0, -24.0, -8.0),
        _hwall("luxury_gallery_screen_e", -30.0, 8.0, 24.0),
        _vwall("luxury_west_suite_hint", -12.0, 18.0, 42.0),
        _vwall("luxury_east_suite_hint", 12.0, 18.0, 42.0),
    ]


def primitive_hospital_obstacles() -> list[RectObstacle]:
    wall = "wall"
    return [
        RectObstacle("outer_north", (0, 30), (28, 0.35)),
        RectObstacle("outer_south", (0, -30), (28, 0.35)),
        RectObstacle("outer_west", (-14, 0), (0.35, 60)),
        RectObstacle("outer_east", (14, 0), (0.35, 60)),
        RectObstacle(f"{wall}_west_room_north", (-8.2, 15), (0.28, 22)),
        RectObstacle(f"{wall}_east_room_north", (8.2, 15), (0.28, 22)),
        RectObstacle(f"{wall}_west_room_south", (-8.2, -16), (0.28, 20)),
        RectObstacle(f"{wall}_east_room_south", (8.2, -16), (0.28, 20)),
        RectObstacle("north_cross_wall_west", (-4.2, 8.4), (5.2, 0.28)),
        RectObstacle("north_cross_wall_east", (4.2, 8.4), (5.2, 0.28)),
        RectObstacle("south_cross_wall_west", (-4.2, -8.4), (5.2, 0.28)),
        RectObstacle("south_cross_wall_east", (4.2, -8.4), (5.2, 0.28)),
        RectObstacle("nurse_station", (0, 3.2), (4.6, 2.2)),
        RectObstacle("central_exam_block", (0, -4.0), (5.4, 3.4)),
        RectObstacle("central_storage_block", (0, -15.2), (5.6, 4.2)),
        RectObstacle("west_divider_1", (-11.1, 20), (5.6, 0.22)),
        RectObstacle("west_divider_2", (-11.1, 12), (5.6, 0.22)),
        RectObstacle("west_divider_3", (-11.1, -2), (5.6, 0.22)),
        RectObstacle("west_divider_4", (-11.1, -12), (5.6, 0.22)),
        RectObstacle("west_divider_5", (-11.1, -22), (5.6, 0.22)),
        RectObstacle("east_divider_1", (11.1, 20), (5.6, 0.22)),
        RectObstacle("east_divider_2", (11.1, 12), (5.6, 0.22)),
        RectObstacle("east_divider_3", (11.1, -2), (5.6, 0.22)),
        RectObstacle("east_divider_4", (11.1, -12), (5.6, 0.22)),
        RectObstacle("east_divider_5", (11.1, -22), (5.6, 0.22)),
    ]


def _ray_rect_intersection(
    ox: float,
    oy: float,
    dx: float,
    dy: float,
    rect: RectObstacle,
    max_range: float,
) -> float | None:
    cx, cy = rect.center
    sx, sy = rect.size
    min_x = cx - sx / 2
    max_x = cx + sx / 2
    min_y = cy - sy / 2
    max_y = cy + sy / 2

    t_min = 0.0
    t_max = max_range
    for origin, direction, low, high in ((ox, dx, min_x, max_x), (oy, dy, min_y, max_y)):
        if abs(direction) < 1e-9:
            if origin < low or origin > high:
                return None
            continue
        inv = 1.0 / direction
        t1 = (low - origin) * inv
        t2 = (high - origin) * inv
        near = min(t1, t2)
        far = max(t1, t2)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return None

    if t_min < 0:
        return None
    return min(t_min, max_range)


def _box(low: list[float], high: list[float], dtype: Any):
    if spaces is None:
        return {"low": np.array(low, dtype=dtype), "high": np.array(high, dtype=dtype)}
    return spaces.Box(
        low=np.array(low, dtype=dtype),
        high=np.array(high, dtype=dtype),
        dtype=dtype,
    )
