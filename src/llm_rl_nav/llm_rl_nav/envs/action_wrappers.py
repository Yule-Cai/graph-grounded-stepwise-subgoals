from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    try:
        import gym
        from gym import spaces
    except ImportError:  # Allows CLI help before RL dependencies are installed.
        gym = None
        spaces = None


DEFAULT_DISCRETE_ACTIONS = np.array(
    [
        [0.00, 0.00],   # stop
        [-0.08, 0.00],  # backward
        [-0.06, 0.75],  # backward left
        [-0.06, -0.75], # backward right
        [0.12, 0.00],   # forward
        [0.18, 0.00],   # fast forward
        [0.10, 0.90],   # forward left
        [0.10, -0.90],  # forward right
        [0.00, 1.20],   # rotate left
        [0.00, -1.20],  # rotate right
    ],
    dtype=np.float32,
)


class DiscreteActionWrapper(gym.ActionWrapper if gym else object):
    """Map DQN-compatible discrete actions to differential-drive commands."""

    def __init__(self, env: gym.Env, actions: np.ndarray | None = None):
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium or gym is required to use DiscreteActionWrapper")
        super().__init__(env)
        self.actions = np.asarray(actions if actions is not None else DEFAULT_DISCRETE_ACTIONS)
        self.action_space = spaces.Discrete(len(self.actions))

    def action(self, action: Any) -> np.ndarray:
        return self.actions[int(action)]
