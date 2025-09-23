import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register as gym_register
from cleanrl_utils.pointmass_continuous_env import PointMassContinuousEnv


class PointMassDiscreteEnv(PointMassContinuousEnv):
    """A continuous point-mass agent in a 2D plane with obstacles.

    - Observation: continuous vector [agent_x, agent_y, goal_x, goal_y, obstacle_info...]
    - Action space: Box(2,) with continuous velocity commands in x and y directions [-max_velocity, max_velocity]
    - Reward: step penalty, goal reward, collision penalty, out-of-bounds penalty
    - Episode ends when agent reaches goal or on time limit
    - Agent has a circular collision radius and cannot pass through rectangular obstacles
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__()

        # TODO: Decide how many discrete actions you need.
        # Action space: some number of discrete actions
        self.action_space = spaces.Discrete(4)

        # Initialize with random positions
        self._reset_positions()

    def step(self, action):
        assert self.action_space.contains(action)

        # TODO: TURN DISCRETE ACTIONS INTO A CONTINUOUS ONE.
        control = np.ones(2)  # placeholder logic

        _, _, terminated, truncated, info = super().step(control)
        observation = self._get_obs()

        # TODO: DEFINE THE REWARD SIGNAL
        reward = 0.0

        return observation, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        obs = super()._get_obs()

        # TODO: CHANGE THE OBSERVATION IF YOU LIKE.

        return obs

    def close(self):
        pass


def make_env(
    world_width: float = 10.0,
    world_height: float = 10.0,
    agent_radius: float = 0.2,
    goal_radius: float = 0.3,
    max_velocity: float = 2.0,
    max_episode_steps: int = 500,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> PointMassDiscreteEnv:
    return PointMassDiscreteEnv(seed=seed, render_mode=render_mode)


# --- Gymnasium registration helpers ---
DEFAULT_ENV_ID = "PointMassDiscrete-v0"


def register_pointmass_discrete_env(env_id: str = DEFAULT_ENV_ID, **kwargs) -> None:
    """Register the PointMassContinuousEnv with Gymnasium."""
    gym_register(
        id=env_id,
        entry_point="cleanrl_utils.pointmass_discrete_env:PointMassDiscreteEnv",
        kwargs=kwargs,
        max_episode_steps=None,
    )
