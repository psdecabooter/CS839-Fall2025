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
        self._step_segments = 5
        self.action_space = spaces.Discrete(self._step_segments * 4 + 1)
        self._previous_distance = None
        obs_len = 4 + len(self._get_obstacles()) * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_len + 5,), dtype=np.float32
        )
        # print(self.observation_space.shape)
        # exit()
        self._last_action = np.zeros(2, dtype=np.float32)

        # Initialize with random positions
        self._reset_positions()

    def step(self, action):
        assert self.action_space.contains(action)

        # TODO: TURN DISCRETE ACTIONS INTO A CONTINUOUS ONE.
        control = np.zeros(2, dtype=np.float32)  # placeholder logic
        # magnitude = 2 ** (action % self._step_segments + 1) / 2**self._step_segments
        magnitude = (action % self._step_segments + 1) / self._step_segments
        if action // self._step_segments == 0:
            control[0] += magnitude
        elif action // self._step_segments == 1:
            control[0] -= magnitude
        elif action // self._step_segments == 2:
            control[1] += magnitude
        elif action // self._step_segments == 3:
            control[1] -= magnitude
        self._last_action = control

        _, _, terminated, truncated, info = super().step(control)
        observation = self._get_obs()
        # print(observation.shape)

        # TODO: DEFINE THE REWARD SIGNAL
        distance = np.linalg.norm(self._agent_pos - self._goal_pos)
        if self._previous_distance is None:
            self._previous_distance = distance
        reward = (
            0.5
            if distance < self.goal_radius
            else (self._previous_distance - distance) * 5.0
        )
        # Collision
        if self._check_out_of_bounds(observation[:2], self.agent_radius):
            reward -= 0.5
        elif self._check_collision(observation[:2], self.agent_radius):
            reward -= 0.5
        # Distance
        # if self._previous_distance is None:
        #     self._previous_distance = distance
        # reward += (self._previous_distance - distance) * 5.0
        self._previous_distance = distance

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._previous_distance = None
        self._last_action = np.zeros(2, dtype=np.float32)
        return super().reset(*args, **kwargs)

    def _get_obs(self) -> np.ndarray:
        """Get normalized observation vector."""
        obs = super()._get_obs()
        # TODO: CHANGE THE OBSERVATION IF YOU LIKE.
        extra_obs = np.array(
            [
                obs[0] - obs[2],
                obs[1] - obs[3],
                np.linalg.norm(obs[:2] - obs[2:4]),
            ],
        )

        # print(np.concatenate([obs, extra_obs, self._last_action]).shape)
        return np.concatenate([obs, extra_obs, self._last_action])

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
