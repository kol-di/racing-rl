
import gymnasium as gym
from gymnasium.core import ObsType, ActType
from typing import Any

import gym.spaces as spaces
from gym.spaces import Box
from gym import ObservationWrapper
import numpy as np
from PIL import Image
from datetime import datetime
from pathlib import Path
from collections import deque


class FlattenStackDimension(ObservationWrapper):
    """
    Observation wrapper that flattens the stack dimension by
    moving it to colour dimension
    """
    def __init__(self, env: gym.Env[ObsType, ActType], num_stack: int, is_torch_image: bool=True):
        super(FlattenStackDimension, self).__init__(env)

        if is_torch_image:
            assert env.observation_space.shape[1] in (1, 3)
            obs_shape = self.observation_space.shape[1:]
            self.observation_space = Box(low=0, high=255, shape=(num_stack, obs_shape[1], obs_shape[2]), dtype=np.uint8)
        else:
            assert env.observation_space.shape[-1] in (1, 3)
            obs_shape = self.observation_space.shape[:2]
            self.observation_space = Box(low=0, high=255, shape=(obs_shape[1], obs_shape[2], num_stack), dtype=np.uint8)

    def observation(self, observation: ObsType) -> ObsType:
        return np.reshape(observation, self.observation_space.shape)
    

class FloatPixels(ObservationWrapper):
    """Observation wrapper that transforms observations to float32"""
    def observation(self, observation: ObsType) -> ObsType:
        return observation.astype(np.float32) / 256
    
class SaveObservationImg(ObservationWrapper):
    """Observation wrapper that saves incoming observations as images"""
    def __init__(
        self, env: gym.Env[ObsType, ActType], output_folder: str, grayscale: bool
    ):
        super(SaveObservationImg, self).__init__(env)
        self.output_fodler = Path(output_folder)
        self.grayscale = grayscale

        self.save_count = 0
        self.save_max = 100
        self.to_save = self.save_count < self.save_max

    def observation(self, observation: ObsType) -> ObsType:
        if self.to_save:
            if self.grayscale:
                img = Image.fromarray(observation[0])
            else:
                img = Image.fromarray(observation)

            self.save_count += 1
            self.to_save = self.save_count < self.save_max
            
            file_name = f'screen_{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}.png'
            img.save(self.output_fodler / file_name)

        return observation
    

# class DelayObservation(ObservationWrapper):

#     def __init__(self, env: gym.Env[ObsType, ActType], delay: int):
#         gym.ObservationWrapper.__init__(self, env)

#         self.delay = int(delay)
#         self.observation_queue = deque()

#     def reset(
#         self, *, seed: int | None = None, options: dict[str, Any] | None = None
#     ) -> tuple[ObsType, dict[str, Any]]:
#         self.observation_queue.clear()
#         return super().reset(seed=seed, options=options)

#     def observation(self, observation: ObsType) -> ObsType:
#         self.observation_queue.append(observation)
#         if len(self.observation_queue) > self.delay:
#             return self.observation_queue.popleft()
#         else:
#             return create_zero_array(self.observation_space)