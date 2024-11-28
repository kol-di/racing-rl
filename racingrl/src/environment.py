import gymnasium as gym

from ptan.common.wrappers import ImageToPyTorch
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from .utils.wrappers import (
    FlattenStackDimension, 
    FloatPixels, 
    SaveObservationImg
)


def get_env(render=False, num_stack=4, resize=False, save_obs=False, save_path=None):
    if save_obs:
        assert save_path is not None, "save_path not provided"

    if render:
        env = gym.make("CarRacing-v2", render_mode='human')
    else:
        env = gym.make("CarRacing-v2")

    if resize:
        env = ResizeObservation(env , shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    env = ImageToPyTorch(env)
    env = MaxAndSkipEnv(env, skip=4)
    if save_obs:
        env = SaveObservationImg(env, save_path, grayscale=True)
    env = FloatPixels(env)
    env = FrameStack(env, num_stack=num_stack)
    env = FlattenStackDimension(env, num_stack=num_stack)

    return env