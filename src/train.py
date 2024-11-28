import gymnasium as gym
# from environment import Environment
# from experience_source import ExpirienceSource
from ptan.experience import ExperienceSourceFirstLast
from ptan.common.wrappers import ImageToPyTorch
# from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv,  WarpFrame
# from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack, SubprocVecEnv 
from gymnasium.wrappers.resize_observation import ResizeObservation
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.frame_stack import FrameStack
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from torch.optim import Adam

from .config.config import read_conf
from .net import A2CNet
from .agent import Agent
from .utils.wrappers import FlattenStackDimension, FloatPixels, SaveObservationImg
from .utils.model_storage import ModelStorage
from .utils.log_model import SummaryWriterSingleton, write_stats


def train():
    config = read_conf()
    batch_size = config['batch_size']
    num_stack = config['num_stack']
    screen_img_output_folder = config['screen_img_output_folder']
    model_storage_path = config['model_storage_path']
    save_model_every = config['save_model_every']
    tensorboard_log_dir = config['tensorboard_log_dir']

    env = gym.make("CarRacing-v2")
    # env = ResizeObservation(env , shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    env = ImageToPyTorch(env)
    env = MaxAndSkipEnv(env, skip=4)
    # env = SaveObservationImg(env, screen_img_output_folder, grayscale=True)
    env = FloatPixels(env)
    env = FrameStack(env, num_stack=num_stack)
    env = FlattenStackDimension(env, num_stack=num_stack)

    net = A2CNet(env.observation_space.shape, env.action_space.shape[0])
    optimizer = Adam(net.parameters())
    agent = Agent(net, optimizer, config)

    exp_source = ExperienceSourceFirstLast(env, agent, gamma=config['gamma'], steps_count=config['reward_steps'])
    # initialise singleton here to set output directory once and for all
    writer = SummaryWriterSingleton(tensorboard_log_dir)

    batch = []
    epoch = 0
    with ModelStorage(agent.net, model_storage_path) as model_storage:
        for exp in exp_source:
            batch.append(exp)
            if len(batch) < batch_size:
                continue

            monitor_stats = agent.update_policy(batch)
            write_stats(writer, monitor_stats, epoch)
            batch.clear()

            print(f'Epoch {epoch}')
            if not epoch % save_model_every:
                model_storage.save_model()
                print(f'Saved model backup after epcoh {epoch}')
            epoch += 1


    env.close()