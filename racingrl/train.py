from ptan.experience import ExperienceSourceFirstLast
from torch.optim import Adam

from .config.config import read_conf
from .src.net import A2CNet
from .src.agent import Agent
from .src.utils.model_storage import ModelStorage
from .src.environment import get_env


def train():
    config = read_conf()
    batch_size = config['batch_size']
    num_stack = config['num_stack']
    screen_img_output_folder = config['screen_img_output_folder']
    model_storage_path = config['model_storage_path']
    save_model_every = config['save_model_every']

    env = get_env(render=False, num_stack=num_stack, resize=False, save_obs=False)

    net = A2CNet(env.observation_space.shape, env.action_space.shape[0])
    optimizer = Adam(net.parameters())
    agent = Agent(net, optimizer, config)

    exp_source = ExperienceSourceFirstLast(
        env, 
        agent, 
        gamma=config['gamma'], 
        steps_count=config['reward_steps']
    )

    batch = []
    epoch = 0
    with ModelStorage(agent.net, model_storage_path) as model_storage:
        for exp in exp_source:
            batch.append(exp)
            if len(batch) < batch_size:
                continue

            monitor_stats = agent.update_policy(batch)
            model_storage.write_stats(monitor_stats, epoch)
            batch.clear()

            print(f'Epoch {epoch}')
            if not epoch % save_model_every:
                model_storage.save_model()
            epoch += 1


    env.close()