import os
from pathlib import Path
import torch
import numpy as np

from .src.net import A2CNet
from .src.environment import get_env
from .config.config import read_conf


def get_model_weights_path(model_storage_path):
    model_storage_path = Path(model_storage_path)

    entries = os.listdir(model_storage_path)
    subfolders = [entry for entry in entries if os.path.isdir(os.path.join(model_storage_path, entry))]

    # Find the folder with greatest name (most recent folder)
    if subfolders:
        latest_run_folder = max(subfolders)
    else:
        raise FileNotFoundError('No runs in the model storage folder')
    
    latest_run_weights_path = model_storage_path / latest_run_folder / 'weights'
    weights_entries = os.listdir(latest_run_weights_path)
    if weights_entries:
        latest_weights = max(weights_entries)
    else:
        raise FileNotFoundError('No weights saved for the latest run')
    
    return latest_run_weights_path / latest_weights


def inference(model_weights_path=None):
    config = read_conf()

    if model_weights_path is None:
        # get the latest weights of the latest run
        model_weights_path = get_model_weights_path(config['model_storage_path'])

    env = get_env(
        render=True, 
        num_stack=config['num_stack'], 
        resize=False, 
        save_obs=False
    )

    net = A2CNet(env.observation_space.shape, env.action_space.shape[0])
    net.load_state_dict(torch.load(model_weights_path))

    obs, _ = env.reset()
    while True:
        obs_t = torch.tensor(np.expand_dims(obs, axis=0), dtype=torch.float32)
        mu_t, _, _ = net(obs_t)
        action = mu_t.squeeze(dim=0).detach().numpy()
        obs = env.step(action)[0]