import torch
from torch.nn import Module
from pathlib import Path
import os
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import traceback


class ModelStorage:
    def __init__(
            self, 
            model: Module, 
            storage_path: str, 
            load_dir: str=None, 
            save_to_new_dir: bool=True, 
            weights_dir: str='weights'
    ):
        """
        Set model folder path, optionally load old model
        :param model: torch model
        :param storage_path: top-level folder with model subfolders
        :param load_dir: subfolder in storage_path to load model from
        :param save_to_new_dir: (True/False) to create new subfolder if 
            model state is loaded from other subfolder
        """
        if load_dir is None:
            assert save_to_new_dir, \
                "save_to_new_dir can be False only if old dir is provided in load_dir"

        self.model = model
        self.weights_dir = weights_dir

        storage_path = Path(storage_path)
        if load_dir is not None:
            search_dir = storage_path / load_dir
            newest_model = sorted(os.listdir(search_dir))[-1]
            self.model.load_state_dict(search_dir / newest_model)
        if save_to_new_dir:
            model_dir = f'model_{datetime.now().strftime("%Y%m%d%H%M%S")}'
        else:
            model_dir = load_dir
        self.model_path = storage_path / model_dir

        self.summary_writer = SummaryWriter(self.model_path)

    def __enter__(self):
        os.mkdir(self.model_path / self.weights_dir)
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        # save trained model
        self.save_model()

        # remove empty folders if any
        if not any((self.model_path / self.weights_dir).iterdir()):
            (self.model_path / self.weights_dir).rmdir()
        if not any(self.model_path.iterdir()):
            self.model_path.rmdir()

        # close tensorboard summary writer
        self.summary_writer.close()

    def save_model(self):
        model_name = f'state_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}.pt'
        torch.save(self.model.state_dict(), self.model_path / self.weights_dir / model_name)
        print(f'Model weights saved')

    def write_stats(self, stats, step_idx):
        if ((grads := stats.get('policy_loss_grads')) is not None):
            self.summary_writer.add_scalar('grad_l2', np.sqrt(np.mean(np.square(grads))), step_idx)
            self.summary_writer.add_scalar('grad_max', np.max(np.abs(grads)), step_idx)
            self.summary_writer.add_scalar('grad_var', np.var(grads), step_idx)
            
        if ((As := stats.get('As')) is not None):
            self.summary_writer.add_scalar('As', As, step_idx)
        if ((Qs := stats.get('Qs')) is not None):
            self.summary_writer.add_scalar('Qs', Qs, step_idx)
        if ((Vs := stats.get('Vs')) is not None):
            self.summary_writer.add_scalar('Vs', Vs, step_idx)

        if ((loss := stats.get('loss')) is not None):
            self.summary_writer.add_scalar('loss', loss, step_idx)
        if ((policy_loss := stats.get('policy_loss')) is not None):
            self.summary_writer.add_scalar('policy_loss', policy_loss, step_idx)
        if ((value_loss := stats.get('value_loss')) is not None):
            self.summary_writer.add_scalar('value_loss', value_loss, step_idx)
        if ((entropy_bonus := stats.get('entropy_bonus')) is not None):
            self.summary_writer.add_scalar('entropy_bonus', entropy_bonus, step_idx)
