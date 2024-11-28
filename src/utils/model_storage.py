import torch
from torch.nn import Module
from pathlib import Path
import os
from datetime import datetime


class ModelStorage:
    def __init__(
            self, model: Module, storage_path: str, load_dir: str=None, save_to_new_dir: bool=True
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

    def __enter__(self):
        os.mkdir(self.model_path)
        return self
    
    def __exit__(self):
        if not any(self.model_path.iterdir()):
            self.model_path.rmdir()

    def save_model(self):
        model_name = f'state_{datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]}.pt'
        torch.save(self.model.state_dict(), self.model_path / model_name)
