import logging
from collections import defaultdict

# import hydra
import numpy as np
import torch
# import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# import engine
from utils import io
# from utils.metrics import calculate_accuracy
# from utils.misc import AverageMeter


class BaseTrainer:
    def __init__(self, config: DictConfig):
        self.config = config

        # self.dataset = None
        # self.dataloader = None
        # self.model = None
        # self.optimizer = None
        # self.scheduler = None
        # self.criterion = None

        self.current_epoch = 0
        # self.inference_datakey = ''

        # self.device = torch.device(config['device'])

    def forward_one_batch(self, x, *args, **kwargs):
        pass

    def save_model_state(self, model, fn):
        modelsd = model.state_dict()
        modelsd = {k: v.clone().cpu() for k, v in modelsd.items()}
        io.fast_save(modelsd, fn)
    
    def load_model_state(self, model, fn):
        modelsd = torch.load(fn, map_location='cpu')
        model.load_state_dict(modelsd)

    def save_training_state(self, optimizer, scheduler, fn):
        optimsd = optimizer.state_dict()
        schedulersd = scheduler.state_dict()
        io.fast_save({'optim': optimsd,
                      'scheduler': schedulersd}, fn)

    # def load_training_state(self, fn):
    #     sd = torch.load(fn, map_location='cpu')
    #     self.optimizer.load_state_dict(sd['optim'])
    #     self.scheduler.load_state_dict(sd['scheduler'])