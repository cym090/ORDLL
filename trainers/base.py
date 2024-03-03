import logging
from collections import defaultdict

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import engine
from utils import io
# from utils.metrics import calculate_accuracy
# from utils.misc import AverageMeter


class BaseTrainer:
    def __init__(self, config: DictConfig):
        self.config = config

        self.dataset = None
        self.dataloader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        self.current_epoch = 0
        self.inference_datakey = ''

        self.device = torch.device(config['device'])

    def forward_one_batch(self, x, *args, **kwargs):
        pass

