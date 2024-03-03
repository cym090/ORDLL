import torch

from trainers.base import BaseTrainer


class OrdlTrainer(BaseTrainer):
    def forward_one_batch(self, images):
        pass
    def train_one_batch(self, *args, **kwargs):
        device = self.device
        pass