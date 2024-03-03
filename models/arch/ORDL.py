import torch.nn as nn

from models.arch.base import BaseNet


class ORDL(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 **kwargs):
        super().__init__(backbone, **kwargs)

        self.fc = self.backbone

    def get_training_modules(self):
        return nn.ModuleDict({'fc': self.fc})

    def forward(self, x):
        x = self.backbone(x)
        return x