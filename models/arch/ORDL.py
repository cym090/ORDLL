import torch.nn as nn
import torch
from models.loss.Dist import Dist
from models.arch.base import BaseNet


class ORDL(BaseNet):
    def __init__(self,
                 backbone: nn.Module,
                 init='random',
                 **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = kwargs['feat_dim']
        self.num_classes = kwargs['num_classes']+1
        self.num_centers = kwargs['num_centers']
        self.fc = backbone
        self.Dist = Dist(num_classes=kwargs['num_classes']+1, feat_dim=kwargs['feat_dim'])
        # self.points = self.Dist.centers
        self.radius1 = nn.Parameter(torch.tensor([1.0]))
        self.radius2 = nn.Parameter(torch.tensor([0.0])) # 径r是待优化参数

        if init == 'random':
            self.centers = nn.Parameter(0.1 * torch.randn(self.num_classes * self.num_centers, self.feat_dim))
        else:
            self.centers = nn.Parameter(torch.Tensor(self.num_classes * self.num_centers, self.feat_dim))
            self.centers.data.fill_(0)

    def get_training_modules(self):
        return [{'params': self.fc.parameters()},{'params':self.centers},{'params':self.radius1}, {'params':self.radius2}]
        # return {'fc': self.fc,'centers':self.centers}

    def forward(self, x):
        x = self.fc(x)
        # return x, self.centers, self.radius1, self.radius2
        return x        