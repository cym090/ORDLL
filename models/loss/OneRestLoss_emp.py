import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist
import pandas as pd
import numpy as np

class OneRestLoss_emp(nn.Module):
    def __init__(self,**options):
        super().__init__()
        self.Dist = Dist(num_classes=options['num_classes']+1, feat_dim=options['feat_dim']).cuda()
        self.points = self.Dist.centers#归一化中心
        self.a = options['a']
        self.b = options['b']
        self.radius1 = nn.Parameter(torch.tensor([1.0]))
        self.radius2 = nn.Parameter(torch.tensor([0.0]))#径r是待优化参数
        self.k_margin_loss = nn.MarginRankingLoss(margin=1)
        self.u_margin_loss = nn.MarginRankingLoss(margin=1)

    def forward(self,x,y,labels=None):

        loss = 0.0
        batch_size = x.size(0)

        dist_dot_p = self.Dist(x, center=self.points, metric='dot')# (batch_size,num_classes+1)
        dist_l2_p = self.Dist(x, center=self.points)  # (batch_size,num_classes+1)

        logits = dist_dot_p - dist_l2_p
        if labels is not None:
            b_dot = []
            for i in range(logits.size(1) - 1):
                temp = torch.vstack((logits[:, -1], logits[:, i]))
                b_dot.append(temp.t())

            cla_loss = 0.0
            ge_loss = 0.0
            
            one_hot_labels = pd.get_dummies(labels.cpu().numpy())
            num_class = len(one_hot_labels.columns)
            for class_label in one_hot_labels.columns:
                one_hot = (torch.from_numpy(np.array(one_hot_labels[class_label]))).long().cuda()
                cla_loss += F.cross_entropy(b_dot[class_label], one_hot)
            loss = cla_loss 

        return logits, loss