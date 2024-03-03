import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist
import pandas as pd
import numpy as np

class OneRestLoss(nn.Module):
    def __init__(self,**options):
        super().__init__()
        self.Dist = Dist(num_classes=options['num_classes']+1, feat_dim=options['feat_dim']).cuda()
        self.points = self.Dist.centers#归一化中心
        self.a = options['alpha']
        self.b = options['beta']
        self.radius1 = nn.Parameter(torch.tensor([1.0]))
        self.radius2 = nn.Parameter(torch.tensor([0.0]))#径r是待优化参数
        self.k_margin_loss = nn.MarginRankingLoss(margin=1)
        self.u_margin_loss = nn.MarginRankingLoss(margin=1)

    def forward(self,x,y,labels=None):

        loss = 0.0
        batch_size = x.size(0) #batch_size

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

                k_k_dist = dist_l2_p[:, class_label][one_hot.bool()] #当前闭集类到当前对应闭集中心的距离
                u_k_dist = dist_l2_p[:, class_label][(1 - one_hot).bool()]#当前开集类到当前对应闭集中心的距离
                u_u_dist = dist_l2_p[:, -1][(1 - one_hot).bool()]#当前开集类到开集中心的距离
                k_u_dist = dist_l2_p[:, -1][one_hot.bool()]#当前闭集类到开集中心的距离
                k_k_target = torch.ones(k_k_dist.size(0)).cuda()
                u_u_target = torch.ones(u_u_dist.size(0)).cuda()
                
                open_risk = self.u_margin_loss(self.radius1,u_u_dist,u_u_target)
                k_margin_loss = self.k_margin_loss(self.radius2, k_k_dist, k_k_target) + self.k_margin_loss(self.radius2,
                                                                                                            u_k_dist,
                                                                                                            -u_u_target)  

                ge_loss += ((1-self.a)*open_risk + self.a*k_margin_loss)
            loss = (1-self.b)*cla_loss + self.b*ge_loss

        return logits, loss