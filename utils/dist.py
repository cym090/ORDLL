import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def dist_func(features, center, num_classes=6, num_centers=1, metric='l2'):
    if metric == 'l2':
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
        if center is None:
            c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
            dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
        else:
            c_2 = torch.sum(torch.pow(center, 2), dim=1, keepdim=True)
            dist = f_2 - 2*torch.matmul(features, torch.transpose(center, 1, 0)) + torch.transpose(c_2, 1, 0)
        dist = dist / float(features.shape[1])
    else:
        dist = features.matmul(center.t())
    dist = torch.reshape(dist, [-1, num_classes, num_centers])
    dist = torch.mean(dist, dim=2)
    return dist