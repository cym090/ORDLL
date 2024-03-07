import torch
import torch.nn as nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*4*4, 2)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

    def forward(self, x, rf=False):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*4*4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        
        if rf:
            return x, y
        return y

class classifier32(nn.Module):
    def __init__(self, num_classes=10):
        super(self.__class__, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3,       64,     3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(64,      64,     3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(64,     128,     3, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv5 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv6 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.conv7 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv8 = nn.Conv2d(128,    128,     3, 1, 1, bias=False)
        self.conv9 = nn.Conv2d(128,    128,     3, 2, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)

        self.bn7 = nn.BatchNorm2d(128)
        self.bn8 = nn.BatchNorm2d(128)
        self.bn9 = nn.BatchNorm2d(128)
        self.bn10 = nn.BatchNorm2d(128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes, bias=False)
        self.dr1 = nn.Dropout2d(0.2)
        self.dr2 = nn.Dropout2d(0.2)
        self.dr3 = nn.Dropout2d(0.2)

        self.apply(weights_init)
        self.cuda()

    def forward(self, x):
        x = self.dr1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.dr3(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = nn.LeakyReLU(0.2)(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # y = self.fc(x)
        # if return_feature:
        #     return x, y
        # else:
        return x

def weights_init(m):
    classname = m.__class__.__name__
    # TODO: what about fully-connected layers?
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)