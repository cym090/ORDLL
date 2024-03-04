import torch.nn as nn
from torchvision.models import resnet18

from models.backbone.base import BaseNet


class ResNet18_224(BaseNet):
    def __init__(self, pretrained=True, **kwargs):
        super(ResNet18_224, self).__init__()
        model = resnet18(pretrained=pretrained)

        # self.set_eval_mode = set_eval_mode
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten()
        )
        self.features_size = 2048

    def forward(self, x):
        # if self.set_eval_mode:
        #     self.features.eval()  # to avoid updating BN stats
        x = self.features(x)
        return x
