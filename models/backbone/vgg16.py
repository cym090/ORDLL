import torch
import torch.nn as nn
from torchvision.models import vgg16

from models.backbone.base import BaseNet
# from base import BaseNet


class VGG16(BaseNet):
    def __init__(self, weights=None, remove_dropout=True, **kwargs):
        super(VGG16, self).__init__()

        model = vgg16(weights=weights)

        self.features = model.features
        self.avgpool = model.avgpool
        fc = []
        lastidx = len(model.classifier) - 1
        for i in range(lastidx):
            if isinstance(model.classifier[i], nn.Dropout) and remove_dropout:
                continue
            fc.append(model.classifier[i])
        self.fc = nn.Sequential(*fc)
        self.features_size = model.classifier[lastidx].in_features

    def forward(self, x):
        # # (1,3,224,224) -> (1,4096)
        x = self.features(x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
if __name__ == "__main__":
    toy_model = VGG16()
    print(toy_model)