import torch.nn as nn
import torchvision.models as models
from .helper import init, make_standard_block


class VGG(nn.Module):
    def __init__(self, use_bn=True):  # Original implementation doesn't use BN
        super(VGG, self).__init__()
        if use_bn:
            vgg = models.vgg19(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:23]
        else:
            vgg = models.vgg19_bn(pretrained=True)
            layers_to_use = list(list(vgg.children())[0].children())[:33]
        self.vgg = nn.Sequential(*layers_to_use)
        self.feature_extractor = nn.Sequential(make_standard_block(512, 256, 3),
                                               make_standard_block(256, 128, 3))
        init(self.feature_extractor)

    def forward(self, x):
        x = self.vgg(x)
        x = self.feature_extractor(x)
        return x
