import torch
import torch.nn as nn
from torchvision import models


class Vgg19(torch.nn.Module):
    def __init__(self, layers, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice = torch.nn.Sequential()
        for x in range(layers):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        f = self.slice(x)
        return f


class ContentLoss(torch.nn.Module):
    def __init__(self, vgg19_model, layer, criterion):
        super(ContentLoss, self).__init__()
        self.feature_extractor = nn.Sequential(*list(vgg19_model.module.features.children())[:layer])
        self.criterion = criterion

    def forward(self, generated, groundtruth):
        generated_vgg = self.feature_extractor(generated)
        groundtruth_vgg = self.feature_extractor(groundtruth)
        groundtruth_vgg_no_grad = groundtruth_vgg.detach()
        return self.criterion(generated_vgg, groundtruth_vgg_no_grad)
