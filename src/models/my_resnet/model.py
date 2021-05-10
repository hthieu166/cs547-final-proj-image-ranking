import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torchsummary import summary
import torchvision.models as models

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict

device = torch.device('cuda')

# 3x3 Convolution with Padding
def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

# 1x1 Convolution
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.stride = stride
        self.expanded_channels = self.out_channels * self.expansion

        self.blocks = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential(
            conv1x1(self.in_channels, self.expanded_channels, stride=self.stride),
            nn.BatchNorm2d(self.expanded_channels)
        )

    def forward(self, x):
        residual = x

        x = self.blocks(x)

        if self.in_channels != self.expanded_channels:
            residual = self.shortcut(residual)

        x += residual
        x = self.relu(x)

        return x

class BasicBlock(ResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, stride=stride)

        self.blocks = nn.Sequential(
            conv3x3(self.in_channels, self.out_channels, stride=self.stride),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            conv3x3(self.out_channels, self.expanded_channels),
            nn.BatchNorm2d(self.expanded_channels)
        )

class Bottleneck(ResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(in_channels, out_channels, expansion=4, stride=stride)

        self.blocks = nn.Sequential(
             conv1x1(self.in_channels, self.out_channels),
             nn.BatchNorm2d(self.out_channels),
             nn.ReLU(inplace=True),

             conv3x3(self.out_channels, self.out_channels, stride=self.stride),
             nn.BatchNorm2d(self.out_channels),
             nn.ReLU(inplace=True),

             conv1x1(self.out_channels, self.expanded_channels),
             nn.BatchNorm2d(self.expanded_channels),
        )

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, depths, block, blocks_sizes=[64, 128, 256, 512]):
        super().__init__()

        self.in_channels = in_channels
        self.block = block

        self.gate = nn.Sequential(
            nn.Conv2d(self.in_channels, blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(blocks_sizes[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))

        self.resnet_layers = []
        self.resnet_layers.append(self._make_layer(blocks_sizes[0], blocks_sizes[0], depths[0], stride=1))

        for i in range(0, len(blocks_sizes) - 1):
            self.resnet_layers.append(self._make_layer(blocks_sizes[i] * self.block.expansion, blocks_sizes[i+1], depths[i + 1], stride=2))

        self.resnet_layers = nn.ModuleList(self.resnet_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(blocks_sizes[-1] * self.block.expansion, n_classes)

    def _make_layer(self, in_channels, out_channels, depth, stride):
        layers = [self.block(in_channels, out_channels, stride=stride)]

        for _ in range(1, depth):
            layers.append(self.block(out_channels * self.block.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.gate(x)

        for i, layer in enumerate(self.resnet_layers):
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

def resnet18(in_channels, n_classes, block=BasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[2, 2, 2, 2], *args, **kwargs)

def resnet34(in_channels, n_classes, block=BasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[3, 4, 6, 3], *args, **kwargs)

def resnet50(in_channels, n_classes, block=Bottleneck, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[3, 4, 6, 3], *args, **kwargs)

def resnet101(in_channels, n_classes, block=Bottleneck, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[3, 4, 23, 3], *args, **kwargs)

def resnet152(in_channels, n_classes, block=Bottleneck, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, depths=[3, 8, 36, 3], *args, **kwargs)

# model = resnet50(3, 1000)
# summary(model.to(device), (3, 224, 224))
# summary(models.resnet50(False).cuda(), (3, 224, 224))

# This code has been adapted (but not directly copied) from various tutorials and open source implementations.
# Source: https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278
# Source: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
