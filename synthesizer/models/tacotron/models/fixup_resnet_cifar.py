import torch
# import torch.nn as nn
import numpy as np
from .modules import Scale, Bias
import itertools

__all__ = ['FixUpResNet', 'fixup_resnet20', 'fixup_resnet32', 'fixup_resnet44',
           'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=False)


class NoBNBasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(NoBNBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias1b = Bias()
        self.relu = torch.nn.ReLU(inplace=True)
        # self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes)
        # self.scale = nn.Parameter(torch.ones(1))
        self.scale = Scale()
        # self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # out = self.conv1(x + self.bias1a)
        # out = self.relu(out + self.bias1b)
        out = self.conv1(self.bias1a(x))
        out = self.relu(self.bias1b(out))

        # out = self.conv2(out + self.bias2a)
        # out = out * self.scale + self.bias2b
        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            # identity = self.downsample(x + self.bias1a)
            identity = self.downsample(self.bias1a(x))
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixUpResNet(torch.nn.Module):

    def __init__(self, block, layers, num_classes=10, **kwargs):
        super(FixUpResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        # self.bias1 = torch.nn.Parameter(torch.zeros(1))
        self.bias1 = Bias()
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.bias2 = nn.Parameter(torch.zeros(1))
        self.bias2 = Bias()
        self.fc = torch.nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                # torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # torch.nn.init.normal_(m.weight, mean=0, std=np.sqrt(
                #     2 / (m.weight.shape[0] * np.prod(m.weight.shape[2:]))) * self.num_layers ** (-0.5))
                torch.nn.init.normal_(m.weight, mean=0, std=np.sqrt(
                    2 / (m.weight.shape[0] * np.prod(m.weight.shape[2:]))))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = torch.nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu(x + self.bias1)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.fc(x + self.bias2)
        x = self.fc(self.bias2(x))

        return x


def fixup_resnet20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [3, 3, 3], **kwargs)
    return model


def fixup_resnet32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [5, 5, 5], **kwargs)
    return model


def fixup_resnet44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [7, 7, 7], **kwargs)
    return model


def fixup_resnet56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [9, 9, 9], **kwargs)
    return model


def fixup_resnet110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [18, 18, 18], **kwargs)
    return model


def fixup_resnet1202(**kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    model = FixUpResNet(NoBNBasicBlock, [200, 200, 200], **kwargs)
    return model    