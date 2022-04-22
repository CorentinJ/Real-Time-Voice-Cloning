import torch
import torch.nn as nn
import itertools

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44',
           'resnet56', 'resnet110', 'resnet1202']


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, bias=bias)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_bn=True):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.stride = stride
        self.use_bn = use_bn

        self.conv1 = conv3x3(inplanes, planes, stride, bias=not use_bn)
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=not use_bn)
        if self.use_bn:
            self.bn2 = torch.nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, use_bn=True, use_zero_init=False, init_multip=1, **kwargs):
        super(ResNet, self).__init__()

        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16, bias=not use_bn)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = torch.nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(64, num_classes)

        if use_zero_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        else:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, torch.nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Linear):
                    if m.bias is not None:
                        m.bias.data.zero_()

        if init_multip != 1:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.weight.data *= init_multip
                    if m.bias is not None:
                        m.bias.data *= init_multip
                elif isinstance(m, torch.nn.BatchNorm2d):
                    m.weight.data *= init_multip
                    m.bias.data *= init_multip
                elif isinstance(m, torch.nn.Linear):
                    m.weight.data *= init_multip
                    if m.bias is not None:
                        m.bias.data *= init_multip

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.AvgPool2d(1, stride=stride),
                    torch.nn.BatchNorm2d(self.inplanes),
                )
            else:
                downsample = nn.Sequential(nn.AvgPool2d(1, stride=stride))

        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, use_bn=self.use_bn))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, use_bn=self.use_bn))

        # return nn.ModuleList(layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)

        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

    def get_plotting_names(self):
        bn_names, conv_names = [], []
        for n, p in self.named_parameters():
            if (('conv' in n and 'layer' in n) or 'fc' in n)and 'weight' in n:
                conv_names.append('module.' + n)
            elif 'bn' in n and 'weight' in n and 'layer' in n:
                bn_names.append('module.' + n)

        if self.use_bn:
            return {'Linear': conv_names, 'BN': bn_names,}
        else:
            return {'Linear': conv_names, }


def resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    """
    model = ResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32(**kwargs):
    """Constructs a ResNet-32 model.

    """
    model = ResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44(**kwargs):
    """Constructs a ResNet-44 model.

    """
    model = ResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56(**kwargs):
    """Constructs a ResNet-56 model.

    """
    model = ResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110(**kwargs):
    """Constructs a ResNet-110 model.

    """
    model = ResNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202(**kwargs):
    """Constructs a ResNet-1202 model.

    """
    model = ResNet(BasicBlock, [200, 200, 200], **kwargs)
    return model
