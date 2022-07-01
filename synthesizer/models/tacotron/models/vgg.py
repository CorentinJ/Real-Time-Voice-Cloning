'''VGG11/13/16/19 in Pytorch.'''
import torch
from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(torch.nn.Module):
    def __init__(self, vgg_name, use_bn=True, use_pt_init=False, init_multip=1, **kwargs):
        super(VGG, self).__init__()
        self.use_bn = use_bn
        self.conv_names = []
        self.bn_names = []
        self._make_layers(cfg[vgg_name])
        self.classifier = torch.nn.Linear(512, 10)
        self.conv_names.append(f'module.classifier.weight')
        if not use_pt_init:
            self._initialize_weights()

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
                    m.bias.data *= init_multip

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        # layers = []
        in_channels = 3
        pool_num, block_num = 0, 0
        self.features = torch.nn.Sequential(OrderedDict([]))
        for x in cfg:
            if x == 'M':
                self.features.add_module(f'pool{pool_num}', torch.nn.MaxPool2d(kernel_size=2, stride=2))
                pool_num += 1
            else:
                self.features.add_module(f'conv{block_num}', torch.nn.Conv2d(in_channels, x, kernel_size=3, padding=1))
                if self.use_bn:
                    self.features.add_module(f'bn{block_num}', torch.nn.BatchNorm2d(x))
                self.features.add_module(f'relu{block_num}', torch.nn.ReLU(inplace=True))
                in_channels = x
                self.conv_names.append(f'module.features.conv{block_num}.weight')
                self.bn_names.append(f'module.features.bn{block_num}.weight')
                block_num += 1

        self.add_module('global_pool', torch.nn.AvgPool2d(kernel_size=1, stride=1))

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

    def get_plotting_names(self):
        if self.use_bn:
            return {'Linear': self.conv_names,
                    'BN': self.bn_names,}
        else:
            return {'Linear': self.conv_names,}

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

def vgg11(**kwargs):
    return VGG('VGG11', **kwargs)


def vgg13(**kwargs):
    return VGG('VGG13', **kwargs)


def vgg16(**kwargs):
    return VGG('VGG16', **kwargs)

def vgg19(**kwargs):
    return VGG('VGG19', **kwargs)
