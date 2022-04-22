# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

__all__ = ['densenet100']


def _bn_function_factory(norm, relu, conv, use_bn=True):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        if use_bn:
            normalized_features = norm(concated_features)
            bottleneck_output = conv(relu(normalized_features))
        else:
            bottleneck_output = conv(relu(concated_features))
        return bottleneck_output

    return bn_function


class _DenseLayer(torch.nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, use_bn=True):
        super(_DenseLayer, self).__init__()
        if use_bn:
            self.add_module('norm1', torch.nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', torch.nn.ReLU(inplace=True)),
        self.add_module('conv1', torch.nn.Conv2d(num_input_features, bn_size * growth_rate,
                                                 kernel_size=1, stride=1, bias=not use_bn)),
        if use_bn:
            self.add_module('norm2', torch.nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', torch.nn.ReLU(inplace=True)),
        self.add_module('conv2', torch.nn.Conv2d(bn_size * growth_rate, growth_rate,
                                                 kernel_size=3, stride=1, padding=1, bias=not use_bn)),
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.use_bn = use_bn

    def forward(self, *prev_features):
        if self.use_bn:
            bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1, use_bn=self.use_bn)
        else:
            bn_function = _bn_function_factory(None, self.relu1, self.conv1, use_bn=self.use_bn)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        if self.use_bn:
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        else:
            new_features = self.conv2(self.relu2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(torch.nn.Sequential):
    def __init__(self, num_input_features, num_output_features, use_bn=True):
        super(_Transition, self).__init__()
        if use_bn:
            self.add_module('norm', torch.nn.BatchNorm2d(num_input_features))
        self.use_bn = use_bn
        self.add_module('relu', torch.nn.ReLU(inplace=True))
        self.add_module('conv', torch.nn.Conv2d(num_input_features, num_output_features,
                                                kernel_size=1, stride=1, bias=not use_bn))
        self.add_module('pool', torch.nn.AvgPool2d(kernel_size=2, stride=2))
        self.gradinit_ = False


class _DenseBlock(torch.nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, use_bn=True):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
                use_bn=use_bn
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class GradInitDenseNet(torch.nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False, use_bn=True, use_pt_init=False, init_multip=1., **kwargs):

        super(GradInitDenseNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        no_bn = not use_bn
        self.use_bn = use_bn

        # First convolution
        if small_inputs:
            self.features = torch.nn.Sequential(OrderedDict([
                ('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=not use_bn)),
            ]))
        else:
            self.features = torch.nn.Sequential(OrderedDict([
                ('conv0', torch.nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=not use_bn)),
            ]))
            if not no_bn:
                self.features.add_module('norm0', torch.nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', torch.nn.ReLU(inplace=True))
            self.features.add_module('pool0', torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
                use_bn=use_bn
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression),
                                    use_bn=use_bn)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        if not no_bn:
            self.features.add_module('norm_final', torch.nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = torch.nn.Linear(num_features, num_classes)

        # Initialization
        if not use_pt_init:
            for name, param in self.named_parameters():
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(math.sqrt(2. / n))
                elif 'conv' in name and 'bias' in name:
                    param.data.zero_()
                elif 'norm' in name and 'weight' in name:
                    param.data.fill_(1)
                elif 'norm' in name and 'bias' in name:
                    param.data.fill_(0)
                elif 'classifier' in name and 'bias' in name:
                    param.data.fill_(0)

        self.gradinit_ = False

        if init_multip != 1:
            for param in self.parameters():
                param.data *= init_multip

    def get_plotting_names(self):
        bn_names, conv_names = [], []
        for n, p in self.named_parameters():
            if (('conv' in n and 'layer' in n) or 'classifier' in n)and 'weight' in n:
                conv_names.append('module.' + n)
            elif 'norm' in n and 'weight' in n and 'layer' in n:
                bn_names.append('module.' + n)

        # bn_names = sorted(bn_names)
        # conv_names = sorted(conv_names)
        if self.use_bn:
            return {'Linear': conv_names, 'BN': bn_names,}
        else:
            return {'Linear': conv_names, }

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet100(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = GradInitDenseNet(growth_rate=12, block_config=(16, 16, 16), **kwargs)
    return model
