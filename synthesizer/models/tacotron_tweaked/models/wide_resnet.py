import torch
import torch.nn.functional as F
from torch.autograd import Variable
import itertools

__all__ = ['wrn_28_10']


def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class wide_basic(torch.nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=True):
        super(wide_basic, self).__init__()

        self.use_bn = use_bn
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm2d(in_planes)
            self.bn2 = torch.nn.BatchNorm2d(planes)
        else:
            # use placeholders
            self.bn1 = torch.nn.Sequential()
            self.bn2 = torch.nn.Sequential()

        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class Wide_ResNet(torch.nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate=0., num_classes=10, use_bn=True, **kwargs):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.use_bn = use_bn

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm2d(nStages[3], momentum=0.9)
        else:
            self.bn1 = torch.nn.Sequential()
        self.linear = torch.nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, use_bn=self.use_bn))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def wrn_28_10(**kwargs):
    return Wide_ResNet(28, 10, **kwargs)


if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
