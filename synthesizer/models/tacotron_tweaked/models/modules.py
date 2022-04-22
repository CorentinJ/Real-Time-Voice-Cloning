import torch


class Scale(torch.nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x * self.weight


class Bias(torch.nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.bias

