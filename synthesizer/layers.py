import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class ZoneoutRNN(nn.Module):
    def __init__(self, forward_cell, backward_cell, zoneout_prob, bidrectional=True, dropout_rate=0.5):
        super(ZoneoutRNN, self).__init__()
        self.forward_cell = forward_cell
        self.backward_cell = backward_cell
        self.zoneout_prob = zoneout_prob
        self.bidrectional = bidrectional
        self.dropout_rate = dropout_rate

        if self.bidrectional:
            if self.forward_cell.hidden_size != self.backward_cell.hidden_size:
                raise TypeError(
                    "The forward cell should be the same as backward!")
        if not isinstance(forward_cell, nn.RNNCellBase):
            raise TypeError("The cell is not a LSTMCell or GRUCell!")
        if isinstance(forward_cell, nn.LSTMCell):
            if not isinstance(zoneout_prob, tuple):
                raise TypeError("The LSTM zoneout_prob must be a tuple!")
        elif isinstance(forward_cell, nn.GRUCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The GRU zoneout_prob must be a float number!")
        elif isinstance(forward_cell, nn.RNNCell):
            if not isinstance(zoneout_prob, float):
                raise TypeError("The RNN zoneout_prob must be a float number!")

    @property
    def hidden_size(self):
        return self.forward_cell.hidden_size

    @property
    def input_size(self):
        return self.forward_cell.input_size

    def forward(self, forward_input, backward_input, forward_state, backward_state):
        if self.bidrectional == True:
            forward_new_state = self.forward_cell(forward_input, forward_state)
            backward_new_state = self.backward_cell(
                backward_input, backward_state)
            if isinstance(self.forward_cell, nn.LSTMCell):
                forward_h, forward_c = forward_state
                forward_new_h, forward_new_c = forward_new_state

                backward_h, backward_c = backward_state
                backward_new_h, backward_new_c = backward_new_state
                zoneout_prob_c, zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h
                forward_new_c = (1 - zoneout_prob_c) * F.dropout(forward_new_c, p=self.dropout_rate,
                                                                 training=self.training) + forward_c

                backward_new_h = (1 - zoneout_prob_h) * F.dropout(backward_new_h, p=self.dropout_rate,
                                                                  training=self.training) + backward_h
                backward_new_c = (1 - zoneout_prob_c) * F.dropout(backward_new_c, p=self.dropout_rate,
                                                                  training=self.training) + backward_c

                forward_new_state = (forward_new_h, forward_new_c)
                backward_new_state = (backward_new_h, backward_new_c)
                forward_output = forward_new_h
                backward_output = backward_new_h

            else:
                forward_h = forward_state
                forward_new_h = forward_new_state

                backward_h = backward_state
                backward_new_h = backward_new_state
                zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h
                backward_new_h = (1 - zoneout_prob_h) * F.dropout(backward_new_h, p=self.dropout_rate,
                                                                  training=self.training) + backward_h

                forward_new_state = forward_new_h
                backward_new_state = backward_new_h
                forward_output = forward_new_h
                backward_output = backward_new_h

            return forward_output, backward_output, forward_new_state, backward_new_state
        else:
            forward_new_state = self.forward_cell(forward_input, forward_state)
            if isinstance(self.forward_cell, nn.LSTMCell):
                forward_h, forward_c = forward_state
                forward_new_h, forward_new_c = forward_new_state

                zoneout_prob_c, zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h
                forward_new_c = (1 - zoneout_prob_c) * F.dropout(forward_new_c, p=self.dropout_rate,
                                                                 training=self.training) + forward_c
                forward_new_state = (forward_new_h, forward_new_c)
                forward_output = forward_new_h

            else:
                forward_h = forward_state
                forward_new_h = forward_new_state

                zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * F.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h

                forward_new_state = forward_new_h
                forward_output = forward_new_h
            return forward_output, forward_new_state
