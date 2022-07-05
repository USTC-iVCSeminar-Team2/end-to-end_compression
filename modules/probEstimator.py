import torch
from torch import nn
from torch.nn import functional as F


class EstimatorUnit(nn.Module):
    def __init__(self, num_channel, tail=False):
        """
        :param num_channel: input's channel number
        :param tail: is the Kth Unit? True/False
        """
        super(EstimatorUnit, self).__init__()
        self.num_channel = num_channel
        self.tail = tail
        self.h = nn.Parameter(nn.init.normal_(torch.zeros(1, self.num_channel, 1, 1), 0, 0.01))
        self.b = nn.Parameter(nn.init.normal_(torch.zeros(1, self.num_channel, 1, 1), 0, 0.01))
        if not tail:
            self.a = nn.Parameter(nn.init.normal_(torch.zeros(1, self.num_channel, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, inputs):
        """
        :param inputs: with batch
        :return: fk(x)
        """
        h = F.softplus(self.h)  # reparameter to ensure h > 0
        a = torch.tanh(self.a)  # reparameter to ensure a > -1
        if self.tail:
            return torch.sigmoid(h * inputs + self.b)  # sigmoid to ensure the range of fx in [0,1]
        else:
            return inputs + a * torch.tanh(inputs)
