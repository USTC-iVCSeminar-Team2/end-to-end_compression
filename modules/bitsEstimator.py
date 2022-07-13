import torch
from torch import nn
from torch.nn import functional as F


class EstimatorUnit(nn.Module):
    def __init__(self, inputs_size, node_size=3, head=False, tail=False):
        """
        :param num_channel: input's channel number
        :param tail: is the Kth Unit? True/False
        """
        super(EstimatorUnit, self).__init__()
        self.inputs_size = inputs_size
        self.batch_size, self.num_channel, self.height, self.width = inputs_size
        self.num_points = self.batch_size * self.height * self.width
        self.head = head
        self.tail = tail
        if self.head:
            self.h = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, 1), 0, 0.01))
            self.a = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, self.num_points), 0, 0.01))
            self.b = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, self.num_points), 0, 0.01))
        elif self.tail:
            self.h = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, 1, node_size), 0, 0.01))
            self.a = None
            self.b = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, 1, self.num_points), 0, 0.01))
        else:
            self.h = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, node_size), 0, 0.01))
            self.a = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, self.num_points), 0, 0.01))
            self.b = nn.Parameter(nn.init.normal_(torch.zeros(self.num_channel, node_size, self.num_points), 0, 0.01))

    def forward(self, inputs):
        """
        :param inputs: with batch
        :return: fk(x)
        """
        h = F.softplus(self.h)  # reparameter to ensure h > 0
        if self.tail:
            return torch.sigmoid(torch.matmul(h, inputs) + self.b)  # sigmoid to ensure the range of fx in [0,1]
        else:
            a = torch.tanh(self.a)  # reparameter to ensure a > -1
            x = torch.matmul(h, inputs)
            ret = x + a * torch.tanh(x)
            return ret


class BitsEstimator(nn.Module):
    def __init__(self, inputs_size, K=4):
        super(BitsEstimator, self).__init__()
        self.inputs_size = inputs_size
        self.batch_size, self.num_channel, self.height, self.width = inputs_size
        self.num_points = self.batch_size * self.height * self.width
        self.units = nn.ModuleList()
        self.units.append(EstimatorUnit(self.inputs_size, head=True))
        for i in range(1, K - 1):
            self.units.append(EstimatorUnit(self.inputs_size))
        self.units.append(EstimatorUnit(self.inputs_size, tail=True))

    def forward(self, inputs):
        x = inputs.view(self.num_channel, 1, self.num_points)
        for unit in self.units:
            x = unit(x)
        return x.view(self.inputs_size)


if __name__ == '__main__':
    inputs = torch.randn((4, 192, 16, 16))
    b = BitsEstimator(inputs.size())
    res = b(inputs)
    print(res.size())
