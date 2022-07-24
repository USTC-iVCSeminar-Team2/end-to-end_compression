import torch
import torch.nn as nn
import math
from .analysis import Analysis_net
from .GDN import GDN


class Synthesis_net(nn.Module):
    def __init__(self, out_channel_N=192):
        super(Synthesis_net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)

        self.deconv2 = nn.ConvTranspose2d(out_channel_N, 256, 5, stride=2, padding=2, output_padding=1)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)

        self.deconv3 = nn.ConvTranspose2d(256, 3, 9, stride=4, padding=4, output_padding=3)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn3 = GDN(256, inverse=True)

    def forward(self, x):
        x = self.deconv1(self.igdn1(x))
        x = self.deconv2(self.igdn2(x))
        x = self.deconv3(self.igdn3(x))
        return x
