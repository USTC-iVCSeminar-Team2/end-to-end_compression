import torch.nn as nn
import torch
from .GDN import GDN
import math

class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=192):
        super(Analysis_net, self).__init__()
        # Input:3   Output:256
        self.conv1 = nn.Conv2d(3, 256, 9, stride=4, padding=4)
        # Init
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2)))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(256)
        # Input:256     Output:192
        self.conv2 = nn.Conv2d(256, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        # Input:192     Output:192
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)


    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return x


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])
        analysis_net = Analysis_net()
        feature = analysis_net(input_image)
        print("input_image : ", input_image.size())
        print("feature : ", feature.size())


if __name__ == '__main__':
    build_model()
