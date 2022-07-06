import torch
import torch.nn as nn
from modules import *


class ImageCompressor(nn.Module):

    def __init__(self) -> None:
        super(ImageCompressor, self).__init__()
        self.encoder = Analysis_net(192)
        self.decoder = Synthesis_net(192)
        self.bit_estimator = BitsEstimator(192, K=5)

    def forward(self, inputs):
        y = self.encoder(inputs)
        bits_map = self.bit_estimator(y)
        rec_imgs = self.decoder(y)

        return bits_map, rec_imgs

    def loss(self, loss_items):
        bits_map, rec_img = loss_items
        bit_rate = 0
        distortion = 0
        loss = bit_rate + 0.5 * distortion
        return loss, bit_rate, distortion

    def quantize(self, y):
        y_hat = torch.round(y)
        return y_hat

    def inference(self, img):
        y = self.encoder(img)
        y = self.quantize(y)
        rec_img = self.decoder(y)
        return rec_img

if __name__ == '__main__':
    i = ImageCompressor()
    i = i.cuda()
    inputs = torch.randn((8,3,256,256)).cuda()
    print(next(i.parameters()).device)
    print(inputs.device)
    bits_map, rec_imgs = i(inputs)
    print(bits_map.size())