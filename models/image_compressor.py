import torch
import torch.nn as nn


class ImageCompressor(nn.Module):

    def __init__(self, h) -> None:
        super(ImageCompressor, self).__init__()
        self.encoder = None
        self.decoder = None
        self.bit_estimator = None

    def forward(self, img):

        y = self.encoder(img)
        bit = self.bit_estimator(y)
        rec_img = self.decoder(y)

        return bit, rec_img

    def loss(self, loss_items):
        bit, rec_img = loss_items
        bit_rate = 0
        distortion = 0
        loss = bit_rate + 0.5*distortion
        return loss, bit_rate, distortion

    def quantize(self, y):
        y_hat = round(y)
        return y_hat
    
    def inference(self, img):
        y = self.encoder(img)
        y = self.quantize(y)
        rec_img = self.decoder(y)
        return rec_img