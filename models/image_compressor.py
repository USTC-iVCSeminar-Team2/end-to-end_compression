import torch
import torch.nn as nn
from modules import *


class ImageCompressor(nn.Module):

    def __init__(self,a, rank) -> None:
        super(ImageCompressor, self).__init__()
        self.a = a
        self.device = torch.device('cuda:{:d}'.format(rank))
        self.encoder = Analysis_net(192)
        self.decoder = Synthesis_net(192)
        self.bit_estimator = BitsEstimator(192, K=5)

    def forward(self, inputs):
        """
        :param inputs: mini-batch
        :return: rec_imgs: 重构图像  bits_map: 累计分布函数
        """
        y = self.encoder(inputs)
        y_hat = self.quantize(y, is_train=True)
        bits_map = self.bit_estimator(y_hat)
        rec_imgs = torch.clamp(self.decoder(y_hat),0,255)

        return bits_map, rec_imgs

    def loss(self, inputs, loss_items):
        """
        :param inputs: original images
        :param loss_items: include bits_map and reconstruced images
        :param Lambda: trade-off
        :return:
        """
        bits_map, rec_imgs = loss_items
        # R loss
        total_bits = torch.sum(
            torch.clamp(
                (-torch.log(
                    self.bit_estimator(bits_map + 0.5) - self.bit_estimator(bits_map - 0.5) + 1e-6)) / torch.log(
                    torch.tensor(2.0)),
                0,
                50))
        img_shape = rec_imgs.size()
        bpp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])
        # D loss
        distortion = torch.mean((inputs - rec_imgs) ** 2)
        # total loss
        loss = bpp + self.a.Lambda * (255 **2 ) * distortion
        return loss, bpp, distortion

    def quantize(self, y, is_train=False):
        if is_train:
            uniform_noise = nn.init.uniform_(torch.zeros_like(y), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            y_hat = y + uniform_noise
        else:
            y_hat = torch.round(y)
        return y_hat

    def inference(self, img):
        """
        only use in test and validate
        """
        y = self.encoder(img)
        y_hat = self.quantize(y, is_train=False)
        rec_img = torch.clamp(self.decoder(y_hat),0,255)
        return rec_img

