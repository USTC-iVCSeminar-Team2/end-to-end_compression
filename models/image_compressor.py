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
        """
        :param inputs: mini-batch
        :return: rec_imgs: 重构图像  bits_map: 累计分布函数
        """
        y = self.encoder(inputs)
        y_hat = self.quantize(y, is_train=True)
        bits_map = self.bit_estimator(y_hat)
        rec_imgs = self.decoder(y_hat)

        return bits_map, rec_imgs

    def loss(self, inputs, loss_items, Lambda=0.0067):
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
        bbp = total_bits / (img_shape[0] * img_shape[2] * img_shape[3])
        # D loss
        distortion = torch.mean((inputs - rec_imgs) ** 2)
        # total loss
        loss = total_bits + Lambda * (255 **2 ) * distortion
        return loss, bbp, distortion

    def quantize(self, y, is_train=False):
        if is_train:
            uniform_noise = nn.init.uniform_(torch.zeros_like(y), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.cuda()
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
        rec_img = self.decoder(y_hat)
        return rec_img


if __name__ == '__main__':
    i = ImageCompressor()
    i = i.cuda()
    input_image = torch.randn((8, 3, 256, 256)).cuda()
    rec_img = i.inference(input_image)
    print(input_image)
