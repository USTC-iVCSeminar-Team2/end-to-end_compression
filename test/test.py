import os
import torch
import json
import argparse
import numpy as np
from env import AttrDict
from dataset import Dataset
from scipy.misc import imread
import matplotlib.pyplot as plt
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from module_list import compressor_list
from peak_signal_noise_ratio import peak_signal_noise_ratio



def test(a,h,rank):
    with torch.no_grad():
        # load test dataset
        test_dataset = Dataset(a.testing_dir, h, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, num_workers=2, shuffle=False, batch_size=1, pin_memory=True)

        # import model
        device = torch.device('cuda:{:d}'.format(rank))
        compressor = compressor_list(a,h,rank).to(device)
        compressor.load_state_dict(torch.load(r"/Users/josie/Desktop/image_compressor_00320000"))

        # test model
        cnt = 0
        sumBpp = 0
        sumPSNR = 0
        sumMsssim = 0
        sumMsssimDB = 0
        for batch_idx, batch in enumerate(test_loader):
            img = batch
            # calculate average bpp,psnr and ms-ssim of Kodak24 with our model
            rec_img,bpp = compressor.inference(img)
            mse_loss = torch.mean((rec_img-img).pow(2))
            mse_loss,bpp = torch.mean(mse_loss),torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            msssim = ms_ssim(rec_img, img, data_range=1, size_average=True)
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumBpp += bpp
            sumPSNR += psnr
            sumMsssim += msssim
            sumMsssimDB += msssimDB
            cnt += 1

        sumBpp /= cnt
        sumPSNR /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        print("psnr is {:.4f}".format(sumPSNR))
        print("bpp is {:.4f}".format(sumBpp))
        print("MS-SSIM is {:.4f}".format(sumMsssim))
        print("MS-SSIM(db) is {:.4f}".format(sumMsssimDB))

    # calculate average bpp,psnr and ms-ssim of JPEG with our model
    jpeg_bpp = np.array([
        os.path.getsize('test/jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)) * 8 /
        (imread('test/jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)).size // 3)
        for i in range(1, 25) for q in range(1, 21)
    ]).reshape(24, 20)
    jpeg_bpp = np.mean(jpeg_bpp, axis=0)
    jpeg_psnr = np.array([peak_signal_noise_ratio(np.asarray(img.format(i,q)) , np.asarray(imread('test/jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q))))
        for i in range(1, 25) for q in range(1, 21)
    ]).reshape(24, 20)

    plt.plot(sumBpp, sumPSNR, label='End-to-End', marker='o')
    plt.plot(jpeg_bpp, jpeg_psnr, label='JPEG', marker='x')
    plt.xlim(0., 2.)
    plt.ylim(0., 70.)
    plt.xlabel('bit per pixel')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()






def main():

    parser = argparse.ArgumentParser(description='test')

    '''
        '--model_name': Name of the model
        '--testing_dir': Training data dir
        '--config_file': Path of your config file
        '--Lambda': The lambda setting for RD loss

    '''

    parser.add_argument('--model_name', default='image_compressor', type=str)
    parser.add_argument('--testing_dir', default='/Users/josie/Desktop/Kodak24', type=str)
    parser.add_argument('--config_file', default='./configs/config.json', type=str)
    parser.add_argument('--Lambda', default=0.0067, type=float)
    parser.add_argument('--training_dir', default=r'E:\dataset\vimoe\train', type=str)
    parser.add_argument('--validation_dir', default=r'E:\dataset\vimoe\test', type=str)
    parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
    parser.add_argument('--training_epochs', default=3000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    a = parser.parse_args()

    with open(a.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    test(a,h,0)


if __name__ == "__main__":
    main()
