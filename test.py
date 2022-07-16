import os
import torch
import json
import argparse
import numpy as np
from dataset import Dataset
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from env import AttrDict
from module_list import compressor_list


def test(a,h,rank):
    with torch.no_grad():
        # load test dataset
        test_dataset = Dataset(a.testing_dir, h, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, num_workers=2, shuffle=False, batch_size=1, pin_memory=True)

        # import model
        device = torch.device('cuda:{:d}'.format(rank))
        compressor = compressor_list(rank, a, h).to(device)
        compressor.load_state_dict(torch.load(r".\image_compressor_00320000"))

        # test
        cnt = 0
        sumBpp = 0
        sumLoss = 0
        sumDitortion = 0
        sumMsssim = 0
        sumMsssimDB = 0
        for batch_idx, batch in enumerate(test_loader):
            img = batch
            rec_img = compressor.forward(img)
            loss_items = compressor(img)
            loss, bpp, distortion = compressor.loss(img, loss_items, Lambda=a.Lambda)
            msssim = ms_ssim(rec_img, img, data_range=1, size_average=True)
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumBpp += bpp
            sumLoss += loss
            sumDitortion += distortion
            sumMsssim += msssim
            sumMsssimDB += msssimDB
            cnt += 1
        sumBpp /= cnt
        sumLoss /= cnt
        sumDitortion /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt

        print("TestLoss", sumLoss)
        print("TestBpp", sumBpp)
        print("TestDistortion", sumDitortion)
        print("TestMS-SSIM", sumMsssim)
        print("TestMS-SSIM(db)", sumMsssimDB)


def main():
    print('Initializing Test Process...')

    parser = argparse.ArgumentParser(description='test')

    '''
        '--model_name': Name of the model
        '--testing_dir': Training data dir
        '--config_file': Path of your config file
        '--Lambda': The lambda setting for RD loss

    '''

    parser.add_argument('--model_name', default='image_compressor', type=str)
    parser.add_argument('--testing_dir', default='./Kodak24', type=str)
    parser.add_argument('--config_file', default='./configs/config.config', type=str)
    parser.add_argument('--Lambda', default=0.0067, type=float)
    a = parser.parse_args()

    with open(a.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    test(a,h,0)


if __name__ == "__main__":
    main()
