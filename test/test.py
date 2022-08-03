import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import KodacDataset
from module_list import compressor_list
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from pytorch_msssim import ms_ssim
import numpy as np
import argparse, os
import json


torch.backends.cudnn.benchmark = True

def test(rank, a, h):
    # GPU device
    device = torch.device('cuda:{:d}'.format(rank))

    # Kodac test set
    test_set = KodacDataset(data_dir=a.test_dir)
    test_loader = DataLoader(dataset=test_set, batch_size=1)

    # Model
    compressor = compressor_list(a, h, rank).to(device)
    compressor.load_state_dict(torch.load(a.checkpoint_path, map_location=device)['compressor'])
    compressor.to(device)
    print(compressor)

    # Transforms
    trans_to_img = transforms.ToPILImage()

    # Test loop
    print("{:8s} {:8s} {} {:6s} {} {}".format("index", "PSNR", "MS-SSIM", "bpp", "Enc.time", "Dec.time"))
    compressor.eval()
    test_result = {}
    with torch.no_grad():
        for cnt, data in enumerate(test_loader):
            data = data.to(device)
            img = data
            img_reco, bpp, t_enc, t_dec = compressor.inference(img)
            # psnr, ssim
            img_pil = trans_to_img(img[0, :])
            img_reco_pil = trans_to_img(img_reco[0, :])
            img_reco_pil.save(os.path.join(a.reco_dir, "kodim_reco_{:02d}.png".format(cnt + 1)))
            psnr = peak_signal_noise_ratio(np.asarray(img_pil), np.asarray(img_reco_pil))
            ms_ssim_ = ms_ssim(img, img_reco, data_range=1.0, size_average=False).item()
            # mssim = structural_similarity(np.asarray(img_pil.convert('L')), np.asarray(img_reco_pil.convert('L')))
            # test result
            # test_result['kodim{:02d}'.format(cnt + 1)] = {'psnr': psnr, 'mssim': mssim, 'bpp_y': bpp_y, 'bpp_z': bpp_z}
            print('kodim{:02d}, {:.4f}, {:.4f}, {:.6f}, {:.3f}, {:.3f}'.format(cnt + 1, psnr, ms_ssim_, bpp, t_enc, t_dec))

        # print(test_result)


def main():
    print('Initializing Test Process...')

    parser_ = argparse.ArgumentParser(description='test')
    '''
        '--model_name': Name of the model
        '--test_dir': Test data dir
        '--config_file': Path of your config file
        '--lambda_': The lambda setting for RD loss
        '--checkpoint_path: The path of models
    '''
    parser_.add_argument('--model_name', default='image_compressor', type=str)
    parser_.add_argument('--test_dir', default="E:\\Datasets\\kodac", type=str)
    parser_.add_argument('--config_file', default="./configs/config.json", type=str)
    parser_.add_argument('--lambda_', default=0.0067, type=float)
    parser_.add_argument('--checkpoint_path', default="../../outputs/image_compressor/models/0_0130_image_compressor_00195000",
                         type=str)
    parser_.add_argument('--reco_dir', default="../../outputs/img_reco", type=str)
    a = parser_.parse_args()

    test(rank=0, a=a, h='')


if __name__ == '__main__':
    main()
