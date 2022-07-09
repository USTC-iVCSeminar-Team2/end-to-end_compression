import os
import torch
import json
import argparse
from dataset import Dataset
from torch.utils.data import DataLoader
from env import AttrDict, build_env
from module_list import compressor_list

def test(rank, a, h):

    with torch.no_grad():

        # load test dataset
        test_dataset = Dataset(a.testing_dir, h, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, num_workers=2, shuffle=False,batch_size=1,pin_memory=True)

        # import model
        device = torch.device('cuda:{:d}'.format(rank))
        compressor = compressor_list(rank, a, h).to(device)
        compressor.load_state_dict(torch.load(r".\image_compressor_00320000"))

        #test
        cnt = 0
        sumBpp = 0
        sumLoss = 0
        sumDitortion = 0
        for batch_idx, batch in enumerate(test_loader):
            img = batch
            loss_items = compressor(img)
            loss, bpp, distortion = compressor.loss(img, loss_items, Lambda=a.Lambda)
            sumBpp += bpp
            sumLoss += loss
            sumDitortion += distortion
            cnt += 1
        sumBpp /= cnt
        sumLoss /= cnt
        sumDitortion /= cnt

        print("TestLoss",sumLoss)
        print("TestBpp",sumBpp)
        print("TestDistortion",sumDitortion)

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
    parser.add_argument('--testing_dir', default=r'.\Kodak24', type=str)
    parser.add_argument('--config_file', default=r'.\configs\config.json',type=str)
    parser.add_argument('--Lambda', default=0.0067, type=float)
    a = parser.parse_args()

    with open(a.config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    test(0, a, h)



if __name__ == "__main__":
    main()