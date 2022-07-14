import torch
from models.image_compressor import ImageCompressor
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms
import json
from env import AttrDict, build_env
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='image_compressor', type=str)
parser.add_argument('--training_dir', default=r'E:\dataset\vimoe\train', type=str)
parser.add_argument('--validation_dir', default=r'E:\dataset\vimoe\test', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
parser.add_argument('--config_file', default=r'E:\Git_repos\end-to-end_compression\configs\config.json', type=str)
parser.add_argument('--training_epochs', default=3000, type=int)
parser.add_argument('--stdout_interval', default=5, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)
parser.add_argument('--summary_interval', default=100, type=int)
parser.add_argument('--validation_interval', default=1000, type=int)
parser.add_argument('--fine_tuning', default=False, type=bool)
parser.add_argument('--Lambda', default=0.0067, type=float)

a = parser.parse_args()

with open(a.config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
build_env(a.config_file, 'config.json', os.path.join(a.checkpoint_path, a.model_name))

device = torch.device('cuda:0')
compressor = ImageCompressor(a, h,0)
state_dict_com = load_checkpoint(r"./checkpoint/image_compressor/image_compressor_00205000.00205000", device)
compressor.load_state_dict(state_dict_com['compressor'])

image = Image.open(r"C:\Users\EsakaK\Desktop\1.png").convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
img = img.unsqueeze(0).cuda()
compressor = compressor.to(device)

y = compressor.encoder(img)
y_hat = compressor.quantize(y)
print(y_hat.size())