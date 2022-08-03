from models.image_compressor import ImageCompressor
import torch
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms
import json
from env import AttrDict, build_env
import os
from thop.profile import profile
from thop import clever_format

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='image_compressor', type=str)
parser.add_argument('--training_dir', default=r'E:\dataset\vimoe\train', type=str)
parser.add_argument('--validation_dir', default=r'E:\dataset\vimoe\test', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
parser.add_argument('--config_file', default='./configs/config.json', type=str)
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
compressor = ImageCompressor(a, h, 0)

state_dict_com = load_checkpoint(r"E:\Projects\Prj_Seminar\Team2\outputs\image_compressor\models\0_0130_image_compressor_00195000", device)
compressor.load_state_dict(state_dict_com['compressor'])

image = Image.open(r"E:\Datasets\kodac\kodim19.png").convert('RGB')

transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
img = img.unsqueeze(0).cuda()
compressor = compressor.to(device)

# thop for model parmas and macs
total_ops, total_params = profile(compressor, (img,), verbose=False)
macs, params = clever_format([total_ops, total_params], "%.3f")
print('MACs:',macs)
print('Paras:',params)

y = compressor.encoder(img)
y_hat = compressor.quantize(y)
total_bits = compressor.entropy_coder.encode(y_hat, filepath=r"./str.bin")
y_hat_dec = compressor.entropy_coder.decode(filepath=r"./str.bin", device=torch.device('cuda:0'))

print("Code decode consistence: " + str(torch.equal(y_hat, y_hat_dec)))
print("Entropy coding length: {:d} bits".format(total_bits))
print(y_hat.size())


inv_transform = transforms.ToPILImage()

img_reco, bpp = compressor.inference(img)
img_reco = inv_transform(img_reco[0])
print(img_reco.size,image.size)

psnr = peak_signal_noise_ratio(np.asarray(image), np.asarray(img_reco))
print("psnr is {:.4f}".format(psnr))
print("bpp is {:.4f}".format(bpp))