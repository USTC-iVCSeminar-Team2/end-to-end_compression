import torch
from models.image_compressor import ImageCompressor
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms

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

device = torch.device('cuda:0')
compressor = ImageCompressor(a, 0)
state_dict_com = load_checkpoint(r"E:\Git_repos\end-to-end_compression\checkpoint\image_compressor\image_compressor_00195000", device)
compressor.load_state_dict(state_dict_com['compressor'])

image = Image.open(r"E:\Git_repos\WallPaper\back_2.png").convert('RGB')
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
rec_img = compressor.inference(img)
PIL_transform = transforms.ToPILImage()
rec_img = PIL_transform(rec_img)
rec_img.save(r"C:\Users\EsakaK\Desktop\res.png")
