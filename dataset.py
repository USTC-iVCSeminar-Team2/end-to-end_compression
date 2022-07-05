import torch
import random
import os


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        h,
        shuffle=True
    ):
        self.training_files = os.listdir(data_dir)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.training_files)

    def __getitem__(self, index):
        filename = self.training_files[index]
        data_1 = 0
        data_2 = 0
        
        return (filename, data_1, data_2)

    def __len__(self):
        return len(self.training_files)