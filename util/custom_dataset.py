import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = Image.merge("RGB", (image, image, image))  # Duplicate channels
        if self.transform:
            image = self.transform(image)
        return image, 0