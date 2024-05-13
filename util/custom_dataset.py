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
        # # Loading Numpy image
        # img_path = os.path.join(self.data_dir, self.file_list[idx])
        # image = np.load(img_path)

        # # Converting to uint8 datatype
        # image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]
        # image = (image * 255).astype(np.uint8)  # Convert to [0, 255] uint8

        # # Converting to PIL object
        # image = Image.fromarray(image).resize((448, 96), resample = Image.NEAREST)
        # if image.mode != 'RGB':
        #     image = Image.merge("RGB", (image, image, image))  # Duplicate channels
        # if self.transform:
        #     image = self.transform(image)
        # return image, 0
        img_path = os.path.join(self.data_dir, self.file_list[idx])
        image = np.load(img_path)
        if image.shape[0] == 99 and image.shape[1] == 450:
            image = image[3::, 2::]
        if len(image.shape) == 2:
            image = np.stack((image, image, image), axis=-1)
        if self.transform:
            image = self.transform(image)
        return image, 0