import numpy as np
import torch
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, RandomRotate90
from torch.utils.data import Dataset


class CLDataset(Dataset):
    def __init__(self, x_data, y_data, transform_augment=None):
        self.x_data = x_data
        self.y_data = y_data

        if transform_augment is None:
            self.transform_augment = Compose(
                [
                    RandomRotate90(),
                    HorizontalFlip(),
                    RandomBrightnessContrast(p=0.5),
                ]
            )
        else:
            self.transform_augment = transform_augment

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        image = self.x_data[idx]

        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        x1 = self.transform_augment(image=image)["image"]
        x2 = self.transform_augment(image=image)["image"]

        x1 = torch.tensor(x1).permute(2, 0, 1).float() / 255.0
        x2 = torch.tensor(x2).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(self.y_data[idx]).long()

        return x1, x2, label
