from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from ValueError import ValueError  # For proper error raising


class CLDataset(Dataset):
    """
    A Contrastive Learning Dataset class that supports both original and augmented data.
    """

    def __init__(
        self,
        x_data: NDArray[np.float32],
        y_data: NDArray[np.int64],
        x_augment: Optional[NDArray[np.float32]] = None,
        y_augment: Optional[NDArray[np.int64]] = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.x_augment = x_augment
        self.y_augment = y_augment if y_augment is not None else y_data

        # Validate shapes if augmented data is provided
        if self.x_augment is not None:
            if len(self.x_data) != len(self.x_augment):
                raise ValueError("Original and augmented data must have the same length")
            if len(self.y_data) != len(self.y_augment):
                raise ValueError("Original and augmented labels must have the same length")
            if not np.array_equal(self.y_data, self.y_augment):
                raise ValueError("Original and augmented labels must have identical values")

    def __len__(self) -> int:
        return len(self.x_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.x_data[idx]

        # Handle different input formats
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Use pre-augmented data if available
        if self.x_augment is not None:
            aug_image = self.x_augment[idx]
            if aug_image.max() <= 1.0:
                aug_image = (aug_image * 255).astype(np.uint8)
            if len(aug_image.shape) == 2:
                aug_image = np.expand_dims(aug_image, axis=-1)

            x1 = self.transform_augment(image=image)["image"]
            x2 = self.transform_augment(image=aug_image)["image"]
        else:
            x1 = self.transform_augment(image=image)["image"]
            x2 = self.transform_augment(image=image)["image"]

        # Convert to tensors and normalize
        x1 = torch.tensor(x1).permute(2, 0, 1).float() / 255.0
        x2 = torch.tensor(x2).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(self.y_data[idx]).long()

        return x1, x2, label
