from typing import Any, Callable, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class CLDataset(Dataset):
    """Custom PyTorch Dataset for Contrastive Learning (CL).

    Takes input data and applies augmentations to generate two different views of the same image.
    Support images in both [0, 1] and [0, 255] ranges, automatically normalizing them.

    :param x_data: Input images in (N, H, W, C) or (N, C, H, W) format.
    :type x_data: np.ndarray
    :param y_data: Labels corresponding to the input images.
    :type y_data: np.ndarray
    :param transform_augment: Albumentations augmentation pipeline for generating the first view (x1).
    :type transform_augment: Optional[Callable]
    :raises AssertionError: If `transform_augment` is not provided.
    """

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, transform_augment: Optional[Callable] = None) -> None:
        if transform_augment is None:
            raise ValueError("transform_augment must be provided")
        self.x_data = x_data
        self.y_data = y_data
        self.transform_augment = transform_augment

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        :return: Number of samples.
        :rtype: int
        """
        return len(self.x_data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Retrieves an augmented pair of images and the corresponding label.

        :param item: Index of the sample to fetch.
        :type item: int
        :return: A tuple containing:
            - x1 (torch.Tensor): First augmented view (normalized to [0, 1]).
            - x2 (torch.Tensor): Second view (original image, normalized to [0, 1]).
            - label: The label of the sample.
        :rtype: Tuple[torch.Tensor, torch.Tensor, Any]
        """
        image = self.x_data[item]
        label = self.y_data[item]

        # Convert [0, 1] range to [0, 255] if needed
        if isinstance(image, np.ndarray) and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Apply augmentations (should include ToTensorV2)
        augmented = self.transform_augment(image=image)
        x1 = augmented["image"]  # Should already be tensor from ToTensorV2

        # For x2 we need to apply same normalization but without augmentations
        # Create minimal transform with just ToTensorV2
        to_tensor_transform = A.Compose([ToTensorV2()])
        x2 = to_tensor_transform(image=image)["image"]

        # Normalize both tensors to [0, 1]
        x1 = x1.float() / 255.0
        x2 = x2.float() / 255.0

        return x1, x2, label
