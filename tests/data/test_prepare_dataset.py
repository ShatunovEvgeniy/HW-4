import albumentations as A
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.data.prepare_dataset import CLDataset


@pytest.fixture
def sample_data():
    """Fixture with sample image data (5 RGB images 32x32) and labels."""
    x_data = np.random.randint(0, 256, size=(5, 32, 32, 3), dtype=np.uint8)  # [0, 255]
    y_data = np.array([0, 1, 2, 0, 1])  # Sample labels
    return x_data, y_data


@pytest.fixture
def augmentations():
    """Simple augmentation pipeline (horizontal flip + ToTensor)."""
    return A.Compose([A.HorizontalFlip(p=1.0), ToTensorV2()])


def test_dataset_init(sample_data, augmentations):
    """Test dataset initialization and basic properties."""
    x_data, y_data = sample_data
    dataset = CLDataset(x_data, y_data, transform_augment=augmentations)

    assert len(dataset) == 5  # Check __len__
    assert dataset.x_data.shape == (5, 32, 32, 3)
    assert dataset.y_data.shape == (5,)


def test_getitem_normalization(sample_data, augmentations):
    """Test if images are correctly normalized to [0, 1]."""
    x_data, y_data = sample_data
    dataset = CLDataset(x_data, y_data, transform_augment=augmentations)
    x1, x2, label = dataset[0]  # Get first sample

    # Check tensor types and ranges
    assert isinstance(x1, torch.Tensor)
    assert isinstance(x2, torch.Tensor)
    assert 0.0 <= x1.min() and x1.max() <= 1.0
    assert 0.0 <= x2.min() and x2.max() <= 1.0
    assert x1.shape == x2.shape == (3, 32, 32)  # CHW format


def test_augmentations(sample_data, augmentations):
    """Test if augmentations are applied correctly."""
    x_data, y_data = sample_data
    dataset = CLDataset(x_data, y_data, transform_augment=augmentations)
    x1, x2, _ = dataset[0]

    # x1 should be augmented (flipped), x2 should be original
    assert not torch.allclose(x1, x2)  # Augmentation changed the image


def test_dataloader_compatibility(sample_data, augmentations):
    """Test compatibility with PyTorch DataLoader."""
    x_data, y_data = sample_data
    dataset = CLDataset(x_data, y_data, transform_augment=augmentations)
    loader = DataLoader(dataset, batch_size=2)

    batch = next(iter(loader))
    assert len(batch) == 3  # x1, x2, labels
    assert batch[0].shape == (2, 3, 32, 32)  # Batched x1
