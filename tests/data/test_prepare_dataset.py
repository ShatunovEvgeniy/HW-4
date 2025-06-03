from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from src.data.prepare_dataset import CLDataset  # Import your class


@pytest.fixture
def sample_data():
    """Fixture providing test data samples."""
    x_data = np.random.rand(10, 28, 28).astype(np.float32)  # 10 grayscale 28x28 images
    y_data = np.random.randint(0, 10, size=(10,))  # 10 class labels
    x_aug = np.random.rand(10, 28, 28).astype(np.float32)  # Augmented data
    return x_data, y_data, x_aug


def test_dataset_length(sample_data):
    """Test dataset length property."""
    x_data, y_data, _ = sample_data
    dataset = CLDataset(x_data, y_data)
    assert len(dataset) == 10


def test_with_augmented_data(sample_data):
    """Test behavior when pre-augmented data is provided."""
    x_data, y_data, x_aug = sample_data
    dataset = CLDataset(x_data, y_data, x_aug)

    # Mock transform_augment to return input unchanged
    dataset.transform_augment = MagicMock(side_effect=lambda **kwargs: {"image": kwargs["image"]})

    x1, x2, label = dataset[0]

    assert isinstance(x1, torch.Tensor)
    assert isinstance(x2, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert x1.shape == (1, 28, 28)  # Check tensor shape (channels, height, width)


def test_without_augmented_data(sample_data):
    """Test behavior when no pre-augmented data is provided."""
    x_data, y_data, _ = sample_data
    dataset = CLDataset(x_data, y_data)

    # Mock transform_augment
    dataset.transform_augment = MagicMock(side_effect=lambda **kwargs: {"image": kwargs["image"]})

    x1, x2, label = dataset[0]

    # Verify both views are created from the same image
    assert torch.allclose(x1, x2)


def test_data_normalization(sample_data):
    """Test data normalization to [0, 1] range."""
    x_data, y_data, _ = sample_data
    dataset = CLDataset(x_data, y_data)
    # Return properly shaped data (28, 28, 1)
    dataset.transform_augment = MagicMock(return_value={"image": np.expand_dims(x_data[0], -1)})

    x1, _, _ = dataset[0]
    assert x1.max() <= 1.0 and x1.min() >= 0.0  # Check normalization bounds
    assert x1.shape == (1, 28, 28)  # Additional shape check


# def test_invalid_augmented_data(sample_data):
#     """Test size validation between original and augmented data."""
#     x_data, y_data, x_aug = sample_data
#
#     # Create mismatched data sizes
#     with pytest.raises(AssertionError):
#         CLDataset(x_data, y_data, x_aug[:5])  # Augmented data shorter than original


# def test_label_consistency(sample_data):
#     """Test that dataset raises error when label values don't match."""
#     x_data, y_data, x_aug = sample_data
#     y_aug = y_data.copy()
#     y_aug[0] = 99  # Change one label value
#
#     with pytest.raises(AssertionError) as excinfo:
#         CLDataset(x_data, y_data, x_aug, y_aug)
#
#     # Verify the error message mentions label values
#     assert "identical values" in str(excinfo.value)


def test_channel_handling():
    """Test handling of images with different channel dimensions."""
    # 2D image without channel dimension
    x_data = np.random.rand(10, 28, 28)
    y_data = np.random.randint(0, 10, size=(10,))
    dataset = CLDataset(x_data, y_data)
    dataset.transform_augment = MagicMock(side_effect=lambda **kwargs: {"image": kwargs["image"]})

    x1, _, _ = dataset[0]
    assert x1.shape == (1, 28, 28)  # Should add channel dimension


# def test_label_length_mismatch(sample_data):
#     """Test that dataset raises error when label lengths don't match."""
#     x_data, y_data, x_aug = sample_data
#     y_aug = y_data[:-1]  # Make labels shorter
#
#     with pytest.raises(AssertionError) as excinfo:
#         CLDataset(x_data, y_data, x_aug, y_aug)
#     assert "same length" in str(excinfo.value)


def test_valid_label_case(sample_data):
    """Test that valid label case passes."""
    x_data, y_data, x_aug = sample_data
    # This should not raise any exceptions
    dataset = CLDataset(x_data, y_data, x_aug, y_data.copy())
    assert dataset is not None
