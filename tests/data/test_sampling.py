import os

import cv2
import numpy as np
import pytest
import torch

from src.data.sampling import extract_sample


@pytest.fixture
def create_test_data(tmp_path):
    img_paths = []
    labels = []
    n_classes = 5
    samples_per_class = 10

    for i in range(n_classes):
        class_dir = tmp_path / f"class_{i}"
        class_dir.mkdir()
        for j in range(samples_per_class):
            img_path = class_dir / f"img_{j}.png"
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:, :] = (i * 10, j * 10, 0)  # different values for different images
            cv2.imwrite(str(img_path), img)
            img_paths.append(str(img_path))
            labels.append(i)

    return np.array(img_paths), np.array(labels)


def test_extract_sample_shapes(create_test_data):
    """We check the correctness of the output data sizes."""
    img_paths, labels = create_test_data
    n_way = 3  # Choose 3 random classes out of 5
    n_support = 2  # 2 examples per class in the support set
    n_query = 3  # 3 examples per class in the query set

    result = extract_sample(n_way, n_support, n_query, img_paths, labels)

    # Checking the structure of the returned dictionary
    assert isinstance(result, dict)
    assert set(result.keys()) == {"images", "n_way", "n_support", "n_query"}

    # We check the dimensions of the image tensor
    images = result["images"]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (n_way, n_support + n_query, 3, 28, 28)

    # Checking the other parameters
    assert result["n_way"] == n_way
    assert result["n_support"] == n_support
    assert result["n_query"] == n_query


def test_extract_sample_no_duplicates(create_test_data):
    """We check that the images are different."""
    img_paths, labels = create_test_data
    n_way = 2
    n_support = 2
    n_query = 2

    result = extract_sample(n_way, n_support, n_query, img_paths, labels)
    images = result["images"]

    for class_images in images:
        class_images_np = class_images.numpy()
        for i in range(class_images_np.shape[0]):
            for j in range(i + 1, class_images_np.shape[0]):
                assert not np.array_equal(class_images_np[i], class_images_np[j]), "Found duplicate images"


def test_extract_sample_image_processing(create_test_data):
    """We check that the images are processed correctly."""
    img_paths, labels = create_test_data
    result = extract_sample(2, 1, 1, img_paths, labels)
    images = result["images"]

    # We check that the images are of the correct size and type
    assert images.dtype == torch.float32
    assert images.shape[2:] == (3, 28, 28)  # C, H, W

    # We check that the pixel values are in the acceptable range
    assert torch.all(images >= 0)
    assert torch.all(images <= 255)


def test_extract_sample_edge_cases(tmp_path):
    """We check for extreme cases."""
    # Create 2 classes with 2 images each
    img_paths = []
    labels = []
    for i in range(2):  # 2 classes
        for j in range(2):  # 2 images per class
            img_path = tmp_path / f"class_{i}_img_{j}.png"
            cv2.imwrite(str(img_path), np.zeros((50, 50, 3), dtype=np.uint8))
            img_paths.append(str(img_path))
            labels.append(i)

    img_paths = np.array(img_paths)
    labels = np.array(labels)

    # Test minimum parameters
    result = extract_sample(1, 1, 1, img_paths, labels)
    assert result["images"].shape == (1, 2, 3, 28, 28)  # Now should pass

    # Test insufficient data
    with pytest.raises(ValueError):
        extract_sample(2, 2, 2, img_paths[:3], labels[:3])  # Only 1.5 images per class


def test_extract_sample_invalid_input():
    """Checking the processing of invalid input data."""
    img_paths = np.array(["nonexistent_path.png"])
    labels = np.array([0])

    # Non-existent path to the image
    with pytest.raises(Exception):
        extract_sample(1, 1, 0, img_paths, labels)

    # Incorrect array sizes
    with pytest.raises(ValueError):
        extract_sample(1, 1, 0, img_paths, np.array([0, 1]))  # Different lengths

    # Invalid parameters
    with pytest.raises(ValueError):
        extract_sample(0, 1, 1, img_paths, labels)  # n_way = 0
    with pytest.raises(ValueError):
        extract_sample(1, -1, 1, img_paths, labels)  # n_support < 0
    with pytest.raises(ValueError):
        extract_sample(1, 1, -1, img_paths, labels)  # n_query < 0

    # Empty arrays
    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        extract_sample(1, 1, 1, np.array([]), np.array([]))

    # Not enough unique classes
    img_paths_multi = np.array(["img1.png", "img2.png", "img3.png"])
    labels_multi = np.array([0, 1, 1])  # Only 2 unique classes
    with pytest.raises(ValueError, match=r"Not enough unique classes. Requested 3, available 2"):
        extract_sample(3, 1, 1, img_paths_multi, labels_multi)

    # Not enough samples per class (added new test case)
    img_paths_limited = np.array(["img1.png", "img2.png", "img3.png"])
    labels_limited = np.array([0, 0, 1])  # Class 0 has 2 samples, class 1 has 1
    with pytest.raises(ValueError, match=r"Class 0 doesn't have enough samples"):
        extract_sample(2, 2, 1, img_paths_limited, labels_limited)


# Calling cleanup after all the tests
@pytest.fixture(autouse=True)
def cleanup_after_each_test():
    yield
    for i in range(3):
        img_path = f"dummy_path_{i}.png"
        if os.path.exists(img_path):
            os.remove(img_path)
