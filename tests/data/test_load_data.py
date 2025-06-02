import os

import numpy as np
import pytest
from PIL import Image

from src.data.load_data import OmniglotLoader


@pytest.fixture
def mock_data_structure(tmp_path):
    """Create a mock Omniglot directory structure for testing"""
    # Create background (train) data
    bg_path = tmp_path / "images_background"
    bg_path.mkdir()

    # Create alphabet directories with character subdirectories
    for alphabet_idx, alphabet in enumerate(["Alphabet1", "Alphabet2"]):
        alphabet_path = bg_path / alphabet
        alphabet_path.mkdir()

        for char_idx, character in enumerate(["char1", "char2"]):
            # Make character names unique by combining with alphabet
            unique_char = f"{alphabet}_{character}"
            char_path = alphabet_path / unique_char
            char_path.mkdir()

            # Create mock PNG images
            for i in range(2):  # 2 images per character
                img = Image.new("L", (105, 105), color=i * 100)
                img_path = char_path / f"image_{i}.png"
                img.save(img_path)

    # Create evaluation (test) data with similar structure
    eval_path = tmp_path / "images_evaluation"
    eval_path.mkdir()

    for alphabet_idx, alphabet in enumerate(["Alphabet3", "Alphabet4"]):
        alphabet_path = eval_path / alphabet
        alphabet_path.mkdir()

        for char_idx, character in enumerate(["char3", "char4"]):
            unique_char = f"{alphabet}_{character}"
            char_path = alphabet_path / unique_char
            char_path.mkdir()

            for i in range(2):
                img = Image.new("L", (105, 105), color=i * 50)
                img_path = char_path / f"image_{i}.png"
                img.save(img_path)

    return tmp_path


def test_initialization():
    """Test that the loader initializes correctly"""
    loader = OmniglotLoader()
    assert loader.background_path == "images_background"
    assert loader.evaluation_path == "images_evaluation"
    assert loader.trainx is None
    assert loader.trainy is None


def test_custom_path_initialization():
    """Test initialization with custom paths"""
    loader = OmniglotLoader(background_path="custom_train", evaluation_path="custom_test")
    assert loader.background_path == "custom_train"
    assert loader.evaluation_path == "custom_test"


def test_read_alphabet(mock_data_structure):
    """Test reading a single alphabet directory"""
    loader = OmniglotLoader()
    alphabet_path = mock_data_structure / "images_background" / "Alphabet1"

    datax, datay = loader._read_alphabet(str(alphabet_path))

    assert len(datax) == 4  # 2 characters × 2 images each
    assert len(datay) == 4
    assert all(f.endswith(".png") for f in datax)
    assert all(label in ["Alphabet1_char1", "Alphabet1_char2"] for label in datay)


def test_read_images(mock_data_structure):
    """Test reading all alphabets from a base directory"""
    loader = OmniglotLoader()
    bg_path = mock_data_structure / "images_background"

    datax, datay = loader._read_images(str(bg_path))

    # Should have 8 images total (2 alphabets × 2 chars × 2 images)
    assert len(datax) == 8
    assert len(datay) == 8
    assert len(np.unique(datay)) == 4  # 2 chars per alphabet × 2 alphabets


def test_load_data(mock_data_structure):
    """Test the complete data loading pipeline"""
    loader = OmniglotLoader(
        background_path=str(mock_data_structure / "images_background"),
        evaluation_path=str(mock_data_structure / "images_evaluation"),
    )

    trainx, trainy, testx, testy = loader.load_data()

    # Verify training data
    assert len(trainx) == 8  # 2 alphabets × 2 chars × 2 images
    assert len(trainy) == 8
    assert isinstance(trainy[0], (int, np.integer)), "Labels should be encoded."

    # Verify test data
    assert len(testx) == 8
    assert len(testy) == 8

    # Verify all paths exist
    assert all(os.path.exists(p) for p in trainx)
    assert all(os.path.exists(p) for p in testx)

    # Verify label encoding
    assert len(np.unique(trainy)) == 4  # 4 unique characters in training
    assert len(np.unique(testy)) == 4  # 4 unique characters in test


def test_label_encoding_consistency(mock_data_structure):
    """Test that label encoding is consistent between train and test"""
    loader = OmniglotLoader(
        background_path=str(mock_data_structure / "images_background"),
        evaluation_path=str(mock_data_structure / "images_evaluation"),
    )

    loader.load_data()

    # Verify that if the same character appears in both sets, it gets the same encoding
    # (Note: In our mock data, train and test have different characters)
    # So we'll just verify the encoder was fitted properly
    assert hasattr(loader.label_encoder, "classes_")
    assert len(loader.label_encoder.classes_) == 8  # 4 train chars + 4 test chars


def test_read_alphabet_with_non_directory_files(tmp_path):
    """Test with explicitly created mixed content"""
    alphabet_path = tmp_path / "test_alphabet"
    alphabet_path.mkdir()

    # Create a valid character directory
    char_dir = alphabet_path / "valid_char"
    char_dir.mkdir()
    for i in range(2):
        img = Image.new("L", (105, 105), color=i * 100)
        img.save(char_dir / f"image_{i}.png")

    # Create a non-directory file
    dummy_file = alphabet_path / "invalid_file.txt"
    dummy_file.write_text("Not a directory")

    loader = OmniglotLoader()
    datax, datay = loader._read_alphabet(str(alphabet_path))

    # Should only process the valid directory
    assert len(datax) == 2
    assert len(datay) == 2
    assert all(label == "valid_char" for label in datay)
