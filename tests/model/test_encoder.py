import pytest
import torch

from src.model.encoder import Encoder


@pytest.fixture
def encoder():
    """Pytest fixture to create an instance of the Encoder."""
    return Encoder()


@pytest.fixture
def input_tensor_rgb():
    """Pytest fixture to create a standard RGB input tensor."""
    return torch.randn(1, 3, 28, 28)


@pytest.fixture
def input_tensor_grayscale():
    """Pytest fixture for a grayscale input tensor."""
    return torch.randn(1, 1, 28, 28)


@pytest.fixture
def input_tensor_batch():
    """Pytest fixture for a batch input tensor"""
    return torch.randn(8, 3, 28, 28)


@pytest.fixture
def input_tensor_small():
    """Pytest fixture for a small size input"""
    return torch.randn(1, 3, 16, 16)


@pytest.fixture
def input_tensor_grad():
    return torch.randn(1, 3, 28, 28, requires_grad=True)


class TestEncoder:
    """Tests for the Encoder module."""

    def test_encoder_output_shape(self, encoder, input_tensor_rgb):
        """Tests if the encoder produces the expected output shape."""
        output_tensor = encoder(input_tensor_rgb)
        assert output_tensor.shape == torch.Size([1, 64])

    def test_encoder_with_different_input_channels(self, encoder, input_tensor_grayscale):
        """Tests the encoder with a different number of input channels."""
        encoder = Encoder(in_channels=1)  # Grayscale image
        output_tensor = encoder(input_tensor_grayscale)
        assert output_tensor.shape == torch.Size([1, 64])

    def test_encoder_with_larger_batch_size(self, encoder, input_tensor_batch):
        """Tests the encoder with a larger batch size."""
        output_tensor = encoder(input_tensor_batch)
        assert output_tensor.shape == torch.Size([8, 64])

    def test_encoder_with_small_input(self, encoder, input_tensor_small):
        """Tests the encoder with a smaller input size."""
        output_tensor = encoder(input_tensor_small)
        assert output_tensor.shape[1] == 64
