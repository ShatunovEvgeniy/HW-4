import numpy as np
import pytest
import torch

from src.utils.l2_norm import l2_norm


class TestL2Norm:
    @pytest.fixture(params=[1, 2, 3])
    def random_input(self, request):
        """Generate random input tensors of different dimensions"""
        dim = request.param
        if dim == 1:
            return torch.randn(10)  # 1D tensor
        elif dim == 2:
            return torch.randn(5, 7)  # 2D tensor
        else:
            return torch.randn(3, 4, 5)  # 3D tensor

    @pytest.fixture(params=[0, -1])
    def axis(self, request):
        """Generate different axis values to test"""
        return request.param

    def test_basic_functionality(self):
        """Test basic normalization works correctly"""
        input_tensor = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        expected_output = torch.tensor([[0.6, 0.8], [0.4472, 0.8944]])

        output = l2_norm(input_tensor)
        assert torch.allclose(output, expected_output, atol=1e-4)

    def test_different_axes(self, random_input, axis):
        """Test normalization along different axes"""
        output = l2_norm(random_input, axis=axis)

        # Verify output shape matches input shape
        assert output.shape == random_input.shape

        # Verify norms along specified axis are 1 (or 0 for zero vectors)
        norms = torch.norm(output, 2, axis)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6) or torch.allclose(
            norms, torch.zeros_like(norms), atol=1e-6
        )

    def test_normalization_property(self, random_input, axis):
        """Verify the L2 norm property after normalization"""
        output = l2_norm(random_input, axis=axis)
        computed_norms = torch.norm(output, 2, axis)

        # All norms should be 1 (except for zero vectors)
        is_non_zero = torch.norm(random_input, 2, axis) > 1e-6
        assert torch.allclose(computed_norms[is_non_zero], torch.ones_like(computed_norms[is_non_zero]), atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow correctly through normalization"""
        x = torch.randn(3, 4, requires_grad=True)
        y = l2_norm(x)

        # Create dummy loss and backprop
        dummy_loss = y.sum()
        dummy_loss.backward()

        # Verify gradients exist
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_invalid_axis(self):
        """Test behavior with invalid axis"""
        input_tensor = torch.randn(3, 4)

        with pytest.raises(IndexError):
            l2_norm(input_tensor, axis=3)  # Axis out of bounds

        with pytest.raises(IndexError):
            l2_norm(input_tensor, axis=-3)  # Axis out of bounds

    def test_empty_tensor(self):
        """Test behavior with empty tensor"""
        input_tensor = torch.tensor([])

        with pytest.raises(IndexError):
            l2_norm(input_tensor)

    def test_against_numpy(self):
        """Test against numpy's l2 norm for verification"""
        np_input = np.random.randn(3, 4)
        torch_input = torch.from_numpy(np_input)

        # Compute with torch
        torch_output = l2_norm(torch_input)

        # Compute with numpy
        np_norms = np.linalg.norm(np_input, ord=2, axis=1, keepdims=True)
        np_output = np_input / np_norms

        assert np.allclose(torch_output.numpy(), np_output, atol=1e-6)

    def test_batch_processing(self):
        """Test that batch processing works correctly"""
        batch_input = torch.randn(5, 3, 4)  # Batch of 5, each 3x4
        output = l2_norm(batch_input, axis=2)

        # Verify each element in batch is normalized correctly
        for i in range(5):
            element = batch_input[i]
            expected_norm = torch.norm(element, 2, 1, keepdim=True)
            expected_output = element / expected_norm
            assert torch.allclose(output[i], expected_output, atol=1e-6)
