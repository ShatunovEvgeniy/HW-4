import pytest
import torch
import torch.nn as nn

from src.model.linear_layer import LinearLayer
from src.model.projection_head import ProjectionHead  # Update import path as needed


@pytest.fixture(params=[32, 64])  # Different input feature sizes
def in_features(request):
    return request.param


@pytest.fixture(params=[128, 256])  # Different hidden layer sizes
def hidden_features(request):
    return request.param


@pytest.fixture(params=[64, 128])  # Different output sizes
def out_features(request):
    return request.param


@pytest.fixture(params=["linear", "nonlinear"])  # Both head types
def head_type(request):
    return request.param


@pytest.fixture
def projection_head(in_features, hidden_features, out_features, head_type):
    """Fixture providing initialized ProjectionHead with different configs"""
    return ProjectionHead(
        in_features=in_features, hidden_features=hidden_features, out_features=out_features, head_type=head_type
    )


@pytest.fixture
def test_input(in_features):
    """Fixture providing test input tensor matching the projection head"""
    return torch.randn(8, in_features)  # Batch size of 8


class TestProjectionHead:
    def test_initialization(self, projection_head, in_features, out_features, head_type):
        """Test that ProjectionHead initializes correctly"""
        assert projection_head.in_features == in_features
        assert projection_head.out_features == out_features
        assert projection_head.head_type == head_type

        if head_type == "linear":
            assert isinstance(projection_head.layers, LinearLayer)
            assert projection_head.layers.out_features == out_features
        elif head_type == "nonlinear":
            assert isinstance(projection_head.layers, nn.Sequential)
            assert len(projection_head.layers) == 3  # Linear -> ReLU -> Linear
            assert projection_head.layers[0].out_features == projection_head.hidden_features
            assert projection_head.layers[2].out_features == out_features

    def test_forward_pass(self, projection_head, test_input):
        """Test forward pass produces correct output shape"""
        output = projection_head(test_input)

        # Verify output shape
        batch_size = test_input.shape[0]
        assert output.shape == (batch_size, projection_head.out_features)

        # Verify no NaN values
        assert not torch.isnan(output).any()

    def test_backward_pass(self, projection_head, test_input):
        """Test backward pass propagates gradients correctly"""
        test_input.requires_grad_(True)
        output = projection_head(test_input)

        # Create dummy loss and backprop
        dummy_loss = output.sum()
        dummy_loss.backward()

        # Verify gradients exist where they should
        assert test_input.grad is not None
        assert test_input.grad.shape == test_input.shape

        # Verify all learnable parameters have gradients
        for param in projection_head.parameters():
            assert param.grad is not None
            assert param.grad.shape == param.shape

    def test_serialization(self, projection_head, test_input, tmp_path):
        """Test can be serialized and deserialized correctly"""
        model_path = tmp_path / "projection_head.pth"
        torch.save(projection_head.state_dict(), model_path)

        # Create new head with same config
        new_head = ProjectionHead(
            in_features=projection_head.in_features,
            hidden_features=projection_head.hidden_features,
            out_features=projection_head.out_features,
            head_type=projection_head.head_type,
        )
        new_head.load_state_dict(torch.load(model_path))

        # Compare outputs
        original_output = projection_head(test_input)
        new_output = new_head(test_input)
        assert torch.allclose(original_output, new_output)

    def test_invalid_head_type(self):
        """Test behavior with invalid head type"""
        with pytest.raises(ValueError):
            ProjectionHead(64, 128, 64, head_type="invalid")
