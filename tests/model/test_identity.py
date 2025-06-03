import pytest
import torch
from src.model.identity import Identity


@pytest.fixture
def identity_layer():
    """Fixture providing initialized Identity layer"""
    return Identity()


@pytest.fixture
def test_input():
    """Fixture providing test input tensor"""
    return torch.randn(3, 5, 7)  # 3D tensor for thorough testing


class TestIdentity:
    def test_initialization(self):
        """Test that Identity initializes correctly"""
        layer = Identity()
        assert isinstance(layer, torch.nn.Module)
        assert len(list(layer.parameters())) == 0  # No learnable parameters

    def test_forward_pass(self, identity_layer, test_input):
        """Test forward pass returns exact input tensor"""
        output = identity_layer(test_input)

        # Verify output is identical to input
        assert torch.equal(output, test_input)
        assert output.data_ptr() == test_input.data_ptr()  # Same memory location

        # Verify no computation graph is created
        assert output.grad_fn is None

    def test_input_types(self, identity_layer):
        """Test handles different input types correctly"""
        # Test with 1D tensor
        input_1d = torch.randn(10)
        output_1d = identity_layer(input_1d)
        assert torch.equal(output_1d, input_1d)

        # Test with 2D tensor
        input_2d = torch.randn(4, 7)
        output_2d = identity_layer(input_2d)
        assert torch.equal(output_2d, input_2d)

        # Test with requires_grad=True
        input_grad = torch.randn(2, 3, requires_grad=True)
        output_grad = identity_layer(input_grad)
        assert output_grad.requires_grad
        assert torch.equal(output_grad, input_grad)

    def test_backward_pass(self, identity_layer):
        """Test backward pass propagates gradients correctly"""
        x = torch.randn(3, 4, requires_grad=True)
        y = identity_layer(x)

        # Create dummy loss and backprop
        dummy_loss = y.sum()
        dummy_loss.backward()

        # Verify gradients exist and are correct
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))


    def test_serialization(self, identity_layer, tmp_path):
        """Test can be serialized and deserialized correctly"""
        model_path = tmp_path / "identity.pth"
        torch.save(identity_layer.state_dict(), model_path)

        new_layer = Identity()
        new_layer.load_state_dict(torch.load(model_path))

        test_input = torch.randn(5, 5)
        assert torch.equal(identity_layer(test_input), new_layer(test_input))
