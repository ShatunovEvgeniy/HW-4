import pytest
import torch
import torchvision

from src.model.pre_model import PreModel
from src.model.projection_head import ProjectionHead


@pytest.fixture
def pre_model():
    """Fixture providing initialized PreModel"""
    return PreModel()


@pytest.fixture
def test_input():
    """Fixture providing test input tensor"""
    return torch.randn(4, 3, 224, 224)  # Batch of 4 RGB 224x224 images


class TestPreModel:
    def test_initialization(self, pre_model):
        """Test that PreModel initializes correctly"""
        # Verify encoder is ResNet50 without final FC layer
        assert isinstance(pre_model.encoder, torch.nn.Sequential)
        assert len(pre_model.encoder) == len(list(torchvision.models.resnet50().children())) - 1

        # Verify encoder is frozen
        for param in pre_model.encoder.parameters():
            assert not param.requires_grad

        # Verify projector is initialized
        assert isinstance(pre_model.projector, ProjectionHead)
        assert pre_model.projector.head_type == "nonlinear"
        assert pre_model.projector.hidden_features == 2048
        assert pre_model.projector.out_features == 128

        # Verify projector is trainable
        for param in pre_model.projector.parameters():
            assert param.requires_grad

    def test_forward_pass(self, pre_model, test_input):
        """Test forward pass produces correct output shape"""
        output = pre_model(test_input)

        # Verify output shape
        batch_size = test_input.shape[0]
        assert output.shape == (batch_size, 128)  # Should match projector out_features

        # Verify no NaN values
        assert not torch.isnan(output).any()

    def test_encoder_frozen(self, pre_model, test_input):
        """Test encoder parameters remain frozen during forward/backward"""
        # Get initial encoder parameters
        initial_params = [p.data.clone() for p in pre_model.encoder.parameters()]

        # Forward and backward pass
        test_input.requires_grad_(True)
        output = pre_model(test_input)
        loss = output.sum()
        loss.backward()

        # Verify encoder parameters didn't change
        for initial, param in zip(initial_params, pre_model.encoder.parameters()):
            assert torch.equal(initial, param.data)

        # Verify encoder gradients are None
        for param in pre_model.encoder.parameters():
            assert param.grad is None

    @pytest.mark.parametrize("input_shape", [(0, 3, 224, 224), (4, 3, 256, 256), (8, 3, 128, 128)])
    def test_various_input_sizes(self, pre_model, input_shape: tuple):
        """Test handles different input sizes correctly"""
        input_tensor = torch.randn(input_shape)
        output = pre_model(input_tensor)

        batch_size = input_shape[0]
        assert output.shape == (batch_size, 128)

    def test_serialization(self, pre_model, test_input, tmp_path):
        """Test can be serialized and deserialized correctly"""
        model_path = tmp_path / "pre_model.pth"
        torch.save(pre_model.state_dict(), model_path)

        # Create new model
        new_model = PreModel()
        new_model.load_state_dict(torch.load(model_path))

        # Compare outputs
        original_output = pre_model(test_input)
        new_output = new_model(test_input)
        assert torch.allclose(original_output, new_output)

    def test_squeeze_operation(self, pre_model):
        """Test the squeeze operation handles different spatial dimensions"""
        # Test with single spatial dimension (1x1)
        input1 = torch.randn(2, 3, 224, 224)
        out1 = pre_model.encoder(input1)  # Should be [2, 2048, 1, 1]
        proj1 = pre_model.projector(torch.squeeze(out1))
        assert proj1.shape == (2, 128)

        # Test with larger spatial dimensions (should still work after adaptive pooling)
        input2 = torch.randn(2, 3, 256, 256)
        out2 = pre_model.encoder(input2)  # Should still be [2, 2048, 1, 1] due to adaptive pooling
        proj2 = pre_model.projector(torch.squeeze(out2))
        assert proj2.shape == (2, 128)
