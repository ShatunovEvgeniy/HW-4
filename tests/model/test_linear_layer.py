import pytest
import torch
import torch.nn as nn

from src.model.linear_layer import LinearLayer


@pytest.fixture(params=[(10, 20), (5, 3), (1, 1)])  # Various (in_features, out_features) combinations
def layer_config(request):
    """Fixture providing different layer configurations"""
    return request.param


@pytest.fixture(params=[True, False])  # Test both with and without batch norm
def use_bn(request):
    """Fixture providing batch norm options"""
    return request.param


@pytest.fixture(params=[True, False])  # Test both with and without bias
def use_bias(request):
    """Fixture providing bias options"""
    return request.param


@pytest.fixture
def linear_layer(layer_config, use_bn, use_bias):
    """Fixture providing initialized LinearLayer with different configurations"""
    in_features, out_features = layer_config
    return LinearLayer(in_features, out_features, use_bias=use_bias, use_bn=use_bn)


@pytest.fixture
def test_input(layer_config):
    """Fixture providing test input tensor matching layer config"""
    in_features, _ = layer_config
    return torch.randn(4, in_features)  # Batch size of 4


class TestLinearLayer:
    def test_initialization(self, linear_layer, layer_config, use_bn, use_bias):
        """Test that LinearLayer initializes correctly"""
        in_features, out_features = layer_config

        # Check attributes
        assert linear_layer.in_features == in_features
        assert linear_layer.out_features == out_features
        assert linear_layer.use_bias == use_bias
        assert linear_layer.use_bn == use_bn

        # Check linear layer configuration
        assert isinstance(linear_layer.linear, nn.Linear)
        assert linear_layer.linear.in_features == in_features
        assert linear_layer.linear.out_features == out_features

        # Check batch norm configuration
        if use_bn:
            assert isinstance(linear_layer.bn, nn.BatchNorm1d)
            assert linear_layer.bn.num_features == out_features
        else:
            assert not hasattr(linear_layer, "bn")

        # Check bias handling
        if use_bn and use_bias:
            assert linear_layer.linear.bias is None  # Bias should be disabled when using BN

    def test_forward_pass(self, linear_layer, test_input):
        """Test forward pass produces correct output shape"""
        output = linear_layer(test_input)

        # Verify output shape
        batch_size = test_input.shape[0]
        assert output.shape == (batch_size, linear_layer.out_features)

        # Verify no NaN values
        assert not torch.isnan(output).any()

    def test_backward_pass(self, linear_layer, test_input):
        """Test backward pass propagates gradients correctly"""
        test_input.requires_grad_(True)
        output = linear_layer(test_input)

        # Create dummy loss and backprop
        dummy_loss = output.sum()
        dummy_loss.backward()

        # Verify gradients exist where they should
        assert test_input.grad is not None
        assert test_input.grad.shape == test_input.shape

        # Verify linear layer gradients
        assert linear_layer.linear.weight.grad is not None
        assert linear_layer.linear.weight.grad.shape == linear_layer.linear.weight.shape

        if linear_layer.linear.bias is not None:
            assert linear_layer.linear.bias.grad is not None
            assert linear_layer.linear.bias.grad.shape == linear_layer.linear.bias.shape

        if linear_layer.use_bn:
            assert linear_layer.bn.weight.grad is not None
            assert linear_layer.bn.bias.grad is not None

    def test_serialization(self, linear_layer, test_input, tmp_path):
        """Test can be serialized and deserialized correctly"""
        model_path = tmp_path / "linear_layer.pth"
        torch.save(linear_layer.state_dict(), model_path)

        # Create new layer with same config
        new_layer = LinearLayer(
            linear_layer.in_features,
            linear_layer.out_features,
            use_bias=linear_layer.use_bias,
            use_bn=linear_layer.use_bn,
        )
        new_layer.load_state_dict(torch.load(model_path))

        # Compare outputs
        original_output = linear_layer(test_input)
        new_output = new_layer(test_input)
        assert torch.allclose(original_output, new_output)

    def test_bias_handling(self):
        """Test special case of bias handling with batch norm"""
        # Case 1: use_bias=True, use_bn=False -> should have bias
        layer1 = LinearLayer(5, 3, use_bias=True, use_bn=False)
        assert layer1.linear.bias is not None

        # Case 2: use_bias=True, use_bn=True -> should NOT have bias
        layer2 = LinearLayer(5, 3, use_bias=True, use_bn=True)
        assert layer2.linear.bias is None

        # Case 3: use_bias=False, use_bn=True -> should NOT have bias
        layer3 = LinearLayer(5, 3, use_bias=False, use_bn=True)
        assert layer3.linear.bias is None
