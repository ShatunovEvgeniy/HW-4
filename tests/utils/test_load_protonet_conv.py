import pytest

from src.model.encoder import Encoder
from src.model.proto_net import ProtoNet
from src.utils.load_protonet_conv import load_protonet_conv


class TestLoadProtonetConv:
    """Tests for the load_protonet_conv function."""

    def test_load_protonet_conv_default(self):
        """Tests load_protonet_conv with default arguments."""
        model = load_protonet_conv()
        assert isinstance(model, ProtoNet)
        assert isinstance(model.encoder, Encoder)

    def test_load_protonet_conv_with_custom_x_dim(self):
        """Tests load_protonet_conv with a custom input dimension."""
        x_dim = (1, 32, 32)  # Grayscale, 32x32 image
        model = load_protonet_conv(x_dim=x_dim)
        assert isinstance(model, ProtoNet)
        assert isinstance(model.encoder, Encoder)

    def test_load_protonet_conv_with_custom_hid_dim(self):
        """Tests load_protonet_conv with a custom hidden dimension (although it's not used directly)."""
        hid_dim = 128
        model = load_protonet_conv(hid_dim=hid_dim)
        assert isinstance(model, ProtoNet)
        assert isinstance(model.encoder, Encoder)

    def test_load_protonet_conv_with_custom_z_dim(self):
        """Tests load_protonet_conv with a custom embedding dimension (z_dim)."""
        z_dim = 128
        model = load_protonet_conv(z_dim=z_dim)
        assert isinstance(model, ProtoNet)
        assert isinstance(model.encoder, Encoder)

    def test_load_protonet_conv_data_validation(self):
        """Tests if load_protonet_conv validates input data correctly."""
        with pytest.raises(TypeError):
            load_protonet_conv(x_dim="not a tuple")
        with pytest.raises(ValueError):
            load_protonet_conv(x_dim=(1, 2))
        with pytest.raises(ValueError):
            load_protonet_conv(x_dim=(1.0, 28, 28))
