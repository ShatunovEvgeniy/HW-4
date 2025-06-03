from typing import Any

import pytest
import torch

from src.model.encoder import Encoder
from src.model.proto_net import ProtoNet
from src.utils.device import setup_device


def generate_sample(
    n_way: int, n_support: int, n_query: int, img_size: int = 28, in_channels: int = 3
) -> dict[str, Any]:
    """
    Generates a dummy sample for testing ProtoNet.

    :param n_way: Number of classes.
    :param n_support: Number of support examples per class.
    :param n_query: Number of query examples per class.
    :param img_size: Size of images. Defaults to 28.
    :param in_channels: Input channels for images.

    :return: A dictionary representing a sample.
    """
    images = torch.randn(n_way, n_support + n_query, in_channels, img_size, img_size)
    return {
        "images": images,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
    }


@pytest.fixture
def protonet(encoder):
    """Pytest fixture to create a ProtoNet instance."""
    return ProtoNet(encoder=encoder)


@pytest.fixture
def encoder():
    """Pytest fixture to create an instance of the Encoder."""
    return Encoder()


class TestProtoNet:
    """Tests for the ProtoNet module."""

    def test_init(self, encoder):
        """Tests if the ProtoNet initializes correctly."""
        protonet = ProtoNet(encoder)
        assert protonet.encoder is not None
        assert isinstance(protonet.encoder, Encoder)

    def test_get_prototypes(self, protonet):
        """Tests the _get_prototypes method."""
        embeddings = torch.randn(2, 5 + 5, 64)
        n_way = 2
        n_support = 5
        prototypes = protonet._get_prototypes(embeddings, n_way, n_support)
        assert prototypes.shape == torch.Size([n_way, 64])

    def test_get_distances(self, protonet):
        """Tests the _get_distances method."""
        prototypes = torch.randn(2, 64)
        query_embeddings = torch.randn(10, 64)
        distances = protonet._get_distances(prototypes, query_embeddings)
        assert distances.shape == torch.Size([10, 2])

    def test_set_forward_loss_output_shape(self, protonet):
        """Tests if the set_forward_loss method produces the correct output shapes."""
        sample = generate_sample(n_way=2, n_support=5, n_query=5)
        loss, results = protonet.set_forward_loss(sample)
        assert loss.shape == torch.Size([])  # Scalar loss
        assert isinstance(results, dict)
        assert "loss" in results and isinstance(results["loss"], float)
        assert "acc" in results and isinstance(results["acc"], float)
        assert "y_hat" in results and results["y_hat"].shape == torch.Size([10])  # n_way * n_query

    def test_set_forward_loss_with_different_sizes(self, protonet):
        """Tests set_forward_loss with different numbers of ways, support, and query."""
        sample = generate_sample(n_way=3, n_support=1, n_query=3)
        loss, results = protonet.set_forward_loss(sample)
        assert results["y_hat"].shape == torch.Size([9])
        sample = generate_sample(n_way=5, n_support=5, n_query=1)
        loss, results = protonet.set_forward_loss(sample)
        assert results["y_hat"].shape == torch.Size([5])

    def test_set_forward_loss_with_grayscale(self):
        """Tests set_forward_loss with a grayscale input."""
        sample = generate_sample(n_way=2, n_support=5, n_query=5, in_channels=1)  # Grayscale
        protonet = ProtoNet(encoder=Encoder(in_channels=1))
        loss, results = protonet.set_forward_loss(sample)
        assert loss.shape == torch.Size([])
        assert results["y_hat"].shape == torch.Size([10])

    def test_set_forward_loss_data_validation(self, protonet):
        """Tests if set_forward_loss validates input data correctly."""
        with pytest.raises(TypeError):
            protonet.set_forward_loss("not a dict")
        with pytest.raises(ValueError):
            protonet.set_forward_loss(
                {"not_images": torch.randn(2, 10, 3, 28, 28), "n_way": 2, "n_support": 5, "n_query": 5}
            )
        with pytest.raises(TypeError):
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            sample["images"] = "not a tensor"
            protonet.set_forward_loss(sample)
        with pytest.raises(ValueError):
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            sample["images"] = torch.randn(2, 10, 3, 28)
            protonet.set_forward_loss(sample)
        with pytest.raises(ValueError):
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            sample["n_way"] = -1
            protonet.set_forward_loss(sample)
        with pytest.raises(ValueError):
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            sample["n_support"] = -1
            protonet.set_forward_loss(sample)
        with pytest.raises(ValueError):
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            sample["n_query"] = -1
            protonet.set_forward_loss(sample)

    def test_set_forward_loss_on_cuda(self, protonet):
        """Tests if the ProtoNet works correctly on CUDA (GPU)."""
        device = setup_device()
        if device.type == "cuda":
            protonet.encoder = protonet.encoder.to(device)
            sample = generate_sample(n_way=2, n_support=5, n_query=5)
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    sample[key] = value.to(device)

            loss, results = protonet.set_forward_loss(sample)
            assert loss.is_cuda
            assert results["y_hat"].is_cuda
        else:
            pytest.skip("CUDA is not available")
