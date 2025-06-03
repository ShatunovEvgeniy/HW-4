import pytest
import torch

from src.utils.simclr_loss import SimCLR_Loss


@pytest.fixture(params=[16, 32])  # Test different batch sizes
def batch_size(request):
    return request.param


@pytest.fixture(params=[0.1, 0.5])  # Test different temperatures
def temperature(request):
    return request.param


@pytest.fixture
def simclr_loss(batch_size, temperature):
    return SimCLR_Loss(batch_size, temperature)


@pytest.fixture
def embeddings(batch_size):
    """Generate random normalized embeddings"""
    emb_dim = 128
    z_i = torch.randn(batch_size, emb_dim)
    z_j = torch.randn(batch_size, emb_dim)

    # Normalize embeddings (important for cosine similarity)
    z_i = torch.nn.functional.normalize(z_i, dim=1)
    z_j = torch.nn.functional.normalize(z_j, dim=1)

    return z_i, z_j


class TestSimCLRLoss:
    def test_initialization(self, simclr_loss, batch_size, temperature):
        """Test that SimCLR_Loss initializes correctly"""
        assert simclr_loss.batch_size == batch_size
        assert simclr_loss.temperature == temperature
        assert isinstance(simclr_loss.criterion, torch.nn.CrossEntropyLoss)
        assert isinstance(simclr_loss.similarity_f, torch.nn.CosineSimilarity)

        # Verify mask shape and properties
        mask = simclr_loss.mask
        N = 2 * batch_size
        assert mask.shape == (N, N)
        assert torch.all(mask.diagonal() == 0)  # Diagonal should be 0

        # Verify augmented pairs are masked
        for i in range(batch_size):
            assert mask[i, batch_size + i] == 0
            assert mask[batch_size + i, i] == 0

    def test_mask_correlated_samples(self, batch_size):
        """Test the mask creation function"""
        mask = SimCLR_Loss.mask_correlated_samples(batch_size)
        N = 2 * batch_size
        assert mask.shape == (N, N)
        assert mask.dtype == torch.bool

        # Verify diagonal is False
        assert not torch.any(mask.diagonal())

        # Verify augmented pairs are False
        for i in range(batch_size):
            assert not mask[i, batch_size + i]
            assert not mask[batch_size + i, i]

        # Verify other entries are True
        expected_true = N * N - 2 * N  # N diagonal + N augmented pairs
        assert torch.sum(mask) == expected_true

    def test_forward_pass(self, simclr_loss, embeddings):
        """Test forward pass produces valid loss"""
        z_i, z_j = embeddings
        loss = simclr_loss(z_i, z_j)

        # Verify loss is scalar
        assert loss.dim() == 0
        assert not torch.isnan(loss)

        # Loss should be positive
        assert loss > 0

    def test_positive_pair_similarity(self, simclr_loss):
        """Test that identical pairs have high similarity"""
        batch_size = simclr_loss.batch_size
        emb_dim = 128

        # Create identical embeddings
        z = torch.randn(batch_size, emb_dim)
        z = torch.nn.functional.normalize(z, dim=1)
        z_i = z
        z_j = z.clone()

        loss = simclr_loss(z_i, z_j)

        # With identical embeddings, loss should be lower than random case
        random_z_j = torch.randn(batch_size, emb_dim)
        random_z_j = torch.nn.functional.normalize(random_z_j, dim=1)
        random_loss = simclr_loss(z_i, random_z_j)

        assert loss < random_loss

    def test_gradient_flow(self, simclr_loss, embeddings):
        """Test that gradients can flow through the loss"""
        z_i, z_j = embeddings
        z_i.requires_grad_(True)
        z_j.requires_grad_(True)

        loss = simclr_loss(z_i, z_j)
        loss.backward()

        # Verify gradients exist
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert z_i.grad.shape == z_i.shape
        assert z_j.grad.shape == z_j.shape

    def test_temperature_effect(self, batch_size):
        """Test that temperature affects the loss"""
        emb_dim = 128
        z_i = torch.randn(batch_size, emb_dim)
        z_j = torch.randn(batch_size, emb_dim)
        z_i = torch.nn.functional.normalize(z_i, dim=1)
        z_j = torch.nn.functional.normalize(z_j, dim=1)

        # High temperature should smooth the loss
        high_temp_loss = SimCLR_Loss(batch_size, temperature=1.0)(z_i, z_j)

        # Low temperature should sharpen the loss
        low_temp_loss = SimCLR_Loss(batch_size, temperature=0.1)(z_i, z_j)

        assert not torch.isclose(high_temp_loss, low_temp_loss)

    def test_negative_samples_count(self, simclr_loss, embeddings):
        """Test the number of negative samples is correct"""
        z_i, z_j = embeddings
        _ = simclr_loss(z_i, z_j)

        N = 2 * simclr_loss.batch_size
        expected_neg = N * (N - 2)  # Each sample has N-2 negatives (excluding self and positive)
        assert simclr_loss.tot_neg == expected_neg
