from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.train import train


@pytest.fixture
def mock_wandb():
    with patch("wandb.log") as mock_log:
        yield mock_log


@pytest.fixture
def mock_model():
    model = MagicMock(spec=nn.Module)

    # Create a proper mock for set_forward_loss
    def set_forward_loss(sample):
        loss = torch.tensor(0.5, requires_grad=True)
        return loss, {"loss": 0.5, "acc": 0.8}

    model.set_forward_loss = MagicMock(side_effect=set_forward_loss)
    return model


@pytest.fixture
def mock_optimizer():
    optimizer = MagicMock(spec=optim.Adam)
    optimizer.param_groups = [{"lr": 0.1}]
    optimizer.state_dict.return_value = {}
    optimizer.step = MagicMock()
    optimizer.zero_grad = MagicMock()
    return optimizer


@pytest.fixture
def mock_extract_sample():
    with patch("src.train.extract_sample") as mock:
        # Return a properly structured sample dictionary
        def mock_sample(n_way, n_support, n_query, train_x, train_y):
            # Create mock tensors for the sample
            sample_images = torch.randn(n_way, n_support + n_query, 3, 28, 28)
            return {"images": sample_images, "n_way": n_way, "n_support": n_support, "n_query": n_query}

        mock.side_effect = mock_sample
        yield mock


@pytest.fixture
def mock_data():
    train_x = np.random.randint(0, 256, size=(100, 3, 28, 28), dtype=np.uint8)
    train_y = np.random.randint(0, 10, size=100)  # 100 labels (0-9)
    return train_x, train_y


def test_train_basic(mock_model, mock_optimizer, mock_data, mock_extract_sample, mock_wandb):
    train_x, train_y = mock_data

    train(
        model=mock_model,
        optimizer=mock_optimizer,
        train_x=train_x,
        train_y=train_y,
        n_way=5,
        n_support=1,
        n_query=1,
        max_epoch=1,
        epoch_size=10,
        use_wandb=True,
    )

    # Verify basic behaviors
    assert mock_model.set_forward_loss.call_count == 10
    assert mock_optimizer.zero_grad.call_count == 10
    assert mock_optimizer.step.call_count == 10
    assert mock_wandb.call_count == 1  # Should log once (every 100 episodes, but we have 10)


def test_train_multiple_epochs(mock_model, mock_optimizer, mock_data, mock_extract_sample):
    train_x, train_y = mock_data

    train(
        model=mock_model,
        optimizer=mock_optimizer,
        train_x=train_x,
        train_y=train_y,
        n_way=5,
        n_support=1,
        n_query=1,
        max_epoch=3,
        epoch_size=5,
        use_wandb=False,
    )

    assert mock_model.set_forward_loss.call_count == 15  # 3 epochs * 5 episodes
    assert mock_optimizer.step.call_count == 15


def test_train_without_wandb(mock_model, mock_optimizer, mock_data, mock_extract_sample, mock_wandb):
    train_x, train_y = mock_data

    train(
        model=mock_model,
        optimizer=mock_optimizer,
        train_x=train_x,
        train_y=train_y,
        n_way=5,
        n_support=1,
        n_query=1,
        max_epoch=1,
        epoch_size=10,
        use_wandb=False,
    )

    assert mock_wandb.call_count == 0  # No wandb calls when use_wandb=False
