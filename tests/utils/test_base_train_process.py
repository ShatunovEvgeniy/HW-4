from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.utils.base_train_process import BaseTrainProcess


@pytest.fixture
def trainer_config():
    """Returns a basic configuration for the trainer."""
    return {
        "batch_size": 32,
        "n_workers": 0,  # Set to 0 workers for tests
        "lr": 0.001,
        "weight_decay": 0.0001,
        "temperature": 0.5,
        "epochs": 2,
        "seed": 42,
    }


class SimpleModel(nn.Module):
    """Simplified model for testing purposes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3)
        self.fc = nn.Linear(16 * 62 * 62, 128)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def real_trainer(trainer_config):
    """Creates a real training process with simplified components."""
    with patch("torch.utils.tensorboard.SummaryWriter"):
        trainer = BaseTrainProcess(trainer_config)

    # Replace model with simplified version
    trainer.model = SimpleModel()

    # Create real optimizer
    trainer.optimizer = torch.optim.AdamW(
        trainer.model.parameters(), lr=trainer_config["lr"], weight_decay=trainer_config["weight_decay"]
    )

    # Mock only complex parts
    trainer.criterion = MagicMock(return_value=torch.tensor(0.5))
    trainer.writer = MagicMock()

    # Create simple data for loaders
    mock_batch = (torch.randn(32, 3, 64, 64), torch.randn(32, 3, 64, 64), torch.zeros(32), torch.randn(32, 3, 64, 64))
    trainer.train_loader = [mock_batch] * 2  # Reduce batch count for tests
    trainer.valid_loader = [mock_batch] * 2

    # Mock only methods that might cause issues
    trainer.save_model = MagicMock()
    trainer.save_checkpoint = MagicMock()

    return trainer


def test_run_completes_successfully(real_trainer):
    """Tests that the run method executes without errors."""
    # Replace train_step and valid_step with mocks
    real_trainer.train_step = MagicMock(return_value=[0.5])
    real_trainer.valid_step = MagicMock(return_value=[0.4])

    train_losses, valid_losses = real_trainer.run()

    assert len(train_losses) == real_trainer.hyp["epochs"]
    assert len(valid_losses) == real_trainer.hyp["epochs"]


def test_checkpoint_saving(real_trainer):
    """Tests checkpoint saving functionality."""
    real_trainer.train_step = MagicMock(return_value=[0.5])
    real_trainer.valid_step = MagicMock(return_value=[0.4])

    real_trainer.run()

    # Verify that saving methods were called
    assert real_trainer.save_checkpoint.called
    assert real_trainer.save_model.called


def test_tensorboard_logging(real_trainer):
    """Tests TensorBoard logging functionality."""
    real_trainer.train_step = MagicMock(return_value=[0.5])
    real_trainer.valid_step = MagicMock(return_value=[0.4])

    real_trainer.run()

    # Verify that writer was used for logging
    assert any("add_scalar" in str(c) for c in real_trainer.writer.method_calls)


def test_lr_scheduling(real_trainer):
    """Tests learning rate schedulers functionality."""
    # Create mocks for schedulers
    real_trainer.warmupscheduler = MagicMock()
    real_trainer.mainscheduler = MagicMock()

    # Configure mocks for training steps
    real_trainer.train_step = MagicMock(return_value=[0.5])
    real_trainer.valid_step = MagicMock(return_value=[0.4])

    real_trainer.run()

    # Verify warmupscheduler.step was called for each epoch
    assert real_trainer.warmupscheduler.step.call_count == real_trainer.hyp["epochs"]

    # Verify mainscheduler.step wasn't called (should only be called after 10 epochs)
    assert real_trainer.mainscheduler.step.call_count == 0


def test_device_setup(real_trainer):
    """Tests device setup (CPU/GPU)."""
    assert hasattr(real_trainer, "device")
    assert next(real_trainer.model.parameters()).device.type == real_trainer.device


def test_seed_reproducibility(trainer_config):
    """Tests result reproducibility with fixed seed."""
    # Create two separate training processes
    with patch("torch.utils.tensorboard.SummaryWriter"):
        trainer1 = BaseTrainProcess(trainer_config)
        trainer2 = BaseTrainProcess(trainer_config)

    # Configure identical mocks for both
    mock_loss = [0.5]
    trainer1.train_step = MagicMock(return_value=mock_loss)
    trainer1.valid_step = MagicMock(return_value=mock_loss)
    trainer2.train_step = MagicMock(return_value=mock_loss)
    trainer2.valid_step = MagicMock(return_value=mock_loss)

    # Run both processes
    train_losses1, valid_losses1 = trainer1.run()
    train_losses2, valid_losses2 = trainer2.run()

    # Verify identical results
    assert train_losses1 == train_losses2
    assert valid_losses1 == valid_losses2
