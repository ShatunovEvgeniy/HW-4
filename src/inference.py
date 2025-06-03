from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.data.load_data import OmniglotLoader
from src.model.hparams import config
from src.utils.device import setup_device
from src.utils.load_protonet_conv import load_protonet_conv
from src.utils.logger import setup_logger

logger = setup_logger("inference")


def load_model_and_data(model_path: Path) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """
    Load model and data from source.
    :param model_path: Path to weights of the model.
    :return: Model, test_x and test_y data.
    """
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Weights file not found at {model_path}")

        DEVICE = setup_device()
        logger.info(f"Devise is {DEVICE}")
        PROJECT_ROOT = Path(__file__).parent.parent

        # Load data
        data_evaluation_path = PROJECT_ROOT / "data" / "images_evaluation"
        data_background_path = PROJECT_ROOT / "data" / "images_background"
        omniglot = OmniglotLoader(background_path=str(data_background_path), evaluation_path=str(data_evaluation_path))
        train_x, train_y, test_x, test_y = omniglot.load_data(augment_with_rotations=False)

        # Init model
        model = load_protonet_conv(
            x_dim=config["x_dim"],
            hid_dim=config["hid_dim"],
            z_dim=config["z_dim"],
        )
        model = model.to(DEVICE)

        # Load weights
        state_dict = torch.load(str(model_path), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model, test_x, test_y

    except Exception as e:
        raise RuntimeError(f"Loading failed: {str(e)}")
