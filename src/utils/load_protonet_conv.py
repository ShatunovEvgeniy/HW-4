from pathlib import Path
import torch

from src.model.encoder import Encoder
from src.model.proto_net import ProtoNet
from src.model.pre_model import PreModel


def load_protonet_conv(
    x_dim: tuple[int, int, int] | None = (3, 28, 28), hid_dim: int | None = 64, z_dim: int | None = 64
) -> ProtoNet:
    """
    Loads a ProtoNet model with a convolutional encoder.

    This function constructs a ProtoNet with an encoder that consists of convolutional
    layers, batch normalization, ReLU activations, and max-pooling layers.

    :param x_dim: Dimension of input image (channels, height, width). Defaults to (3, 28, 28).
    :param hid_dim: Dimension of hidden layers in conv blocks.  This isn't actually used in the provided `Encoder`
                    implementation, but kept for potential future flexibility. Defaults to 64.
    :param z_dim: Dimension of the embedded image (output feature vector size). Defaults to 64.

    :return: ProtoNet: The constructed ProtoNet model.
    """
    if not isinstance(x_dim, tuple):
        raise TypeError("x_dim must be a tuple.")
    if len(x_dim) != 3:
        raise ValueError("x_dim must have length 3 (channels, height, width).")
    if not all(isinstance(dim, int) for dim in x_dim):
        raise ValueError("All elements of x_dim must be integers.")

    encoder = Encoder(in_channels=x_dim[0])  # Use in_channels from x_dim

    return ProtoNet(encoder)


def load_protonet_with_simCLR(sim_clr_path: Path) -> ProtoNet:
    """
    Loads a ProtoNet model with a pretrained SimCLR model.

    This function constructs a ProtoNet with a pretrained SimCLR.

    :param sim_clr_path: Path to weights of SimCLR.
    :return: ProtoNet: The constructed ProtoNet model.
    """
    if not isinstance(sim_clr_path, Path):
        raise TypeError("sim_clr_path must be a Path.")

    try:
        sim_clr = PreModel()
        state_dict = torch.load(str(sim_clr_path))["model_state_dict"]
        sim_clr.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Weights file not found at {sim_clr_path}")
        exit()
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        exit()

    return ProtoNet(sim_clr, False)
