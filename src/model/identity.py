import torch.nn as nn
import torch


class Identity(nn.Module):
    """Identity mapping layer that returns input unchanged.

    This is used as a placeholder when no transformation is needed.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns input unchanged.

        :param x: Input tensor
        :return: Same as input tensor
        """
        return x
