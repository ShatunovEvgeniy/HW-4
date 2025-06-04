from typing import Any

import torch
import torch.nn as nn

from src.model.linear_layer import LinearLayer
from src.utils.l2_norm import l2_norm


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning.

    Maps encoder outputs to space where contrastive loss is applied.
    Can be either linear or nonlinear (MLP) as specified.
    """

    def __init__(
        self, in_features: int, hidden_features: int, out_features: int, head_type: str = "nonlinear", **kwargs: Any
    ) -> None:
        """
        Initialize projection head.

        :param in_features: Size of input features from encoder
        :param hidden_features: Size of hidden layer (for nonlinear head)
        :param out_features: Size of output/projection space
        :param head_type: Type of head ('linear' or 'nonlinear')
        :param kwargs: Additional arguments to pass to parent class
        """
        super().__init__(**kwargs)

        if head_type not in ["nonlinear", "linear"]:
            raise ValueError('head_type must be either "linear" or "nonlinear".')

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        # Initialize appropriate architecture based on head type
        if self.head_type == "linear":
            # Simple linear projection with batch norm
            self.layers = LinearLayer(self.in_features, self.out_features, use_bias=False, use_bn=True)
        if self.head_type == "nonlinear":
            # Nonlinear projection head (MLP with one hidden layer)
            self.layers = nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, use_bias=True, use_bn=False),
                nn.ReLU(),  # Nonlinearity between layers
                LinearLayer(self.hidden_features, self.out_features, use_bias=False, use_bn=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head.

        Applies L2 normalization before projection layers.

        :param x: Input tensor from encoder
        :return: Projected representation in contrastive space
        """
        # Apply projection layers
        x = self.layers(x)
        # Normalize input before projection
        x = l2_norm(x)
        return x
