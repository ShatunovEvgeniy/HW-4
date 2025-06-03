from typing import Any

import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    """Configurable linear layer with optional batch normalization.

    This layer combines a linear transformation with optional batch normalization
    in a single module for cleaner architecture definitions.
    """

    def __init__(
        self, in_features: int, out_features: int, use_bias: bool = True, use_bn: bool = False, **kwargs: Any
    ) -> None:
        """
        Initialize the linear layer.

        :param in_features: Size of input features
        :param out_features: Size of output features
        :param use_bias: Whether to use bias in linear layer
        :param use_bn: Whether to use batch normalization
        :param kwargs: Additional arguments to pass to parent class
        """
        super().__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        # Initialize linear layer
        # Note: Bias is disabled if using batch norm to avoid redundancy
        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.use_bias and not self.use_bn)

        # Initialize batch norm if requested
        if self.use_bn:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        :param x: Input tensor of shape (batch_size, in_features)
        :return: Output tensor of shape (batch_size, out_features)
        """
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x
