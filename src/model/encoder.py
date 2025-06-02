import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    A convolutional encoder module.

    This encoder takes an image as input and processes it through a series of convolutional
    layers, batch normalization, ReLU activation, and max-pooling layers. It outputs a
    flattened vector representation of the input image.

    """

    def __init__(self, in_channels: int = 3):
        """
        Initializes the Encoder.

        :param in_channels: The number of input channels (e.g., 3 for RGB images).  Defaults to 3.
        """
        super(Encoder, self).__init__()

        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.module2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.module3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.module4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the encoder.

        :param x: The input tensor, expected to be a batch of images (batch_size, channels, height, width).
        :return: The output tensor, a flattened vector of shape (batch_size, 64).
        """
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)

        # Flatten the output to get a vector of length 64
        x = x.view(x.size(0), -1)  # Flattening, -1 infers the size
        return x
