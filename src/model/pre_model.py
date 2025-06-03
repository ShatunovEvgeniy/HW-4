import torch
import torch.nn as nn
import torchvision

from src.model.projection_head import ProjectionHead


class PreModel(nn.Module):
    """Pre-training model for SimCLR with ResNet50 backbone.

    This implements the SimCLR architecture with:
    - ResNet50 encoder (pre-trained, frozen)
    - Projection head (trainable)
    """

    def __init__(self) -> None:
        super().__init__()

        # Load pretrained ResNet50 model
        model = torchvision.models.resnet50(pretrained=True)

        # Use all layers except final fully connected layer as encoder
        self.encoder = nn.Sequential(*tuple(model.children())[:-1])

        # Get embedding size from original model's fc layer
        emb_size = tuple(model.children())[-1].in_features

        # Freeze encoder parameters (only train projection head)
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Initialize projection head
        # Note: Using nonlinear head with hidden size 2048 as in original SimCLR
        self.projector = ProjectionHead(emb_size, hidden_features=2048, out_features=128, head_type="nonlinear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        :param x: Input images (batch_size, 3, H, W)
        :return: Projected representations (batch_size, 128)
        """
        # Get encoder features (output shape: [batch_size, 2048, 1, 1])
        out = self.encoder(x)

        # Remove spatial dimensions and project
        xp = self.projector(torch.squeeze(out))

        return xp
