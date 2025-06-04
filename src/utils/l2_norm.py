import torch

from src.utils.logger import setup_logger

logger = setup_logger("l2_norm")


def l2_norm(input: torch.Tensor, axis: int = 1) -> torch.Tensor:
    """Normalize input tensor along specified axis using L2 norm.

    This is used for projection head outputs in contrastive learning.

    :param input: Input tensor to normalize
    :param axis: Axis along which to compute norm
    :return: L2-normalized tensor
    """
    # Compute L2 norm along specified axis (keeping dimensions)
    norm = torch.norm(input, 2, axis, True)
    # Normalize by dividing by norm
    if (norm != 0).all():
        output = torch.div(input, norm)
        return output
    else:
        logger.warning("l2 norm is 0!!!")
        return input
