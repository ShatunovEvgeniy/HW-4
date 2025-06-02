from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.device import setup_device


class ProtoNet(nn.Module):
    """
    Prototypical Network for few-shot classification.

    This class implements a Prototypical Network, which learns a representation space
    where classes are represented by prototypes (mean embeddings of support set examples).
    Classification is performed by assigning query examples to the nearest prototype.
    """

    def __init__(self, encoder: nn.Module):
        """
        Initializes the ProtoNet.

        :param encoder: CNN encoding the images in sample. Should output a feature vector.
        """
        super(ProtoNet, self).__init__()
        self.encoder = encoder

    def _get_prototypes(self, embeddings: torch.Tensor, n_way: int, n_support: int) -> torch.Tensor:
        """
        Calculates the prototypes (class means) from the support set embeddings.

        :param embeddings: Embeddings of support and query images. Shape: (n_way, n_support + n_query, embedding_dim)
        :param n_way: Number of classes.
        :param n_support: Number of support examples per class.

        :return: Prototypes for each class. Shape: (n_way, embedding_dim)
        """
        # Reshape to (n_way, n_support, embedding_dim) then take the mean along the support dimension (dim=1)
        prototypes = embeddings[:, :n_support].mean(dim=1)
        return prototypes

    def _get_distances(self, prototypes: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Euclidean distances between query embeddings and prototypes.

        :param prototypes: Prototypes for each class. Shape: (n_way, embedding_dim)
        :param query_embeddings: Embeddings of the query images. Shape: (n_way * n_query, embedding_dim)

        :return: Distances between query embeddings and prototypes.  Shape: (n_way * n_query, n_way)
        """
        # Reshape prototypes to (1, n_way, embedding_dim) to allow broadcasting.
        # Expand prototypes to (n_way * n_query, n_way, embedding_dim)
        prototypes_expanded = prototypes.unsqueeze(0).expand(query_embeddings.size(0), *prototypes.shape)
        # Reshape query_embeddings to (n_way * n_query, 1, embedding_dim) so it can be broadcasted for distance calc.
        query_embeddings_expanded = query_embeddings.unsqueeze(1)  # (n_way * n_query, 1, embedding_dim)
        # Calculate the distances between query and prototype
        distances = torch.sum((query_embeddings_expanded - prototypes_expanded) ** 2, dim=2)
        return distances

    def set_forward_loss(self, sample: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Union[float, torch.Tensor]]]:
        """
        Computes the loss, accuracy, and predicted labels for a few-shot classification task.

        :param sample: A dictionary containing:
                        'images' (torch.Tensor):  A tensor of shape (n_way, n_support + n_query, C, H, W)
                                           containing the images.
                        'n_way' (int): Number of classes in the episode.
                        'n_support' (int): Number of support examples per class.
                        'n_query' (int): Number of query examples per class.

        :return: A tuple containing:
                - loss_val (torch.Tensor): The loss value.
                - A dictionary with keys:
                    - 'loss' (float): The loss value (itemized).
                    - 'acc' (float): The accuracy.
                    - 'y_hat' (torch.Tensor): The predicted labels for the query set.
        """
        # Data Validation
        if not isinstance(sample, dict):
            raise TypeError("sample must be a dictionary.")
        if "images" not in sample or "n_way" not in sample or "n_support" not in sample or "n_query" not in sample:
            raise ValueError("sample must contain 'images', 'n_way', 'n_support', and 'n_query' keys.")
        images = sample["images"]
        n_way = sample["n_way"]
        n_support = sample["n_support"]
        n_query = sample["n_query"]

        if not isinstance(images, torch.Tensor):
            raise TypeError("'images' must be a torch.Tensor.")
        if images.dim() != 5:
            raise ValueError("'images' tensor must have 5 dimensions (n_way, n_support + n_query, C, H, W).")
        if not isinstance(n_way, int) or n_way <= 0:
            raise ValueError("'n_way' must be a positive integer.")
        if not isinstance(n_support, int) or n_support <= 0:
            raise ValueError("'n_support' must be a positive integer.")
        if not isinstance(n_query, int) or n_query <= 0:
            raise ValueError("'n_query' must be a positive integer.")

        # Move images to the device
        device = setup_device()
        images = images.to(device)  # (n_way, n_support + n_query, C, H, W)

        # Encode the images
        embeddings = self.encoder(images.reshape(-1, *images.shape[2:])).reshape(
            images.shape[0], images.shape[1], -1
        )  # (n_way, n_support + n_query, embedding_dim)

        # Calculate prototypes
        prototypes = self._get_prototypes(embeddings, n_way, n_support)  # (n_way, embedding_dim)

        # Extract query embeddings
        query_embeddings = embeddings[:, n_support:]  # (n_way, n_query, embedding_dim)
        query_embeddings = query_embeddings.reshape(n_way * n_query, -1)  # (n_way * n_query, embedding_dim)

        # Calculate distances
        distances = self._get_distances(prototypes, query_embeddings)  # (n_way * n_query, n_way)

        # Calculate the loss
        log_p_y = F.log_softmax(-distances, dim=1)  # (n_way * n_query, n_way)
        target_inds = (
            torch.arange(0, n_way).view(n_way, 1).expand(n_way, n_query).reshape(-1).to(device)
        )  # (n_way * n_query)
        loss_val = -log_p_y.gather(1, target_inds.view(-1, 1)).squeeze().mean()

        # Calculate accuracy
        _, y_hat = torch.max(log_p_y, dim=1)
        acc_val = torch.eq(y_hat, target_inds).float().mean()

        return loss_val, {"loss": loss_val.item(), "acc": acc_val.item(), "y_hat": y_hat}
