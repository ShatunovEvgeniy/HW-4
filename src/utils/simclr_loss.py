import torch
import torch.nn as nn


class SimCLR_Loss(nn.Module):
    """SimCLR contrastive loss function implementation.

    This computes the normalized temperature-scaled cross entropy loss (NT-Xent)
    as described in the SimCLR paper: https://arxiv.org/pdf/2002.05709

    Args:
        batch_size: Number of samples in the batch (before augmentation)
        temperature: Temperature parameter τ for scaling the similarity scores
    """

    def __init__(self, batch_size: int, temperature: float) -> None:
        """
        Initialize the SimCLR loss function.

        :param batch_size: Number of samples in the original batch (before augmentation)
        :param temperature: Temperature parameter τ for scaling the similarity scores
        """
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        # Create mask to exclude self and augmented pairs from negative samples
        self.mask = self.mask_correlated_samples(batch_size)

        # Cross entropy loss for the similarity scores
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        # Cosine similarity function for comparing embeddings
        self.similarity_f = nn.CosineSimilarity(dim=2)

        # Counter for debugging/analysis
        self.tot_neg = 0

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute the contrastive loss between two sets of embeddings.

        Each pair (z_i[k], z_j[k]) are positive pairs (different augmentations
        of the same sample), all other pairs are considered negative samples.

        :param z_i: First set of embeddings (from first augmentation pipeline)
                    Shape: (batch_size, embedding_dim)
        :param z_j: Second set of embeddings (from second augmentation pipeline)
                    Shape: (batch_size, embedding_dim)
        :return: Computed contrastive loss value
        """
        N = 2 * self.batch_size

        # Concatenate both sets of embeddings
        z = torch.cat((z_i, z_j), dim=0)  # Shape: (2*batch_size, embedding_dim)

        # Compute pairwise cosine similarities
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature  # Shape: (2N, 2N)

        # Extract positive pairs (diagonals of the two main blocks)
        sim_i_j = torch.diag(sim, self.batch_size)  # pos for (z_i, z_j)
        sim_j_i = torch.diag(sim, -self.batch_size)  # pos for (z_j, z_i)

        # Combine positive samples
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # Shape: (2N, 1)

        # Get negative samples using mask
        negative_samples = sim[self.mask].reshape(N, -1)  # Shape: (2N, 2N-2)
        self.tot_neg += negative_samples.numel()  # Track for debugging

        # Prepare labels (all zeros since positive samples are first column)
        labels = torch.zeros(N, dtype=torch.long).to(z_i.device)

        # Combine positive and negative samples for cross entropy
        logits = torch.cat((positive_samples, negative_samples), dim=1)  # Shape: (2N, 2N-1)

        # Compute loss
        loss = self.criterion(logits, labels)
        loss /= N  # Normalize by batch size

        return loss

    @staticmethod
    def mask_correlated_samples(batch_size: int) -> torch.Tensor:
        """Create mask to exclude self and augmented pairs from negative samples.

        The mask is a boolean tensor where:
        - True indicates valid negative pairs
        - False indicates pairs to exclude (diagonal and augmented pairs)

        :param batch_size: Number of samples in the original batch
        :return: Boolean mask tensor of shape (2*batch_size, 2*batch_size)
        """
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # Exclude self-similarity

        # Exclude pairs between original and augmented versions of same sample
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask
