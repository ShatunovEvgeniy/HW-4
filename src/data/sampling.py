from typing import Any, Dict, List

import cv2
import numpy as np
import torch


def extract_sample(
    n_way: int,
    n_support: int,
    n_query: int,
    datax: np.ndarray,
    datay: np.ndarray,
) -> Dict[str, Any]:
    """
    Extract random samples for few-shot learning task.

    :param n_way: Number of classes in the classification task.
    :param n_support: Number of labeled examples per class in the support set.
    :param n_query: Number of labeled examples per class in the query set.
    :param datax: Dataset of images (file paths) as a NumPy array.
    :param datay: Dataset of corresponding labels as a NumPy array.
    :return: Dictionary containing:
             - 'images': Tensor of sampled images with shape (n_way, n_support + n_query, C, H, W).
             - 'n_way': Number of classes.
             - 'n_support': Size of support set per class.
             - 'n_query': Size of query set per class.
    """
    sample: List[List[np.ndarray]] = []
    K: np.ndarray = np.random.choice(np.unique(datay), n_way, replace=False)

    for cls in K:
        datax_cls: np.ndarray = datax[datay == cls]
        perm: np.ndarray = np.random.permutation(datax_cls)
        sample_cls: np.ndarray = perm[: (n_support + n_query)]
        sample.append([cv2.resize(cv2.imread(fname), (28, 28)) for fname in sample_cls])

    sample_array: np.ndarray = np.array(sample)
    sample_tensor: torch.Tensor = torch.from_numpy(sample_array).float()
    sample_tensor = sample_tensor.permute(0, 1, 4, 2, 3)

    return {
        "images": sample_tensor,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
    }
