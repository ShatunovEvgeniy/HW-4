import cv2
import numpy as np
import torch
from torch import Tensor


def extract_sample(
    n_way: int, n_support: int, n_query: int, datax: np.ndarray, datay: np.ndarray
) -> dict[str, Tensor | int]:
    """
    Picks random sample of size n_support + n_querry, for n_way classes.
    :param n_way: Number of classes in a classification task.
    :param n_support: Number of labeled examples per class in the support set.
    :param n_query: Number of labeled examples per class in the query set.
    :param datax: Dataset of images.
    :param datay: Dataset of labels.
    :return:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support + n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
    """
    sample = []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    for cls in K:
        datax_cls = datax[datay == cls]
        perm = np.random.permutation(datax_cls)
        sample_cls = perm[: (n_support + n_query)]
        sample.append([cv2.resize(cv2.imread(fname), (28, 28)) for fname in sample_cls])

    sample = np.array(sample)
    sample = torch.from_numpy(sample).float()
    sample = sample.permute(0, 1, 4, 2, 3)
    return {"images": sample, "n_way": n_way, "n_support": n_support, "n_query": n_query}
