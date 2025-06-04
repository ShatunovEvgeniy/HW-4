from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

from src.data.load_data import OmniglotLoader
from src.data.sampling import extract_sample
from src.model.hparams import config
from src.utils.device import setup_device
from src.utils.load_protonet_conv import load_protonet_conv
from src.utils.logger import setup_logger

logger = setup_logger("train")


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_x: np.ndarray,
    train_y: np.ndarray,
    n_way: int,
    n_support: int,
    n_query: int,
    max_epoch: int,
    epoch_size: int,
    use_wandb: bool = True,
) -> None:
    """
    Trains the Protonet.
    :param model: Model to train.
    :param optimizer: Optimizer for training process.
    :param train_x: Images of training set.
    :param train_y: Labels of training set.
    :param n_way: Number of classes in a classification task.
    :param n_support: Number of labeled examples per class in the support set.
    :param n_query: Number of labeled examples per class in the query set.
    :param max_epoch: Max epochs to train on.
    :param epoch_size: Episodes per epoch.
    :param use_wandb: If True it logs info in wandb.
    :return: None.
    """
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc=f"Epoch {epoch + 1} train"):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
            if episode % 100 == 0 and use_wandb:
                step = epoch * epoch_size + episode
                metrics = dict(loss=output["loss"], accuracy=output["acc"])
                wandb.log(metrics, step=step)

        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        logger.info("Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(epoch + 1, epoch_loss, epoch_acc))
        epoch += 1
        scheduler.step()


def model_test(
    model: nn.Module,
    test_x: np.ndarray,
    test_y: np.ndarray,
    n_way: int,
    n_support: int,
    n_query: int,
    test_episode: int,
) -> dict[str, Union[float, torch.Tensor]]:
    """
    Tests the Protonet.
    :param model: Trained model.
    :param test_x: Images of testing set.
    :param test_y: Labels of testing set.
    :param n_way: Number of classes in a classification task.
    :param n_support: Number of labeled examples per class in the support set.
    :param n_query: Number of labeled examples per class in the query set.
    :param test_episode: Number of episodes to test on.
    :return: None.
    """
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output["loss"]
        running_acc += output["acc"]

    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    logger.info("Test results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))
    return output


if __name__ == "__main__":
    DEVICE = setup_device()
    logger.info(f"Devise is {DEVICE}")
    PROJECT_ROOT = Path(__file__).parent.parent

    # Load data
    data_background_path = PROJECT_ROOT / "data" / "images_background"
    data_evaluation_path = PROJECT_ROOT / "data" / "images_evaluation"
    omniglot = OmniglotLoader(background_path=str(data_background_path), evaluation_path=str(data_evaluation_path))
    train_x, train_y, test_x, test_y = omniglot.load_data(augment_with_rotations=config["augment_flag"])

    # Init model
    model = load_protonet_conv(
        x_dim=config["x_dim"],
        hid_dim=config["hid_dim"],
        z_dim=config["z_dim"],
    )
    model = model.to(DEVICE)

    # Init optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Initialize wandb session
    wandb.init(
        config=config,
        project="ML Homework-4",
        name=f"Protonet without SimCLR: {config["n_way"]}-way, {config["n_support"]}-shot, {config["n_query"]}-query",
    )
    wandb.watch(model)

    # Train
    train(
        model=model,
        optimizer=optimizer,
        train_x=train_x,
        train_y=train_y,
        n_way=config["n_way"],
        n_support=config["n_support"],
        n_query=config["n_query"],
        max_epoch=config["max_epoch"],
        epoch_size=config["epoch_size"],
    )

    model_test(
        model=model,
        test_x=test_x,
        test_y=test_y,
        n_way=config["n_way"],
        n_support=config["n_support"],
        n_query=config["n_query"],
        test_episode=1000,
    )

    model.save_model(
        f"protonet_without_simclr_" f"{config["n_way"]}-way, {config["n_support"]}-shot, {config["n_query"]}-query"
    )
    wandb.finish()
