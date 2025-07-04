import random
from pathlib import Path
from time import gmtime, strftime
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.load_data import OmniglotLoader
from src.data.prepare_dataset import CLDataset
from src.model.pre_model import PreModel
from src.utils.device import setup_device
from src.utils.logger import setup_logger
from src.utils.simclr_loss import SimCLR_Loss

logger = setup_logger("BaseTrainProcess")


class BaseTrainProcess:
    """
    Implements the base training process for contrastive learning models.

    This class handles the complete training pipeline including:
    - Data loading and preparation
    - Model initialization
    - Training and validation loops
    - Learning rate scheduling
    - Checkpoint saving
    - TensorBoard logging

    :param hyp: Dictionary containing hyperparameters for training.
                Expected keys: 'batch_size', 'n_workers', 'lr',
                'weight_decay', 'temperature', 'epochs'.
    """

    def __init__(self, hyp: Dict[str, Any], save_dir: str = "output") -> None:
        """
        Initializes the training process with given hyperparameters.

        Creates logging directory, initializes TensorBoard writer,
        and sets up device configuration (CPU/GPU).

        :param hyp: Hyperparameters dictionary.
        :param save_dir: Base directory for saving outputs.
        """
        self.save_dir = Path(save_dir) / "SimCLR"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        start_time = strftime("%Y-%m-%d %H-%M-%S", gmtime())
        log_dir = (self.save_dir / "logs" / start_time).as_posix()
        logger.info(f"Log dir: {log_dir}")
        self.writer = SummaryWriter(log_dir)

        self.best_loss = 1e100
        self.best_acc = 0.0
        self.current_epoch = -1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.hyp = hyp

        self.lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[torch.nn.Module] = None

        self.train_loader: Optional[DataLoader] = None
        self.valid_loader: Optional[DataLoader] = None

        self.init_params()

    def _init_data(self) -> None:
        """
        Initializes data loaders for training and validation.

        Creates DataLoader instances with specified batch size and workers.
        Uses pin_memory for faster GPU transfer when available.
        """
        train_transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.Rotate(limit=(90, 90), p=1),  # поворот на 90°
                        A.Rotate(limit=(180, 180), p=1),  # поворот на 180°
                        A.Rotate(limit=(270, 270), p=1),  # поворот на 270°
                    ],
                    p=1,
                ),
                ToTensorV2(),
            ]
        )

        valid_transform = A.Compose([A.ToFloat(max_value=255), ToTensorV2()])

        data_loader = OmniglotLoader()
        trainx, trainy, testx, testy = data_loader.load_data(augment_with_rotations=False)

        trainx = [cv2.resize(cv2.imread(trainx[idx]), (28, 28)) for idx in range(len(trainx))]
        testx = [cv2.resize(cv2.imread(testx[idx]), (28, 28)) for idx in range(len(testx))]

        train_dataset = CLDataset(trainx, trainy, train_transform)
        valid_dataset = CLDataset(testx, testy, valid_transform)
        logger.info(f"Train size: {len(train_dataset)}, Valid size: {len(valid_dataset)}")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyp["batch_size"],
            shuffle=True,
            num_workers=self.hyp["n_workers"],
            pin_memory=False,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.hyp["batch_size"],
            shuffle=True,
            num_workers=self.hyp["n_workers"],
            pin_memory=False,
            drop_last=True,
        )

    def _init_model(self) -> None:
        """
        Initializes model architecture and training components.

        Sets up:
        - The PreModel architecture
        - AdamW optimizer with specified learning rate and weight decay
        - Learning rate warmup scheduler (first 10 epochs)
        - Cosine annealing with restarts scheduler (main training)
        - SimCLR loss function with temperature parameter
        """
        self.model = PreModel()
        self.model.to(self.device)

        model_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(model_params, lr=self.hyp["lr"], weight_decay=self.hyp["weight_decay"])

        self.warmupscheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: (epoch + 1) / 10.0)

        self.mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=500,
            eta_min=0.05,
            last_epoch=-1,
        )

        self.criterion = SimCLR_Loss(batch_size=self.hyp["batch_size"], temperature=self.hyp["temperature"]).to(
            self.device
        )

    def init_params(self) -> None:
        """
        Initializes all training parameters.

        Sequentially calls data and model initialization methods.
        """
        self._init_data()
        self._init_model()

    def save_checkpoint(self, loss_valid: List[float], path: str) -> None:
        """
        Saves model checkpoint if validation loss improves.

        Compares current validation loss with best recorded loss.
        Only saves model if current loss is better than previous best.

        :param loss_valid: Current validation loss values.
        :param path: Path to save the checkpoint file.
        """
        if loss_valid[0] <= self.best_loss:
            self.best_loss = loss_valid[0]
            self.save_model(path)

    def save_model(self, path: str) -> None:
        """
        Saves complete model state to file.

        Includes:
        - Model parameters
        - Optimizer state
        - Scheduler state

        :param path: Destination file path for saving.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.mainscheduler.state_dict(),
            },
            path,
        )

    def train_step(self) -> List[float]:
        """
        Executes one complete training epoch.

        Processes all batches in training dataset:
        1. Moves data to target device
        2. Computes embeddings for augmented pairs
        3. Calculates contrastive loss
        4. Performs backpropagation
        5. Updates progress bar with current loss

        :return: List containing average training loss for the epoch.
        """
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        cum_loss = 0.0
        proc_loss = 0.0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}',
        )

        for idx, (xi, xj, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)

            with torch.set_grad_enabled(True):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.zero_grad()

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            pbar.set_description(f'Train {self.current_epoch}/{self.hyp["epochs"] - 1}, ' f'Loss: {proc_loss:4.3f}')

        return [cum_loss / len(self.train_loader)]

    def valid_step(self) -> List[float]:
        """
        Executes one complete validation epoch.

        Processes all batches in validation dataset:
        1. Moves data to target device
        2. Computes embeddings in eval mode
        3. Calculates contrastive loss
        4. Updates progress bar with current loss

        :return: List containing average validation loss for the epoch.
        """
        self.model.eval()

        cum_loss = 0.0
        proc_loss = 0.0

        pbar = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc=f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}',
        )

        for idx, (xi, xj, _) in pbar:
            xi, xj = xi.to(self.device), xj.to(self.device)

            with torch.set_grad_enabled(False):
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)

            cur_loss = loss.detach().cpu().numpy()
            cum_loss += cur_loss
            proc_loss = (proc_loss * idx + cur_loss) / (idx + 1)

            pbar.set_description(f'Valid {self.current_epoch}/{self.hyp["epochs"] - 1}, ' f'Loss: {proc_loss:4.3f}')

        return [cum_loss / len(self.valid_loader)]

    def run(self) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Executes complete training process.

        Handles:
        - Training loop for specified number of epochs
        - Learning rate scheduling
        - Validation after each epoch
        - Checkpoint saving
        - Metric logging to TensorBoard
        - Final model cleanup

        :return: Tuple of (train_losses, valid_losses) where each is a
                 list of loss values per epoch.
        """
        # Создаем подпапки для чекпоинтов
        checkpoints_dir = self.save_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)

        best_w_path = checkpoints_dir / "best.pt"
        last_w_path = checkpoints_dir / "last.pt"

        train_losses = []
        valid_losses = []

        for epoch in range(self.hyp["epochs"]):
            self.current_epoch = epoch

            loss_train = self.train_step()
            train_losses.append(loss_train)

            if epoch < 10:
                self.warmupscheduler.step()
            else:
                self.mainscheduler.step()

            loss_valid = self.valid_step()
            valid_losses.append(loss_valid)

            self.save_checkpoint(loss_valid, best_w_path.as_posix())
            lr = self.optimizer.param_groups[0]["lr"]

            self.writer.add_scalar("Train/Loss", loss_train[0], epoch)
            self.writer.add_scalar("Valid/Loss", loss_valid[0], epoch)
            self.writer.add_scalar("Lr", lr, epoch)

        self.save_model(last_w_path.as_posix())
        torch.cuda.empty_cache()
        self.writer.close()

        return train_losses, valid_losses


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    DEVICE = setup_device()
    logger.info(f"Device is {DEVICE}")
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    hyp_params_path = PROJECT_ROOT / "src" / "utils" / "hyp_params.yaml"
    with open(str(hyp_params_path), "r") as f:
        hyps = yaml.load(f, Loader=yaml.SafeLoader)

    logger.info(f"Hyper parameters: {hyps}")

    set_seed(hyps["seed"])

    save_directory = PROJECT_ROOT / "model"
    trainer = BaseTrainProcess(hyps, save_dir=str(save_directory))
    trainer.device = DEVICE

    train_losses, valid_losses = trainer.run()
