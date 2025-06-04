from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.data.load_data import OmniglotLoader
from src.data.sampling import extract_sample
from src.model.hparams import config
from src.utils.device import setup_device
from src.utils.load_protonet_conv import load_protonet_conv
from src.utils.logger import setup_logger

logger = setup_logger("Inference")


def load_model_and_data(model_path: Path) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """
    Load model and data from source.
    :param model_path: Path to weights of the model.
    :return: Model, test_x and test_y data.
    """
    try:
        if not model_path.exists():
            raise FileNotFoundError(f"Weights file not found at {model_path}")

        DEVICE = setup_device()
        logger.info(f"Devise is {DEVICE}")
        PROJECT_ROOT = Path(__file__).parent.parent

        # Load data
        data_evaluation_path = PROJECT_ROOT / "data" / "images_evaluation"
        data_background_path = PROJECT_ROOT / "data" / "images_background"
        omniglot = OmniglotLoader(background_path=str(data_background_path), evaluation_path=str(data_evaluation_path))
        train_x, train_y, test_x, test_y = omniglot.load_data(augment_with_rotations=False)

        # Init model
        model = load_protonet_conv(
            x_dim=config["x_dim"],
            hid_dim=config["hid_dim"],
            z_dim=config["z_dim"],
        )
        model = model.to(DEVICE)

        # Load weights
        state_dict = torch.load(str(model_path), map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        return model, test_x, test_y

    except Exception as e:
        raise RuntimeError(f"Loading failed: {str(e)}")


def visualize_inference(y_hat, target_inds, acc_val, title="Inference Results"):
    """
    Визуализирует результаты инференса, показывая предсказанные и истинные метки.

    Args:
        y_hat (torch.Tensor): Тензор с предсказанными классами.
        target_inds (torch.Tensor): Тензор с истинными метками.
        acc_val (float): Точность.
        title (str): Заголовок графика.
    """
    y_hat = y_hat.cpu().numpy()  # Move to CPU and convert to NumPy arrays
    target_inds = target_inds.cpu().numpy()
    n_samples = len(y_hat)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size for better visualization

    # Create a bar chart
    bar_width = 0.4  # Increased bar width for readability
    x = np.arange(n_samples)

    # Correct predictions in green, incorrect in red
    colors = ["green" if y_hat[i] == target_inds[i] else "red" for i in range(n_samples)]
    bars = ax.bar(x, 1, bar_width, bottom=0, color=colors)  # draw a bar for each sample, height is always 1
    # Add labels to the bars (pred_label / true_label)
    for bar, pred, true in zip(bars, y_hat, target_inds):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f"{pred} / {true}", ha="center", va="bottom", fontsize=8)

    # Customize the plot
    ax.set_xlabel("Запросные изображения", fontsize=12)  # Added label
    ax.set_ylabel("Правильность предсказания", fontsize=12)
    ax.set_title(f"{title} - Точность: {acc_val:.2f}", fontsize=14)  # added title
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_ylim(0, 1.2)  # Correct height range

    # Add a legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor="green", label="Правильно"), Patch(facecolor="red", label="Неправильно")]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()


if __name__ == "__main__":
    device = setup_device()
    PROJECT_ROOT = Path(__file__).parent.parent
    model_path = PROJECT_ROOT / "model" / "protonet_without_simclr_5-way, 5-shot, 5-query"
    model, test_x, test_y = load_model_and_data(model_path)
    model = model.to(device)

    n_way = 2
    n_support = 5
    n_query = 6

    sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
    loss, output = model.set_forward_loss(sample)
    target_inds = torch.arange(0, n_way).view(n_way, 1).expand(n_way, n_query).reshape(-1).to(device)

    y_hat = output["y_hat"]
    acc_val = output["acc"]

    yaml_path = PROJECT_ROOT / "notebooks" / "inference_results.yaml"

    results = {
        "predicted_classes": y_hat.cpu().tolist(),  # Convert to list for YAML
        "target_labels": target_inds.cpu().tolist(),
        "accuracy": acc_val,
        "loss": loss.item(),  # Assuming loss is a tensor
    }

    try:
        with open(str(yaml_path), "w") as outfile:  # changed
            yaml.dump(results, outfile, default_flow_style=False)  # Use default_flow_style=False for readability
        logger.info(f"Results saved to {yaml_path}")
    except Exception as e:
        logger.error(f"Error saving YAML: {e}")

    visualize_inference(y_hat, target_inds, acc_val)
