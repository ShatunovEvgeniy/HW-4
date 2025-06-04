from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

config = dict(
    n_way=1,
    n_support=5,
    n_query=5,
    max_epoch=5,
    epoch_size=2000,
    lr=0.001,
    x_dim=(3, 28, 28),
    hid_dim=64,
    z_dim=64,
    augment_flag=True,
    use_simclr=True,
    simclr_path=PROJECT_ROOT / "model" / "SimCLR" / "checkpoints" / "best.pt",
)
