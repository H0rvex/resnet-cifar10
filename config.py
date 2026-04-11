from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0  # 0 is required on Windows

    # Model
    num_classes: int = 10

    # Training
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_step_size: int = 15
    lr_gamma: float = 0.1

    # Checkpointing
    checkpoint_path: str = "checkpoint.pth"
