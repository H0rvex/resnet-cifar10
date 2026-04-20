from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 0  # 0 is required on Windows

    # Model
    num_classes: int = 10

    # Optimization
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    label_smoothing: float = 0.1

    # LR schedule: linear warmup then cosine annealing to zero
    warmup_epochs: int = 5
    warmup_start_factor: float = 0.1

    # Mixed precision (fp16 autocast + GradScaler; CUDA only)
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    # Checkpointing
    checkpoint_path: str = "checkpoint.pth"
