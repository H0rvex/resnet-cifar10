from dataclasses import dataclass


@dataclass
class Config:
    # Data
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4

    # Model (He et al. CIFAR: depth = 6n + 2, e.g. 20, 32, 44, 56, 110)
    model_depth: int = 20
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

    # Mixed precision (fp16 autocast + GradScaler; CUDA only — disabled on CPU CI)
    use_amp: bool = True

    # Reproducibility
    seed: int = 42

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.epochs <= self.warmup_epochs:
            raise ValueError(
                f"epochs ({self.epochs}) must be greater than warmup_epochs ({self.warmup_epochs})"
            )
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.momentum < 0 or self.momentum > 1:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.label_smoothing < 0 or self.label_smoothing >= 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.warmup_start_factor <= 0:
            raise ValueError(
                f"warmup_start_factor must be positive, got {self.warmup_start_factor}"
            )
        if (self.model_depth - 2) % 6 != 0:
            raise ValueError(
                f"model_depth must satisfy depth = 6n + 2 (e.g. 20, 32), got {self.model_depth}"
            )
        if (self.model_depth - 2) // 6 < 1:
            raise ValueError(f"model_depth too small for a valid ResNet stack: {self.model_depth}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
