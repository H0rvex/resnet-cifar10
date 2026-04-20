import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs and configure cuDNN/PyTorch for deterministic execution.

    deterministic=True enables torch.use_deterministic_algorithms (warn_only),
    which catches non-deterministic ops without aborting the training run.
    Set deterministic=False to trade reproducibility for speed on unsupported ops.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded CPU Generator for DataLoader shuffle reproducibility."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """Per-worker seed derived from the DataLoader's base Generator seed.

    PyTorch sets each worker's torch seed via the generator before this is
    called, so torch.initial_seed() is already unique per worker. We propagate
    that seed to Python random and NumPy so all sampling in transforms is
    consistent across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
