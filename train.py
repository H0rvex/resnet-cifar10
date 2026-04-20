"""Entry point: train ResNet on CIFAR-10 and save the best checkpoint."""

import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset import get_dataloaders
from model import ResNet
from trainer import evaluate, train_epoch


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs and force deterministic cuDNN kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    model = ResNet(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    warmup = LinearLR(
        optimizer,
        start_factor=cfg.warmup_start_factor,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=0.0,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:>3}/{cfg.epochs}  lr={lr_now:.4f}  loss={loss:.4f}  acc={acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.checkpoint_path)

    print(f"\nBest accuracy: {best_acc:.2f}%  (saved to {cfg.checkpoint_path})")


if __name__ == "__main__":
    main(Config())
