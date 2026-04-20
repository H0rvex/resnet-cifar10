"""Entry point: train ResNet on CIFAR-10 and save the best checkpoint."""

import argparse

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset import get_dataloaders
from model import ResNet
from trainer import evaluate, train_epoch
from utils.seeding import make_generator, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    p.add_argument("--seed", type=int, default=None, help="Random seed (default: Config.seed = 42)")
    return p.parse_args()


def main(cfg: Config) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = make_generator(cfg.seed)
    train_loader, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers, generator)

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
    args = parse_args()
    cfg = Config()
    if args.seed is not None:
        cfg.seed = args.seed
    main(cfg)
