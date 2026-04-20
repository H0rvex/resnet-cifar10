"""Entry point: train ResNet on CIFAR-10 and save the best checkpoint."""

import argparse
import dataclasses
import datetime
import os
import typing

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import Config
from dataset import get_dataloaders
from model import ResNet
from trainer import evaluate, train_epoch
from utils.seeding import make_generator, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    p.add_argument(
        "--config", type=str, default=None, metavar="PATH",
        help="YAML config file; CLI flags override loaded values",
    )
    hints = typing.get_type_hints(Config)
    for f in dataclasses.fields(Config):
        flag = f"--{f.name.replace('_', '-')}"
        ftype = hints[f.name]
        if ftype is bool:
            p.add_argument(flag, default=None, action=argparse.BooleanOptionalAction)
        else:
            p.add_argument(flag, type=ftype, default=None, metavar=ftype.__name__.upper())
    return p


def resolve_config(args: argparse.Namespace) -> Config:
    cfg_dict: dict = {f.name: f.default for f in dataclasses.fields(Config)}

    if args.config is not None:
        with open(args.config) as fh:
            yaml_cfg = yaml.safe_load(fh) or {}
        cfg_dict.update(yaml_cfg)

    for f in dataclasses.fields(Config):
        val = getattr(args, f.name, None)
        if val is not None:
            cfg_dict[f.name] = val

    return Config(**cfg_dict)


def setup_run(cfg: Config) -> tuple[str, str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "best.pth")
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    return run_dir, checkpoint_path


def main(cfg: Config) -> None:
    run_dir, checkpoint_path = setup_run(cfg)
    print(f"Run dir: {run_dir}")

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
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nBest accuracy: {best_acc:.2f}%  (saved to {checkpoint_path})")


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = resolve_config(args)
    main(cfg)
