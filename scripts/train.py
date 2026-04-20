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

from resnet_cifar10.config import Config
from resnet_cifar10.dataset import get_dataloaders
from resnet_cifar10.logger import Logger
from resnet_cifar10.model import ResNet
from resnet_cifar10.trainer import evaluate, train_epoch
from resnet_cifar10.utils.seeding import make_generator, set_seed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="YAML config file; CLI flags override loaded values",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from a checkpoint saved by this script",
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


def setup_run(cfg: Config) -> tuple[str, str, str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    return run_dir, os.path.join(run_dir, "best.pth"), os.path.join(run_dir, "last.pth")


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_acc: float,
    cfg: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
            "config": dataclasses.asdict(cfg),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def main(cfg: Config, resume: str | None = None) -> None:
    run_dir, best_path, last_path = setup_run(cfg)
    print(f"Run dir: {run_dir}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator = make_generator(cfg.seed)
    train_loader, test_loader = get_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers, generator
    )

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

    start_epoch = 1
    best_acc = 0.0
    if resume is not None:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best acc {best_acc:.2f}%)")

    logger = Logger(run_dir)
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_loss, imgs_per_sec = train_epoch(
                model, train_loader, optimizer, loss_fn, device, scaler
            )
            scheduler.step()
            test_acc, test_loss = evaluate(model, test_loader, device, loss_fn)
            lr_now = optimizer.param_groups[0]["lr"]
            logger.log(epoch, cfg.epochs, train_loss, test_loss, test_acc, lr_now, imgs_per_sec)

            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_acc, cfg)
            save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_acc, cfg)
    finally:
        logger.close()

    print(f"\nBest accuracy: {best_acc:.2f}%  (saved to {best_path})")


if __name__ == "__main__":
    args = build_parser().parse_args()
    cfg = resolve_config(args)
    main(cfg, resume=args.resume)
