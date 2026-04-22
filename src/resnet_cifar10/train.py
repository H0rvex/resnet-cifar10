"""Training orchestration: config resolution, checkpoints, and the main epoch loop."""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import time
import typing
from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR

from resnet_cifar10.config import Config
from resnet_cifar10.dataset import get_dataloaders
from resnet_cifar10.logger import Logger
from resnet_cifar10.model import infer_model_depth_from_state_dict, make_resnet_cifar
from resnet_cifar10.provenance import collect_run_provenance, write_run_info
from resnet_cifar10.trainer import evaluate, train_epoch
from resnet_cifar10.utils.seeding import make_generator, set_seed


@dataclass
class TrainResult:
    """Outcome of a completed ``train()`` call."""

    run_dir: str
    best_acc: float
    best_checkpoint: str
    last_checkpoint: str
    wall_time_sec: float


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
        help=(
            "Resume from a checkpoint; continues in that run's directory "
            "(appends metrics.jsonl, overwrites best/last.pth)"
        ),
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


def build_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: Config) -> SequentialLR:
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
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )


def setup_run(cfg: Config) -> tuple[str, str, str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", ts)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
        yaml.dump(dataclasses.asdict(cfg), fh, default_flow_style=False, sort_keys=False)
    return run_dir, os.path.join(run_dir, "best.pth"), os.path.join(run_dir, "last.pth")


def prepare_run_paths(cfg: Config, resume: str | None) -> tuple[str, str, str, bool]:
    """Return ``(run_dir, best_path, last_path, is_new_run)``.

    When ``resume`` is set, training continues in the checkpoint's directory (same
    ``metrics.jsonl`` / TensorBoard folder); no new timestamped run is created.
    """
    if resume is None:
        run_dir, best_path, last_path = setup_run(cfg)
        return run_dir, best_path, last_path, True
    run_dir = os.path.dirname(os.path.abspath(resume))
    if not os.path.isdir(run_dir):
        raise ValueError(f"Resume path directory does not exist: {run_dir}")
    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    return run_dir, best_path, last_path, False


def validate_checkpoint_against_config(ckpt: dict, cfg: Config) -> None:
    """Ensure architecture fields in ``cfg`` match the checkpoint before loading state."""
    if "model" not in ckpt:
        raise ValueError("Checkpoint is missing required key 'model'")
    sd = ckpt["model"]
    raw = ckpt.get("config") or {}
    inferred_depth = infer_model_depth_from_state_dict(sd)
    if "model_depth" in raw and int(raw["model_depth"]) != inferred_depth:
        raise ValueError(
            f"Checkpoint config model_depth={raw['model_depth']} does not match "
            f"weights (inferred depth {inferred_depth})"
        )
    ckpt_depth = inferred_depth
    if cfg.model_depth != ckpt_depth:
        raise ValueError(
            f"model_depth mismatch: current config has {cfg.model_depth}, "
            f"checkpoint weights imply depth {ckpt_depth}"
        )
    fc_w = sd.get("fc.weight")
    if fc_w is None:
        raise ValueError("Checkpoint model state_dict is missing 'fc.weight'")
    ckpt_classes = int(fc_w.shape[0])
    if cfg.num_classes != ckpt_classes:
        raise ValueError(
            f"num_classes mismatch: current config has {cfg.num_classes}, "
            f"checkpoint classifier has {ckpt_classes}"
        )


def restore_rng_from_checkpoint(ckpt: dict, device: torch.device) -> None:
    """Restore PyTorch (and CUDA if applicable) RNG state saved in a training checkpoint."""
    rs = ckpt.get("torch_rng_state")
    if rs is not None:
        torch.set_rng_state(rs)
    cuda_rs = ckpt.get("cuda_rng_state")
    if cuda_rs is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rs)
    elif cuda_rs is None and device.type == "cuda" and torch.cuda.is_available():
        print("Warning: checkpoint has no CUDA RNG state; CUDA randomness may diverge.")


def _rng_checkpoint_payload() -> dict[str, typing.Any]:
    out: dict[str, typing.Any] = {"torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        out["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return out


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    best_acc: float,
    cfg: Config,
    provenance: dict[str, typing.Any],
) -> None:
    payload: dict[str, typing.Any] = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_acc": best_acc,
        "config": dataclasses.asdict(cfg),
        "provenance": provenance,
    }
    payload.update(_rng_checkpoint_payload())
    torch.save(payload, path)


def train(cfg: Config, resume: str | None = None) -> TrainResult:
    """Train ``make_resnet_cifar(cfg.model_depth)`` on CIFAR-10; return paths and best accuracy."""
    t_wall0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, best_path, last_path, is_new_run = prepare_run_paths(cfg, resume)
    print(f"Run dir: {run_dir}")
    if not is_new_run:
        print("Resuming in existing run directory (metrics append, RNG restored from checkpoint).")

    set_seed(cfg.seed)
    print(f"Using device: {device}")

    provenance = collect_run_provenance(device)
    if is_new_run:
        write_run_info(os.path.join(run_dir, "run_info.json"), dataclasses.asdict(cfg), provenance)

    generator = make_generator(cfg.seed)
    train_loader, test_loader = get_dataloaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers, generator
    )

    model = make_resnet_cifar(cfg.model_depth, cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    scheduler = build_lr_scheduler(optimizer, cfg)

    amp_enabled = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    start_epoch = 1
    best_acc = 0.0
    if resume is not None:
        # Full checkpoint contains arbitrary Python objects in optimizer state; safe=True would reject.
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        validate_checkpoint_against_config(ckpt, cfg)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_acc = float(ckpt["best_acc"])
        start_epoch = int(ckpt["epoch"]) + 1
        restore_rng_from_checkpoint(ckpt, device)
        print(f"Resumed from epoch {ckpt['epoch']} (best acc {best_acc:.2f}%)")

    logger = Logger(run_dir, append_metrics=not is_new_run)
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
                save_checkpoint(
                    best_path, epoch, model, optimizer, scheduler, best_acc, cfg, provenance
                )
            save_checkpoint(
                last_path, epoch, model, optimizer, scheduler, best_acc, cfg, provenance
            )
    finally:
        logger.close()

    wall_time_sec = time.perf_counter() - t_wall0
    print(f"\nBest accuracy: {best_acc:.2f}%  (saved to {best_path})")

    return TrainResult(
        run_dir=run_dir,
        best_acc=best_acc,
        best_checkpoint=best_path,
        last_checkpoint=last_path,
        wall_time_sec=wall_time_sec,
    )
