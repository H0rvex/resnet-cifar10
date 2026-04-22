"""Evaluate a trained ResNet checkpoint on the CIFAR-10 test set.

Outputs
-------
artifacts/per_class.png       -- per-class accuracy bar chart
artifacts/confusion_matrix.png -- row-normalised confusion matrix heatmap
artifacts/eval.json           -- all metrics as JSON
"""

import argparse
import dataclasses
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from thop import profile as thop_profile

from resnet_cifar10.config import Config
from resnet_cifar10.dataset import get_dataloaders
from resnet_cifar10.model import infer_model_depth_from_state_dict, make_resnet_cifar

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ResNet checkpoint on CIFAR-10")
    p.add_argument(
        "--checkpoint", required=True, metavar="PATH", help="Path to best.pth saved by train.py"
    )
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out-dir", default="artifacts", help="Directory to write PNG and JSON outputs")
    return p.parse_args()


def _config_from_checkpoint(ckpt: dict, state_dict: dict[str, torch.Tensor]) -> Config:
    raw = ckpt.get("config") or {}
    base = {f.name: f.default for f in dataclasses.fields(Config)}
    base.update(raw)
    if "model_depth" not in raw:
        base["model_depth"] = infer_model_depth_from_state_dict(state_dict)
    return Config(**base)


def load_model(path: str, device: torch.device) -> tuple[nn.Module, Config]:
    # weights_only=False: training checkpoints include config/optimizer dicts, not tensors-only.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    cfg = _config_from_checkpoint(ckpt, state_dict)
    model = make_resnet_cifar(cfg.model_depth, cfg.num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        all_preds.append(model(images).argmax(dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def count_params_and_macs(model: nn.Module, device: torch.device) -> tuple[float, float]:
    dummy = torch.zeros(1, 3, 32, 32, device=device)
    macs, params = thop_profile(model, inputs=(dummy,), verbose=False)
    return params / 1e6, macs / 1e6


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    model: nn.Module,
    device: torch.device,
    cfg: Config,
) -> dict:
    n = len(CLASSES)
    top1 = float((preds == labels).mean() * 100)

    per_class: dict[str, float] = {}
    for i, cls in enumerate(CLASSES):
        mask = labels == i
        per_class[cls] = float((preds[mask] == i).mean() * 100) if mask.any() else 0.0

    conf = np.zeros((n, n), dtype=int)
    for p, true_label in zip(preds, labels):
        conf[true_label, p] += 1

    params_m, macs_m = count_params_and_macs(model, device)

    return {
        "top1_acc": round(top1, 4),
        "per_class_acc": {k: round(v, 4) for k, v in per_class.items()},
        "confusion_matrix": conf.tolist(),
        "params_M": round(params_m, 4),
        "macs_M": round(macs_m, 2),
        "model_depth": cfg.model_depth,
    }


def plot_per_class(per_class: dict[str, float], out_path: str) -> None:
    classes = list(per_class.keys())
    accs = list(per_class.values())
    mean_acc = sum(accs) / len(accs)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(classes, accs, color="steelblue", width=0.6)
    ax.axhline(
        mean_acc, color="tomato", linestyle="--", linewidth=1.2, label=f"Mean {mean_acc:.1f}%"
    )
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class accuracy — CIFAR-10 test set")
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(conf: np.ndarray, out_path: str) -> None:
    row_sum = conf.sum(axis=1, keepdims=True)
    conf_pct = conf / np.where(row_sum == 0, 1, row_sum) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        conf_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=ax,
        vmin=0,
        vmax=100,
        linewidths=0.4,
        linecolor="lightgrey",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — row-normalised (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg = load_model(args.checkpoint, device)
    print(f"model_depth={cfg.model_depth}  num_classes={cfg.num_classes}")

    _, test_loader = get_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    print("Running inference…")
    preds, labels = run_inference(model, test_loader, device)

    metrics = compute_metrics(preds, labels, model, device, cfg)

    print(f"\nTop-1 accuracy : {metrics['top1_acc']:.2f}%")
    print(f"Params         : {metrics['params_M']:.3f} M")
    print(f"MACs           : {metrics['macs_M']:.1f} M")
    print("\nPer-class accuracy:")
    for cls, acc in metrics["per_class_acc"].items():
        print(f"  {cls:<12} {acc:.2f}%")

    per_class_path = os.path.join(args.out_dir, "per_class.png")
    conf_path = os.path.join(args.out_dir, "confusion_matrix.png")
    json_path = os.path.join(args.out_dir, "eval.json")

    print("\nSaving artifacts…")
    plot_per_class(metrics["per_class_acc"], per_class_path)
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), conf_path)
    with open(json_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"  {per_class_path}")
    print(f"  {conf_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
