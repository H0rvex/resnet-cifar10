"""Generate training-curve plot from a metrics.jsonl file.

Usage:
    python scripts/plot_curves.py --metrics runs/<timestamp>/metrics.jsonl
    python scripts/plot_curves.py --metrics runs/<timestamp>/metrics.jsonl --out artifacts/training_curves.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training curves from metrics.jsonl")
    p.add_argument("--metrics", required=True, metavar="PATH", help="Path to metrics.jsonl")
    p.add_argument(
        "--out",
        default="artifacts/training_curves.png",
        metavar="PATH",
        help="Output PNG path (default: artifacts/training_curves.png)",
    )
    return p.parse_args()


def load_metrics(path: str) -> dict[str, list]:
    records: dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
    }
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            for key in records:
                records[key].append(row[key])
    return records


def plot(metrics: dict[str, list], out_path: str) -> None:
    epochs = metrics["epoch"]
    best_acc = max(metrics["test_acc"])
    best_epoch = metrics["epoch"][metrics["test_acc"].index(best_acc)]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_loss = "#4C72B0"
    color_acc = "#DD8452"

    ax1.plot(epochs, metrics["train_loss"], color=color_loss, linewidth=1.4, label="Train loss")
    ax1.plot(
        epochs,
        metrics["test_loss"],
        color=color_loss,
        linewidth=1.4,
        linestyle="--",
        alpha=0.7,
        label="Test loss",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color_loss)
    ax1.tick_params(axis="y", labelcolor=color_loss)

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        metrics["test_acc"],
        color=color_acc,
        linewidth=1.6,
        label=f"Test acc (best {best_acc:.2f}%)",
    )
    ax2.axvline(best_epoch, color=color_acc, linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_ylabel("Test accuracy (%)", color=color_acc)
    ax2.tick_params(axis="y", labelcolor=color_acc)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("ResNet-20 on CIFAR-10 — training curves")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  (best {best_acc:.2f}% @ epoch {best_epoch})")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    plot(metrics, args.out)


if __name__ == "__main__":
    main()
