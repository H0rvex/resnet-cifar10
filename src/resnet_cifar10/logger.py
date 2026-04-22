"""Three-sink logger: stdout, JSONL, and TensorBoard."""

import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir: str, *, append_metrics: bool = False) -> None:
        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        mode = "a" if append_metrics else "w"
        self._jsonl = open(root / "metrics.jsonl", mode, encoding="utf-8")
        self._tb = SummaryWriter(log_dir=str(root / "tb"))

    def log(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        test_loss: float,
        test_acc: float,
        lr: float,
        imgs_per_sec: float,
    ) -> None:
        print(
            f"Epoch {epoch:>3}/{total_epochs}"
            f"  lr={lr:.4f}"
            f"  loss={train_loss:.4f}"
            f"  test_loss={test_loss:.4f}"
            f"  acc={test_acc:.2f}%"
            f"  {imgs_per_sec:.0f} img/s"
        )

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 4),
            "lr": lr,
            "imgs_per_sec": round(imgs_per_sec, 1),
        }
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        self._tb.add_scalar("loss/train", train_loss, epoch)
        self._tb.add_scalar("loss/test", test_loss, epoch)
        self._tb.add_scalar("acc/test", test_acc, epoch)
        self._tb.add_scalar("lr", lr, epoch)
        self._tb.add_scalar("throughput/imgs_per_sec", imgs_per_sec, epoch)

    def close(self) -> None:
        self._jsonl.close()
        self._tb.close()
