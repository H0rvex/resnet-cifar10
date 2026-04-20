import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> tuple[float, float]:
    """Run one training epoch. Returns (mean_loss, imgs_per_sec)."""
    model.train()
    total_loss = 0.0
    t_start: float | None = None
    timed_samples = 0
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()
        ):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        # Start clock after first batch so CUDA kernel launch overhead is excluded.
        if i == 0:
            t_start = time.perf_counter()
        else:
            timed_samples += images.size(0)
    elapsed = (time.perf_counter() - t_start) if t_start is not None else 1.0
    imgs_per_sec = timed_samples / elapsed if timed_samples > 0 else 0.0
    return total_loss / len(loader), imgs_per_sec


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    """Return (top-1 accuracy 0–100, mean loss) on the given loader."""
    model.eval()
    total_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total * 100, total_loss / len(loader)
