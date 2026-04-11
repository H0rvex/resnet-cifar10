"""Entry point: train ResNet on CIFAR-10 and save the best checkpoint."""

import torch
import torch.nn as nn

from config import Config
from dataset import get_dataloaders
from model import ResNet
from trainer import evaluate, train_epoch


def main(cfg: Config) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.num_workers)

    model = ResNet(num_classes=cfg.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)

    best_acc = 0.0
    for epoch in range(1, cfg.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:>2}/{cfg.epochs}  loss={loss:.4f}  acc={acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), cfg.checkpoint_path)

    print(f"\nBest accuracy: {best_acc:.1f}%  (saved to {cfg.checkpoint_path})")


if __name__ == "__main__":
    main(Config())
