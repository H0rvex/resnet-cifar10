import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from resnet_cifar10.model import make_resnet_cifar
from resnet_cifar10.trainer import evaluate, train_epoch
from resnet_cifar10.utils.seeding import set_seed

_N = 256
_BATCH = 256  # single batch so loss-decrease is reliable
_LR = 0.1


def _make_loader(seed: int = 0) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn(_N, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (_N,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=_BATCH)


def _build(seed: int = 42) -> tuple:
    set_seed(seed)
    device = torch.device("cpu")
    model = make_resnet_cifar(20).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=_LR)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, optimizer, loss_fn, scaler, device


def test_loss_decreases():
    """Two epochs on a fixed single-batch dataset: loss must fall."""
    model, optimizer, loss_fn, scaler, device = _build()
    loader = _make_loader()
    loss1, _ = train_epoch(model, loader, optimizer, loss_fn, device, scaler)
    loss2, _ = train_epoch(model, loader, optimizer, loss_fn, device, scaler)
    assert loss2 < loss1, f"Loss did not decrease: {loss1:.4f} → {loss2:.4f}"


def test_initial_loss_near_log10():
    """Random-init cross-entropy loss should be close to log(10) ≈ 2.303."""
    model, _, loss_fn, _, device = _build()
    model.eval()
    loader = _make_loader()
    with torch.no_grad():
        images, labels = next(iter(loader))
        loss = loss_fn(model(images.to(device)), labels.to(device)).item()
    assert abs(loss - 2.303) < 0.5, f"Unexpected initial loss: {loss:.4f}"


def test_evaluate_returns_acc_and_loss():
    model, _, loss_fn, _, device = _build()
    loader = _make_loader()
    acc, loss = evaluate(model, loader, device, loss_fn)
    assert 0.0 <= acc <= 100.0
    assert loss > 0.0


def test_determinism():
    """Identical seeds must produce bit-exact loss across two independent runs."""

    def one_run() -> float:
        model, optimizer, loss_fn, scaler, device = _build(seed=42)
        loader = _make_loader(seed=7)
        loss, _ = train_epoch(model, loader, optimizer, loss_fn, device, scaler)
        return loss

    assert one_run() == one_run(), "Training is not deterministic under the same seed"
