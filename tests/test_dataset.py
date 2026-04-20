import os

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from resnet_cifar10.dataset import _MEAN, _STD, eval_transform, get_dataloaders

_CIFAR10_DIR = os.path.join(".", "data", "cifar-10-batches-py")


def _fake_loader(n: int = 64, batch_size: int = 32) -> DataLoader:
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_loader_image_shape_and_dtype():
    loader = _fake_loader()
    images, labels = next(iter(loader))
    assert images.shape == (32, 3, 32, 32)
    assert images.dtype == torch.float32


def test_loader_label_shape_and_dtype():
    loader = _fake_loader()
    _, labels = next(iter(loader))
    assert labels.shape == (32,)
    assert labels.dtype == torch.int64


def test_eval_transform_normalises_to_near_zero():
    """An image filled with CIFAR-10 mean pixel values should normalise close to 0."""
    mean_u8 = (np.array(_MEAN) * 255).astype(np.uint8)
    img_array = np.broadcast_to(mean_u8, (32, 32, 3)).copy()
    tensor = eval_transform(Image.fromarray(img_array))
    assert tensor.shape == (3, 32, 32)
    assert abs(tensor.mean().item()) < 0.05


def test_eval_transform_output_not_in_unit_range():
    """After normalisation, values should extend outside [0, 1]."""
    rng = np.random.default_rng(0)
    img_array = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    tensor = eval_transform(Image.fromarray(img_array))
    assert tensor.min().item() < 0.0 or tensor.max().item() > 1.0


@pytest.mark.skipif(
    not os.path.isdir(_CIFAR10_DIR),
    reason="CIFAR-10 data not present; run train.py once to download",
)
def test_get_dataloaders_test_split_shape():
    _, test_loader = get_dataloaders("./data", batch_size=64, num_workers=0)
    images, labels = next(iter(test_loader))
    assert images.shape == (64, 3, 32, 32)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64


@pytest.mark.skipif(
    not os.path.isdir(_CIFAR10_DIR),
    reason="CIFAR-10 data not present; run train.py once to download",
)
def test_get_dataloaders_test_split_normalised():
    """Real CIFAR-10 test batch (eval_transform, no padding) should have mean near 0.

    The train loader uses RandomCrop with zero-padding which pushes the batch
    mean to ~-0.3; the test loader uses only ToTensor+Normalize, so a large
    batch should be centred within ±0.2 of zero.
    """
    _, test_loader = get_dataloaders("./data", batch_size=512, num_workers=0)
    images, _ = next(iter(test_loader))
    assert abs(images.mean().item()) < 0.2
