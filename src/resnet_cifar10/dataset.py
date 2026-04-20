import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from resnet_cifar10.utils.seeding import worker_init_fn

_MEAN = (0.4914, 0.4822, 0.4465)
_STD = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 (if needed) and return train/test DataLoaders.

    Pass a seeded generator (from utils.seeding.make_generator) to make
    shuffle order and worker augmentation deterministic across runs.
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
