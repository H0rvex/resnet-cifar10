import re

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-layer residual block with a projection shortcut when dimensions change."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = None
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            residual = self.shortcut(x)
        return torch.relu(out + residual)


def blocks_per_stage_from_depth(depth: int) -> int:
    """He et al. (2015) CIFAR ResNets: depth = 6n + 2 with n blocks per stage."""
    if (depth - 2) % 6 != 0:
        raise ValueError(
            f"Invalid depth {depth}: expected depth = 6n + 2 (e.g. 20, 32, 44, 56, 110)"
        )
    n = (depth - 2) // 6
    if n < 1:
        raise ValueError(f"Invalid depth {depth}: need n >= 1 blocks per stage")
    return n


def infer_model_depth_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    """Recover depth from weights when older checkpoints omit ``model_depth``."""
    block_indices: set[int] = set()
    pat = re.compile(r"^blocks\.(\d+)\.")
    for key in state_dict:
        m = pat.match(key)
        if m:
            block_indices.add(int(m.group(1)))
    if not block_indices:
        return 20
    n_blocks = max(block_indices) + 1
    if n_blocks % 3 != 0:
        raise ValueError(
            f"Cannot infer depth: found {n_blocks} residual blocks (expected multiple of 3)"
        )
    n = n_blocks // 3
    return 6 * n + 2


class ResNet(nn.Module):
    """CIFAR-style ResNet (He et al. §4.2): stem → 3×n residual blocks → GAP → FC.

    Each stage has ``blocks_per_stage`` basic blocks; strides apply at the first block
    of stage 2 and stage 3. Total weighted depth is ``6 * blocks_per_stage + 2``.
    """

    def __init__(self, num_classes: int = 10, *, blocks_per_stage: int):
        super().__init__()
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be >= 1")
        self.blocks_per_stage = blocks_per_stage
        self.depth = 6 * blocks_per_stage + 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        n = blocks_per_stage
        layers: list[nn.Module] = []
        # Stage 1: n blocks at 32×32, 16 channels
        for _ in range(n):
            layers.append(ResidualBlock(16, 16, stride=1))
        # Stage 2: stride-2 downsample 16→32, then n−1 blocks at 16×16, 32 channels
        layers.append(ResidualBlock(16, 32, stride=2))
        for _ in range(n - 1):
            layers.append(ResidualBlock(32, 32, stride=1))
        # Stage 3: 32→64, then n−1 blocks at 8×8, 64 channels
        layers.append(ResidualBlock(32, 64, stride=2))
        for _ in range(n - 1):
            layers.append(ResidualBlock(64, 64, stride=1))
        self.blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def make_resnet_cifar(depth: int, num_classes: int = 10) -> ResNet:
    """Build a CIFAR ResNet with ``depth = 6n + 2`` (e.g. 20, 32, 44, 56, 110)."""
    n = blocks_per_stage_from_depth(depth)
    return ResNet(num_classes=num_classes, blocks_per_stage=n)
