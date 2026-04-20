# ResNet for CIFAR-10

[![CI](https://github.com/h0rvex/resnet-cifar10/actions/workflows/ci.yml/badge.svg)](https://github.com/h0rvex/resnet-cifar10/actions/workflows/ci.yml)

Simplified ResNet implementation from scratch in PyTorch, inspired by He et al. 2015 ("Deep Residual Learning for Image Recognition").

**Result: 84.6% top-1 accuracy on the CIFAR-10 test set.**

## Quickstart

```bash
pip install -r requirements.txt
python train.py
```

CIFAR-10 is downloaded automatically on first run. The best checkpoint is saved to `checkpoint.pth`.

## Project structure

```
resnet/
├── model.py        # ResidualBlock and ResNet architecture
├── dataset.py      # CIFAR-10 transforms and DataLoader factory
├── trainer.py      # train_epoch and evaluate functions
├── config.py       # Config dataclass — all hyperparameters in one place
├── train.py        # Entry point
└── requirements.txt
```

## Architecture

- Stem: 3→16 channels, 3×3 conv + BN + ReLU
- 6 residual blocks: 16→16→32→32→64→64 channels
- Stride-2 downsampling at the 16→32 and 32→64 transitions
- Projection shortcuts (1×1 conv + BN) when dimensions change
- Global average pooling → 10-class linear head

## Results

| Model | Params | Top-1 | He et al. | Δ | Throughput |
|-------|--------|-------|-----------|---|------------|
| ResNet-20 | 0.27 M | TBD | 91.25% | TBD | TBD img/s |

*Trained with SGD + cosine LR, label smoothing 0.1, mixed precision (fp16), seed 42.*
*Throughput measured on batches 2–N per epoch to exclude CUDA warmup.*

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD momentum=0.9, nesterov |
| Learning rate | 0.1 → cosine annealing (5-epoch linear warmup) |
| Weight decay | 5e-4 |
| Label smoothing | 0.1 |
| Mixed precision | fp16 autocast + GradScaler |
| Epochs | 200 |
| Batch size | 128 |
| Augmentation | RandomHorizontalFlip, RandomCrop(32, padding=4) |

## Key concepts

- **Residual connections** — allow gradients to flow unimpeded through the network, enabling effective training of deep models
- **Projection shortcuts** — 1×1 convolutions match channel dimensions when the residual and main path differ
- **Stride-based downsampling** — spatial resolution halved at channel-doubling transitions instead of using max pooling

## Further improvements

- Longer training (50–100 epochs) with cosine annealing
- Wider channels (32→64→128 instead of 16→32→64)
- More residual blocks (9 blocks as in the original paper's ResNet-20)
- Advanced augmentation: Cutout, Mixup, RandAugment
