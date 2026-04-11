# ResNet for CIFAR-10

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

## Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| LR schedule | StepLR ×0.1 every 15 epochs |
| Epochs | 30 |
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
