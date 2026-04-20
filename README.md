# ResNet-CIFAR10

[![CI](https://github.com/h0rvex/resnet-cifar10/actions/workflows/ci.yml/badge.svg)](https://github.com/h0rvex/resnet-cifar10/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

ResNet-20 reimplemented from scratch in PyTorch, reproducing the [He et al. (2015)](https://arxiv.org/abs/1512.03385) CIFAR-10 result (**91.67% top-1**) on a single consumer GPU.

## Results

| Model     | Params  | FLOPs  | Top-1      | He et al. | Δ     | Throughput         |
| --------- | ------- | ------ | ---------- | --------- | ----- | ------------------ |
| ResNet-20 | 0.175 M | 54.4 M | **91.67%** | 91.25%    | +0.42 | 5170 ± 45 imgs/sec |

*SGD + momentum 0.9 + Nesterov, weight decay 5e-4, linear warmup 5 epochs → cosine anneal to 0 over 200 epochs, label smoothing 0.1, fp16 autocast. Seed 42. Throughput excludes first-batch warmup.*

![Training curves](artifacts/training_curves.png)
![Confusion matrix](artifacts/confusion_matrix.png)

## Reproduce this result

```bash
pip install -e .
python scripts/train.py --config configs/resnet20.yaml --seed 42
python scripts/evaluate.py --checkpoint runs/<timestamp>/best.pth
```

Results reproduce bitwise on the same GPU architecture with `--seed 42`.

## Hardware

| GPU      | VRAM  | Epoch time | Total wall-clock |
| -------- | ----- | ---------- | ---------------- |
| Tesla T4 | 16 GB | 11.4 s     | ~38.0 min        |

## Architecture

Faithful to the CIFAR-10 variant described in [He et al. (2015), §4.2](https://arxiv.org/abs/1512.03385):

- Stem: 3→16 channels, 3×3 conv + BN + ReLU.
- Three stages of residual blocks at 16 / 32 / 64 channels; stride-2 downsampling at the 16→32 and 32→64 transitions.
- Projection shortcuts (1×1 conv + BN) when channel dimensions or spatial resolution change; identity shortcuts otherwise.
- Global average pooling → 10-way linear classifier.

## What this project demonstrates

- Careful reproduction of a paper baseline on fixed hardware with deterministic seeding and a pinned recipe.
- End-to-end training infrastructure kept intentionally small: typed config, JSONL + TensorBoard logging, resumable checkpoints, mixed precision, CI-gated tests.
- Honest reporting: headline number sits alongside the paper's number and the exact command that produced it.

## Limitations

- Single-seed result; no mean ± std across runs.
- No depth scan beyond ResNet-20 — ResNet-32/56 are left out to keep the "did the recipe work" signal clean.
- No robustness evaluation (adversarial, OOD, calibration); deliberately out of scope.

## References

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
