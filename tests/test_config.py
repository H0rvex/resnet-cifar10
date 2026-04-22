import tempfile
from pathlib import Path

import pytest
import yaml

from resnet_cifar10.config import Config
from resnet_cifar10.train import build_lr_scheduler, build_parser, resolve_config


def test_config_rejects_epochs_not_greater_than_warmup():
    with pytest.raises(ValueError, match="epochs"):
        Config(epochs=5, warmup_epochs=5)


def test_config_rejects_invalid_depth():
    with pytest.raises(ValueError, match="model_depth"):
        Config(model_depth=19)


def test_resolve_config_yaml_then_cli_override():
    defaults = Config()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.dump({"batch_size": 64, "lr": 0.05}, fh)
        path = fh.name
    try:
        p = build_parser()
        args = p.parse_args(["--config", path, "--seed", "7"])
        cfg = resolve_config(args)
        assert cfg.batch_size == 64
        assert cfg.lr == 0.05
        assert cfg.seed == 7
        assert cfg.epochs == defaults.epochs
    finally:
        Path(path).unlink(missing_ok=True)


def test_build_lr_scheduler_cosine_t_max():
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import SequentialLR

    cfg = Config(epochs=200, warmup_epochs=5, lr=0.1)
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    sched = build_lr_scheduler(opt, cfg)
    assert isinstance(sched, SequentialLR)
    cosine = sched._schedulers[1]
    assert cosine.T_max == cfg.epochs - cfg.warmup_epochs


def test_resolve_config_cli_overrides_yaml():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.dump({"epochs": 100, "model_depth": 32, "warmup_epochs": 1}, fh)
        path = fh.name
    try:
        args = build_parser().parse_args(["--config", path, "--epochs", "50"])
        cfg = resolve_config(args)
        assert cfg.epochs == 50
        assert cfg.model_depth == 32
    finally:
        Path(path).unlink(missing_ok=True)
