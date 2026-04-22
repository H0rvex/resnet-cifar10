import dataclasses
import os
import tempfile

import pytest
import torch

from resnet_cifar10.config import Config
from resnet_cifar10.model import make_resnet_cifar
from resnet_cifar10.train import (
    build_lr_scheduler,
    prepare_run_paths,
    restore_rng_from_checkpoint,
    save_checkpoint,
    validate_checkpoint_against_config,
)


def test_checkpoint_roundtrip_model_optimizer():
    cfg = Config(epochs=6, warmup_epochs=1, model_depth=20)
    device = torch.device("cpu")
    model = make_resnet_cifar(cfg.model_depth, cfg.num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    scheduler = build_lr_scheduler(optimizer, cfg)
    provenance = {"torch_version": torch.__version__, "git_commit": None}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pth")
        save_checkpoint(path, 3, model, optimizer, scheduler, 88.5, cfg, provenance)

        # Full training dict is not loadable with weights_only=True.
        ckpt = torch.load(path, map_location=device, weights_only=False)

        model2 = make_resnet_cifar(cfg.model_depth, cfg.num_classes).to(device)
        model2.load_state_dict(ckpt["model"])
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2)

        assert ckpt["epoch"] == 3
        assert ckpt["best_acc"] == 88.5
        assert ckpt["config"]["model_depth"] == 20
        assert ckpt["provenance"]["torch_version"] == torch.__version__

        optimizer2 = torch.optim.SGD(model2.parameters(), lr=cfg.lr)
        optimizer2.load_state_dict(ckpt["optimizer"])
        assert len(optimizer2.state_dict()["state"]) == len(optimizer.state_dict()["state"])


def test_validate_checkpoint_rejects_missing_model():
    with pytest.raises(ValueError, match="missing required key"):
        validate_checkpoint_against_config({}, Config())


def test_validate_checkpoint_rejects_depth_mismatch():
    cfg = Config(model_depth=20)
    sd = make_resnet_cifar(14).state_dict()
    ckpt = {"model": sd, "config": {}}
    with pytest.raises(ValueError, match="model_depth mismatch"):
        validate_checkpoint_against_config(ckpt, cfg)


def test_validate_checkpoint_rejects_config_depth_disagreeing_with_weights():
    sd = make_resnet_cifar(14).state_dict()
    ckpt = {"model": sd, "config": {"model_depth": 20}}
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_against_config(ckpt, Config(model_depth=14))


def test_validate_checkpoint_rejects_num_classes_mismatch():
    cfg = Config(num_classes=10, model_depth=20)
    m = make_resnet_cifar(20)
    sd = dict(m.state_dict())
    sd["fc.weight"] = torch.randn(100, 64)
    ckpt = {"model": sd, "config": {"model_depth": 20}}
    with pytest.raises(ValueError, match="num_classes mismatch"):
        validate_checkpoint_against_config(ckpt, cfg)


def test_validate_checkpoint_accepts_matching():
    cfg = Config(model_depth=20, num_classes=10)
    m = make_resnet_cifar(20)
    ckpt = {"model": m.state_dict(), "config": dataclasses.asdict(cfg)}
    validate_checkpoint_against_config(ckpt, cfg)


def test_prepare_run_paths_resume_uses_checkpoint_parent():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "last.pth")
        open(ckpt_path, "w").close()
        run_dir, best, last, is_new = prepare_run_paths(Config(), ckpt_path)
        assert is_new is False
        assert run_dir == tmp
        assert best == os.path.join(tmp, "best.pth")
        assert last == os.path.join(tmp, "last.pth")


def test_prepare_run_paths_resume_requires_existing_parent_dir():
    missing = os.path.join(tempfile.gettempdir(), "nonexistent_run_xyz", "last.pth")
    with pytest.raises(ValueError, match="directory does not exist"):
        prepare_run_paths(Config(), missing)


def test_restore_rng_from_checkpoint_cpu():
    device = torch.device("cpu")
    torch.manual_seed(123)
    _ = torch.rand(40)
    saved = torch.get_rng_state()
    expected_next = torch.rand(5)
    torch.manual_seed(0)
    _ = torch.rand(200)
    restore_rng_from_checkpoint({"torch_rng_state": saved}, device)
    actual_next = torch.rand(5)
    assert torch.equal(expected_next, actual_next)
