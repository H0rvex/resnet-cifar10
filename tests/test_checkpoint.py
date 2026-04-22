import os
import tempfile

import torch

from resnet_cifar10.config import Config
from resnet_cifar10.model import make_resnet_cifar
from resnet_cifar10.train import build_lr_scheduler, save_checkpoint


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
