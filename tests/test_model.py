import pytest
import torch

from resnet_cifar10.model import (
    ResidualBlock,
    infer_model_depth_from_state_dict,
    make_resnet_cifar,
)

_EXPECTED_PARAMS_DEPTH20 = 272_474
_EXPECTED_PARAMS_DEPTH14 = 175_258


@pytest.mark.parametrize(
    "depth,expected", [(20, _EXPECTED_PARAMS_DEPTH20), (14, _EXPECTED_PARAMS_DEPTH14)]
)
def test_output_shape(depth: int, expected: int):
    model = make_resnet_cifar(depth)
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)
    params = sum(p.numel() for p in model.parameters())
    assert params == expected


def test_output_shape_batch_1():
    model = make_resnet_cifar(20)
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    assert model(x).shape == (1, 10)


def test_param_count_resnet20():
    model = make_resnet_cifar(20)
    params = sum(p.numel() for p in model.parameters())
    assert params == _EXPECTED_PARAMS_DEPTH20


@pytest.mark.parametrize("depth", [20, 32, 44])
def test_depth_formula_matches_block_count(depth: int):
    model = make_resnet_cifar(depth)
    n = (depth - 2) // 6
    assert len(model.blocks) == 3 * n


def test_projection_shortcut_on_stride():
    block = ResidualBlock(16, 16, stride=2)
    assert block.shortcut is not None


def test_projection_shortcut_on_channel_change():
    block = ResidualBlock(16, 32, stride=1)
    assert block.shortcut is not None


def test_no_shortcut_when_dims_match():
    block = ResidualBlock(16, 16, stride=1)
    assert block.shortcut is None


def test_residual_block_output_shape_no_shortcut():
    block = ResidualBlock(16, 16)
    x = torch.randn(2, 16, 32, 32)
    assert block(x).shape == (2, 16, 32, 32)


def test_residual_block_output_shape_with_shortcut():
    block = ResidualBlock(16, 32, stride=2)
    x = torch.randn(2, 16, 32, 32)
    assert block(x).shape == (2, 32, 16, 16)


def test_infer_model_depth_from_state_dict():
    sd = make_resnet_cifar(14).state_dict()
    assert infer_model_depth_from_state_dict(sd) == 14
    sd20 = make_resnet_cifar(20).state_dict()
    assert infer_model_depth_from_state_dict(sd20) == 20


def test_golden_forward_resnet20():
    """Guards silent architecture drift for fixed init + input (CPU, seed 0)."""
    from resnet_cifar10.utils.seeding import set_seed

    set_seed(0)
    model = make_resnet_cifar(20)
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    assert y.shape == (2, 10)
    assert y.sum().item() == pytest.approx(0.24738942086696625, rel=0.0, abs=1e-5)
