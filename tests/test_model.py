import torch

from model import ResNet, ResidualBlock

_EXPECTED_PARAMS = 175_258


def test_output_shape():
    model = ResNet()
    model.eval()
    x = torch.randn(4, 3, 32, 32)
    assert model(x).shape == (4, 10)


def test_output_shape_batch_1():
    model = ResNet()
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    assert model(x).shape == (1, 10)


def test_param_count():
    model = ResNet()
    params = sum(p.numel() for p in model.parameters())
    assert abs(params - _EXPECTED_PARAMS) / _EXPECTED_PARAMS < 0.05, (
        f"Param count {params} deviates >5% from expected {_EXPECTED_PARAMS}"
    )


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
