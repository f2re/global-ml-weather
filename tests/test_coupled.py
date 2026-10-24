import torch
from torch import nn

from global_weather.coupled import LatentToPressureBridge


def test_bridge_starts_as_noop_and_has_required_shape():
    bridge = LatentToPressureBridge(8, 4, heads=4, max_residual=.25)
    latent = torch.randn(3, 8, 8)
    background = torch.randn(3, 37, 4)
    result = bridge(latent, background)
    assert result.shape == background.shape
    assert torch.equal(result, background)
    result.sum().backward()
    assert bridge.delta[-2].weight.grad is not None


def test_bridge_rejects_nonfinite_state():
    bridge = LatentToPressureBridge(8, 4, heads=4)
    latent = torch.randn(2, 8, 8)
    latent[0, 0, 0] = float("nan")
    background = torch.randn(2, 37, 4)
    try:
        bridge(latent, background)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite analysis state must be rejected")
