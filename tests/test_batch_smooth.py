"""
Unit tests for src.aux.batch_smooth with a batch of one spectrum, the shape
PhysicsModel.baselines() passes it when forward() simulates a single spectrum.
"""
import torch

from src.aux import batch_smooth


def test_a_single_spectrum_is_smoothed():
    x = torch.randn(1, 1, 2048)
    out = batch_smooth(x, torch.full((1, 1, 1), 0.2), "constant")
    assert out.shape == (1, 1, 2048)


def test_a_spectrum_is_smoothed_the_same_alone_or_in_a_batch():
    torch.manual_seed(0)
    x = torch.randn(3, 1, 2048)
    windows = torch.full((3, 1, 1), 0.2)
    batch = batch_smooth(x, windows, "constant")
    alone = batch_smooth(x[1:2], windows[1:2], "constant")
    torch.testing.assert_close(alone, batch[1:2])
