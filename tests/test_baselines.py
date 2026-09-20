"""
Unit tests for src.baselines.bounded_random_walk (v2.0 handover section 13:
"baseline generation and spline fitting" -- this covers generation; spline
fitting itself is covered in tests/test_splines.py). Pure function, no
basis set needed.

Most tests use shape (batch, 1, 1), matching how PhysicsModel.baselines()/
residual_water() actually call this (see e.g. sample_resWater()'s
torch.zeros(N, 1, 1) construction in src/aux/aux.py). A previously-found
bug (a 2-D (batch, 1) start/end silently cross-broadcasting into
(batch, batch, length) instead of (batch, 1, length)) is now fixed --
see test_2d_start_end_no_longer_cross_broadcasts below and
docs/v2/progress_log.md.
"""
import torch

from src.baselines import bounded_random_walk


def test_output_shape_matches_start_batch_dims_and_requested_length():
    start = torch.zeros(4, 1, 1)
    end = torch.zeros(4, 1, 1)
    out = bounded_random_walk(start, end, std=0.1, length=64)
    assert out.shape == (4, 1, 64)


def test_stays_within_bounds():
    torch.manual_seed(0)
    start = torch.zeros(8, 1, 1)
    end = torch.zeros(8, 1, 1)
    out = bounded_random_walk(start, end, std=0.5, lower_bound=-1, upper_bound=1, length=256)
    assert (out >= -1 - 1e-5).all()
    assert (out <= 1 + 1e-5).all()


def test_starts_and_ends_at_requested_values():
    start = torch.tensor([[[0.3]], [[-0.5]]])
    end = torch.tensor([[[-0.2]], [[0.7]]])
    out = bounded_random_walk(start, end, std=0.05, length=128)
    torch.testing.assert_close(out[..., 0:1], start, atol=1e-4, rtol=0)
    torch.testing.assert_close(out[..., -1:], end, atol=1e-4, rtol=0)


def test_reproducible_with_same_seed():
    start = torch.zeros(2, 1, 1)
    end = torch.zeros(2, 1, 1)
    torch.manual_seed(42)
    a = bounded_random_walk(start, end, std=0.2, length=64)
    torch.manual_seed(42)
    b = bounded_random_walk(start, end, std=0.2, length=64)
    torch.testing.assert_close(a, b)


def test_different_seeds_give_different_walks():
    start = torch.zeros(2, 1, 1)
    end = torch.zeros(2, 1, 1)
    torch.manual_seed(1)
    a = bounded_random_walk(start, end, std=0.2, length=64)
    torch.manual_seed(2)
    b = bounded_random_walk(start, end, std=0.2, length=64)
    assert not torch.allclose(a, b)


def test_zero_std_gives_a_straight_trend_line():
    start = torch.tensor([[[0.0]]])
    end = torch.tensor([[[1.0]]])
    out = bounded_random_walk(start, end, std=0.0, length=11)
    expected = torch.linspace(0.0, 1.0, 11).view(1, 1, 11)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=0)


def test_2d_start_end_no_longer_cross_broadcasts():
    """
    Regression test: a 2-D (batch, 1) start/end used to silently produce
    a cross-broadcast (batch, batch, length) result instead of
    (batch, 1, length) -- fixed by normalizing start/end to at least 3-D
    before any arithmetic. Also checks the actual per-sample start/end
    values land correctly (not just the shape), which the cross-broadcast
    bug would have gotten wrong even where the shape coincidentally
    matched (e.g. batch size 1).
    """
    start = torch.tensor([[0.2], [-0.6], [0.1]])
    end = torch.tensor([[-0.3], [0.4], [0.5]])
    out = bounded_random_walk(start, end, std=0.05, length=32)
    assert out.shape == (3, 1, 32)
    torch.testing.assert_close(out[:, 0, 0:1], start, atol=1e-4, rtol=0)
    torch.testing.assert_close(out[:, 0, -1:], end, atol=1e-4, rtol=0)


def test_bounds_assertion_rejects_out_of_range_start_end():
    start = torch.tensor([[[2.0]]])  # outside [-1, 1]
    end = torch.tensor([[[0.0]]])
    try:
        bounded_random_walk(start, end, std=0.1, lower_bound=-1, upper_bound=1, length=32)
        assert False, "expected an AssertionError for out-of-bounds start"
    except AssertionError:
        pass
