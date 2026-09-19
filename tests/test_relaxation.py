"""
Unit tests for src.relaxation (v2.0 handover section 10).

The four equations tested here were provided directly by the repo owner
(first author of the cited paper) from Table 1 of its supplement, not
derived from memory -- see src/relaxation.py's module docstring.
"""
import math

import pytest
import torch

from src.relaxation import t1_recovery, t1_star_recovery, t2_decay, t2_star_decay


# ---------------------------------------------------------------------------
# T1 recovery: M0 * (1 - exp(-TR/T1))
# ---------------------------------------------------------------------------

def test_t1_recovery_zero_tr_gives_zero_signal():
    """No recovery time at all -> no longitudinal magnetization available."""
    assert t1_recovery(TR=0.0, T1=1.0) == pytest.approx(0.0)


def test_t1_recovery_tr_much_greater_than_t1_approaches_m0():
    """Fully relaxed (TR >> T1) -> full M0 recovered."""
    assert t1_recovery(TR=1000.0, T1=1.0, m0=5.0) == pytest.approx(5.0, abs=1e-6)


def test_t1_recovery_known_value():
    # TR = T1 -> 1 - exp(-1) ~= 0.6321
    assert t1_recovery(TR=1.0, T1=1.0) == pytest.approx(1 - math.exp(-1))


def test_t1_recovery_scalar_and_tensor_agree():
    scalar = t1_recovery(TR=1.5, T1=0.8, m0=2.0)
    tensor = t1_recovery(TR=torch.tensor(1.5), T1=torch.tensor(0.8), m0=torch.tensor(2.0))
    assert tensor.item() == pytest.approx(scalar)


def test_t1_recovery_broadcasts_over_tensors():
    TR = torch.tensor([1.0, 2.0, 3.0])
    T1 = torch.tensor(1.0)
    out = t1_recovery(TR, T1)
    expected = 1 - torch.exp(-TR / T1)
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# T1* apparent/steady-state recovery (Kaptein et al. 1976, Taylor et al. 2016)
# ---------------------------------------------------------------------------

def test_t1_star_recovery_90_degrees_reduces_to_t1_recovery():
    """At a 90-degree flip angle, cos(theta)=0 and sin(theta)=1, so the
    steady-state (T1*) equation reduces exactly to plain T1 recovery."""
    TR, T1 = 1.2, 0.9
    star = t1_star_recovery(TR=TR, T1=T1, flip_angle=90.0)
    plain = t1_recovery(TR=TR, T1=T1)
    assert star == pytest.approx(plain)


def test_t1_star_recovery_zero_flip_angle_gives_zero_signal():
    """No excitation -> no transverse signal, regardless of relaxation."""
    assert t1_star_recovery(TR=1.0, T1=1.0, flip_angle=0.0) == pytest.approx(0.0, abs=1e-9)


def test_t1_star_recovery_degrees_vs_radians_agree():
    TR, T1, angle_deg = 1.0, 0.8, 30.0
    via_degrees = t1_star_recovery(TR, T1, angle_deg, degrees=True)
    via_radians = t1_star_recovery(TR, T1, math.radians(angle_deg), degrees=False)
    assert via_degrees == pytest.approx(via_radians)


def test_t1_star_recovery_matches_manual_ernst_equation():
    TR, T1, angle = 1.3, 0.85, 42.0
    theta = math.radians(angle)
    e1 = math.exp(-TR / T1)
    expected = (1 - e1) / (1 - math.cos(theta) * e1) * math.sin(theta)
    assert t1_star_recovery(TR, T1, angle) == pytest.approx(expected)


def test_t1_star_recovery_tensor_broadcast():
    TR = torch.tensor(1.0)
    T1 = torch.tensor(0.8)
    angles = torch.tensor([10.0, 45.0, 90.0])
    out = t1_star_recovery(TR, T1, angles)
    assert out.shape == angles.shape
    # 90 degrees should match plain T1 recovery
    torch.testing.assert_close(out[-1], t1_recovery(TR, T1))


# ---------------------------------------------------------------------------
# T2 / T2* decay: M0 * exp(-TE/T2)
# ---------------------------------------------------------------------------

def test_t2_decay_zero_te_gives_full_signal():
    assert t2_decay(TE=0.0, T2=0.03, m0=3.0) == pytest.approx(3.0)


def test_t2_decay_te_much_greater_than_t2_approaches_zero():
    assert t2_decay(TE=100.0, T2=0.03) == pytest.approx(0.0, abs=1e-9)


def test_t2_decay_known_value():
    assert t2_decay(TE=0.03, T2=0.03) == pytest.approx(math.exp(-1))


def test_t2_star_decay_is_the_same_formula_as_t2_decay():
    assert t2_star_decay(TE=0.05, T2_star=0.04) == pytest.approx(t2_decay(TE=0.05, T2=0.04))


def test_t2_decay_tensor_broadcast():
    TE = torch.tensor([0.01, 0.02, 0.03])
    T2 = torch.tensor(0.03)
    out = t2_decay(TE, T2)
    torch.testing.assert_close(out, torch.exp(-TE / T2))


def test_t2_decay_shorter_t2_decays_faster():
    TE = 0.03
    long_t2 = t2_decay(TE, T2=0.5)
    short_t2 = t2_decay(TE, T2=0.05)
    assert short_t2 < long_t2
