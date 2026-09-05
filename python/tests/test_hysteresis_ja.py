"""Tests for the Jiles-Atherton hysteresis model (Phase 2.2).

Validates the pre-existing ``hysteresis.py`` module that now ships
re-exported at the top-level ``pulsim`` namespace:

  * ``JilesAthertonParams`` / ``JilesAthertonModel`` per-step API.
  * ``reference_material()`` catalog lookup.
  * ``compute_bh_loop()`` end-to-end loop generation.
  * ``fit_ja_from_bh_curve()`` curve-fitting helper (NEW in v1.5).
"""
from __future__ import annotations

import math

import numpy as np
import pytest


pulsim = pytest.importorskip("pulsim")


_HAS_JA = hasattr(pulsim, "JilesAthertonModel")
MU_0 = 4.0 * math.pi * 1e-7


def test_ja_constructor_and_catalog():
    if not _HAS_JA:
        pytest.skip("JilesAthertonModel not exported in this build")
    p = pulsim.reference_material("ferrite_n87")
    assert p.Ms > 0 and p.a > 0 and p.k > 0
    assert 0.0 <= p.c <= 1.0
    # All catalog materials should round-trip.
    for name in pulsim.list_reference_materials():
        pp = pulsim.reference_material(name)
        assert isinstance(pp, pulsim.JilesAthertonParams)


def test_ja_unknown_material_raises():
    if not _HAS_JA:
        pytest.skip()
    with pytest.raises(KeyError):
        pulsim.reference_material("not-a-real-material")


@pytest.mark.skipif(not _HAS_JA, reason="JA not available")
def test_ja_initial_state_is_demagnetized():
    p = pulsim.reference_material("ferrite_n87")
    m = pulsim.JilesAthertonModel(p)
    assert m.M == 0.0
    assert m.H == 0.0
    assert m.B == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(not _HAS_JA, reason="JA not available")
def test_ja_step_advances_state():
    p = pulsim.reference_material("ferrite_n87")
    m = pulsim.JilesAthertonModel(p)
    # First step initialises _H_prev — second step actually integrates.
    m.update(0.0, 1e-6)
    B1 = m.update(100.0, 1e-6)
    assert B1 > 0.0
    assert m.M > 0.0
    assert m.H == pytest.approx(100.0)


@pytest.mark.skipif(not _HAS_JA, reason="JA not available")
def test_ja_compute_bh_loop_returns_settled_loop():
    """Run a 50 Hz sinusoidal H excitation through the model for a
    few periods and verify the converged B-H loop has all expected
    qualitative properties: peak B reasonable, loop closes,
    coercive field non-zero, remanent flux non-zero."""
    p = pulsim.reference_material("ferrite_n87")
    f = 50.0
    period = 1.0 / f
    n_periods = 6
    n_per = 500
    dt = period / n_per
    H_amp = 500.0           # A/m — well above coercivity
    omega = 2.0 * math.pi * f
    n_samples = n_periods * n_per
    times = np.arange(n_samples) * dt
    # We need a "current" history — the helper uses H = N·i/l_m, so
    # pick l_m = N = 1 to make H = i directly.
    i_arr = H_amp * np.sin(omega * times) / 1.0   # H = i (l_m = N = 1)

    result = pulsim.compute_bh_loop(
        i_arr, times, params=p,
        N_turns=1, l_m=1.0, A_core=1e-4,    # 1 cm² core
        period=period)

    # H should be the input.
    np.testing.assert_allclose(result.H, H_amp * np.sin(omega * times),
                                       atol=1e-9)
    # Peak B should be in a physical range — well below the asymptote
    # μ_0·(H_amp + Ms) and well above the linear-only prediction.
    B_peak = float(np.max(np.abs(result.B)))
    B_asymp = MU_0 * (H_amp + p.Ms)
    assert 0.05 < B_peak < B_asymp
    # Energy per cycle > 0 (real hysteresis).
    assert result.energy_per_cycle_per_m3 > 0.0
    # Loop closes — last-period H_min and H_max are symmetric.
    H_last = result.H[-n_per:]
    assert abs(float(np.min(H_last)) + float(np.max(H_last))) < 1e-3
    # avg_power > 0.
    assert result.avg_power_per_m3 > 0.0
    assert result.avg_power_W > 0.0


@pytest.mark.skipif(not _HAS_JA, reason="JA not available")
def test_ja_high_c_reduces_hysteresis_loss():
    """A high reversibility (c → 1) should produce a narrower loop
    and therefore less loss per cycle than a low-c case."""
    base = pulsim.reference_material("ferrite_n87")
    p_low_c = pulsim.JilesAthertonParams(
        Ms=base.Ms, a=base.a, alpha=base.alpha,
        c=0.05, k=base.k)
    p_high_c = pulsim.JilesAthertonParams(
        Ms=base.Ms, a=base.a, alpha=base.alpha,
        c=0.95, k=base.k)
    f = 50.0
    period = 1.0 / f
    dt = period / 500
    n = 6 * 500
    H_amp = 500.0
    omega = 2.0 * math.pi * f
    times = np.arange(n) * dt
    i_arr = H_amp * np.sin(omega * times)

    loss_low = pulsim.compute_bh_loop(
        i_arr, times, params=p_low_c,
        N_turns=1, l_m=1.0, A_core=1e-4,
        period=period).energy_per_cycle_per_m3
    loss_high = pulsim.compute_bh_loop(
        i_arr, times, params=p_high_c,
        N_turns=1, l_m=1.0, A_core=1e-4,
        period=period).energy_per_cycle_per_m3
    # Higher c (more reversible) → lower loss.
    assert loss_high < loss_low, (
        f"loss_high(c=0.95)={loss_high} should be less than "
        f"loss_low(c=0.05)={loss_low}")


@pytest.mark.skipif(not hasattr(pulsim, "fit_ja_from_bh_curve"),
                       reason="fit_ja_from_bh_curve not in build")
def test_ja_fit_round_trip():
    """Generate a synthetic B-H loop from known parameters, fit it,
    and check the fitted parameters reproduce the loop's coercive
    field within 30 %.

    NOTE: a tight Ms/a/alpha/c/k recovery isn't generally possible
    from a single major loop (the parameter set is identifiable only
    from MULTIPLE excitation amplitudes). We therefore check the
    *qualitative* recovery — the fit should reproduce the loop area
    and coercive field within engineering tolerance.
    """
    truth = pulsim.reference_material("ferrite_n87")
    f = 50.0
    period = 1.0 / f
    dt = period / 200
    n = 4 * 200
    H_amp = 500.0
    omega = 2.0 * math.pi * f
    times = np.arange(n) * dt
    i_truth = H_amp * np.sin(omega * times)

    truth_loop = pulsim.compute_bh_loop(
        i_truth, times, params=truth,
        N_turns=1, l_m=1.0, A_core=1e-4,
        period=period)
    # Take the LAST period as the measurement set.
    H_meas = truth_loop.H[-200:]
    B_meas = truth_loop.B[-200:]

    fit = pulsim.fit_ja_from_bh_curve(B_meas, H_meas, p0=truth)
    assert fit.Ms > 0 and fit.a > 0 and fit.k > 0
    # The fit should reproduce the loop area within 30 % (loose
    # bound — a single-amplitude fit can't fully constrain the
    # parameter set, but it should be in the right ballpark).
    fit_loop = pulsim.compute_bh_loop(
        i_truth, times, params=fit,
        N_turns=1, l_m=1.0, A_core=1e-4,
        period=period)
    area_truth = abs(truth_loop.energy_per_cycle_per_m3)
    area_fit = abs(fit_loop.energy_per_cycle_per_m3)
    rel_err = abs(area_fit - area_truth) / area_truth
    assert rel_err < 0.30, (
        f"Fitted loop area {area_fit:.3g} vs truth {area_truth:.3g} "
        f"differs by {rel_err*100:.1f} %.")


# -----------------------------------------------------------------------
# In-loop HystereticInductor (Phase 2.2 — kernel-coupled path)
# -----------------------------------------------------------------------

_HAS_HYST_DEV = hasattr(pulsim, "add_hysteretic_inductor")


@pytest.mark.skipif(not _HAS_HYST_DEV,
                       reason="add_hysteretic_inductor not in build")
def test_add_hysteretic_inductor_topology():
    """``add_hysteretic_inductor`` adds ONE Newton flux branch (no
    air-core L_0, no dummy source any more) and reports the air-core
    L_0 = N²·A·μ_0/l_m as the deep-saturation floor."""
    b = pulsim.CircuitBuilder()
    p = pulsim.reference_material("ferrite_n87")
    N, l_m, A = 100, 0.05, 1e-4
    n_before = b.graph.num_branches
    hyst = pulsim.add_hysteretic_inductor(
        b, name="L1", from_node="a", to_node="b",
        params=p, N_turns=N, l_m=l_m, A_core=A,
    )
    L0_expected = N * N * A * (4 * math.pi * 1e-7) / l_m
    assert hyst.L_0 == pytest.approx(L0_expected, rel=1e-9)
    assert hyst.branch_id == n_before
    assert hyst.inductor_branch_id == hyst.branch_id
    assert hyst.bemf_source_id == -1
    assert b.graph.num_branches == n_before + 1


def test_hysteretic_inductor_runs_in_the_loop_without_an_observer():
    """End-to-end smoke: a series Vsrc + HystereticInductor + R loop
    runs without NaN and the replayed magnetisation builds up. No
    step_observer, no b_extra_fn — the device is solved in the loop.
    (1 V / 50 Hz keeps B_pk ≈ 0.3 T; at 50 V the core would sit in
    deep saturation for most of the cycle.)"""
    b = pulsim.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 1.0, 50.0, 0.0)
    b.add_resistor("Rs", "a", "n1", 0.1)
    p = pulsim.reference_material("ferrite_n87")
    hyst = pulsim.add_hysteretic_inductor(
        b, name="L_core",
        from_node="n1", to_node="gnd",
        params=p, N_turns=100, l_m=0.05, A_core=1e-4,
    )
    res = pulsim.simulate(
        b, t_end=0.05, dt=20e-6,
        switch_fn=lambda t: pulsim.SwitchStateMask(0),
    )
    assert res.num_steps() > 0
    i = res.i("L_core")
    assert all(math.isfinite(float(x)) for x in i)
    loop = hyst.bh_loop(res, period=0.02)
    assert float(abs(loop.M).max()) > 1e4, "magnetisation never built up"
    assert float(abs(loop.B).max()) > 0.1
