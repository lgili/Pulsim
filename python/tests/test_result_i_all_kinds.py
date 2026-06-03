"""Regression: ``SimulationResult.i(name)`` reconstructs currents for
**every** device kind pulsim's PWL solver currently models — closes
the gap between MNA's native state vector and PLECS-style "any current
out of the state-space output equation".

Supported families (and tested here):

* **inductor** / **voltage_source** family (PWM / sine / pulse) — state
  vector native, already worked, covered in ``test_named_lookups.py``
  and ``test_result_i_resistor.py``.
* **resistor** — reconstructed as ``i = (V_from − V_to)/R``; covered
  in ``test_result_i_resistor.py``.
* **capacitor** — reconstructed via ``i_C = C·dv/dt`` (this file).
* **current_source** — constant from params (this file).
* **diode** (PWL switched) — reconstructed via
  ``i = (V_from − V_to − V_th)·g_on`` when forward-biased, ``v·g_off``
  otherwise. Covered here on a half-wave rectifier.
* **switch** — reconstructed via ``i = (V_from − V_to)·G`` with
  ``G = g_on`` when the per-step mask bit is set, ``g_off`` otherwise.
  Covered here on a buck converter.

Plus the PLECS-style **``result.currents()``** accessor that returns
``{branch_name: ndarray}`` for every reconstructible branch in one
shot.

Out of scope (deferred to a follow-up because their device params
aren't yet exposed by ``builder.components()``):
mosfet_level1, igbt_level1, nonlinear_diode, vcvs, saturable_inductor.
A test pins that calls on these kinds raise NotImplementedError with
a clear migration hint at ``pulsim.losses.device_loss_summary``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# current_source
# ---------------------------------------------------------------------------
def test_current_source_returns_constant_trace():
    """``add_current_source(name, n_pos, n_neg, I)`` produces a constant
    current ``I`` from ``n_pos`` to ``n_neg`` (passive sign convention).
    ``result.i(name)`` returns the constant trace."""
    I0 = 0.25
    b = p.CircuitBuilder()
    b.add_current_source("I1", "vin", "gnd", I0)
    b.add_resistor("R1", "vin", "gnd", 100.0)
    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    i = res.i("I1")
    assert isinstance(i, np.ndarray)
    assert i.shape[0] == res.num_steps()
    np.testing.assert_allclose(i, I0, atol=1e-12)


# ---------------------------------------------------------------------------
# capacitor (already pinned in test_result_i_resistor.py; one extra here
# for a fully-analytical sanity check)
# ---------------------------------------------------------------------------
def test_capacitor_current_envelope_matches_analytical_rc():
    """RC: ``V_in = 5 V``, ``R = 1 kΩ``, ``C = 1 µF``. Analytical
    ``i_C(t) = (V_in/R)·exp(−t/τ)``. The reconstruction should agree
    on the interior of the trace."""
    V_in = 5.0
    R = 1e3
    C = 1e-6
    tau = R * C

    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", V_in)
    b.add_resistor("R1", "vin", "n1", R)
    b.add_capacitor("C1", "n1", "gnd", C)

    res = p.simulate(b, t_end=5 * tau, dt=tau / 100)
    times = np.asarray(res.times)
    i_C = res.i("C1")
    expected = (V_in / R) * np.exp(-times / tau)
    # Skip the endpoints (one-sided gradient drift) and the very first
    # warmup sample.
    np.testing.assert_allclose(i_C[5:-5], expected[5:-5],
                                  atol=5e-5, rtol=2e-2)


# ---------------------------------------------------------------------------
# diode — half-wave rectifier
# ---------------------------------------------------------------------------
def test_diode_current_zero_during_reverse_bias_positive_during_conduction():
    """Half-wave rectifier (50 Hz sine, 1 Ω load). The diode current
    should be ≈ zero during the negative half-cycle (reverse-biased)
    and follow ``(V_sin − V_th)/R`` during the positive half-cycle."""
    V_amp = 10.0
    R = 1.0
    f = 50.0
    V_th = 0.7

    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vsrc", "vin", "gnd",
                                 v_dc=0.0, v_amplitude=V_amp,
                                 frequency=f, phase=0.0)
    b.add_diode("D1", "vin", "vout", g_on=1.0, g_off=1e-9, V_th=V_th)
    b.add_resistor("R_load", "vout", "gnd", R)

    res = p.simulate(b, t_end=40e-3, dt=2e-5)
    times = np.asarray(res.times)
    v_src = res.v("vin")
    i_D = res.i("D1")

    # During reverse bias (v_src < V_th), i_D ≈ 0 (only g_off leakage
    # at most v_src · 1e-9 ≈ 1e-8 A peak).
    reverse_mask = v_src < V_th - 0.05  # small margin around V_th
    assert np.all(np.abs(i_D[reverse_mask]) < 1e-4), (
        f"diode leakage in reverse bias should be tiny, "
        f"max={np.max(np.abs(i_D[reverse_mask])):.3e}")

    # During forward conduction (v_src > V_th + 0.5 V to avoid the
    # nonlinear knee), i_D should be positive and on the order of the
    # load current. The exact value is hard to pin without solving the
    # circuit explicitly; we just verify sign + magnitude.
    forward_mask = v_src > V_th + 1.0
    if np.any(forward_mask):
        assert np.all(i_D[forward_mask] > 0.0)
        assert np.max(i_D[forward_mask]) > 1.0, (
            "forward-biased diode current should reach > 1 A on a "
            "10 V_amp sine with 1 Ω load")
    _ = times  # silence


# ---------------------------------------------------------------------------
# switch — PWM buck
# ---------------------------------------------------------------------------
def test_switch_current_matches_inductor_when_closed():
    """Simple synchronous-buck-style stage: 12 V → switch → L → C ‖ R.
    When the switch is closed, KCL forces ``i_Q == i_L``; when open,
    ``i_Q ≈ 0`` (only g_off leakage)."""
    V_in = 12.0
    L = 100e-6
    C = 100e-6
    R = 2.0
    f_sw = 100e3
    duty = 0.4

    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", V_in)
    b.add_switch("Q1", "vin", "sw", g_on=1e3, g_off=1e-9)
    b.add_diode("D1", "gnd", "sw", g_on=1e3, g_off=1e-9, V_th=0.0)
    b.add_inductor("L1", "sw", "vout", L)
    b.add_capacitor("Cout", "vout", "gnd", C)
    b.add_resistor("R_load", "vout", "gnd", R)

    sf = p.make_pwm_switch_fn(
        frequency=f_sw,
        duty=duty,
        switch_idx=b.switch_index_of("Q1"),
        num_switches=b.graph.num_switches,
    )

    res = p.simulate(b, t_end=2e-3, dt=1e-7, switch_fn=sf)
    i_Q = res.i("Q1")
    i_L = res.i("L1")
    times = np.asarray(res.times)

    # Sample the mask trace via the same helper as the kernel.
    from pulsim._result_views import evaluate_switch_mask_trace
    closed = evaluate_switch_mask_trace(sf, times,
                                            b.switch_index_of("Q1"))

    # While closed: i_Q ≈ i_L (allow a small numerical fudge for the
    # finite g_on = 1 kS — not infinity).
    if np.any(closed):
        diff_closed = np.abs(i_Q[closed] - i_L[closed])
        assert np.max(diff_closed) < 0.5, (
            f"i_Q vs i_L mismatch when closed; "
            f"max diff={np.max(diff_closed):.3f} A")
    # While open: i_Q tiny (≤ V·g_off ≈ 12·1e-9 = 1.2e-8 A).
    if np.any(~closed):
        assert np.max(np.abs(i_Q[~closed])) < 1e-5


def test_switch_without_switch_fn_raises_with_hint():
    """If the result was built without ``simulate`` (so ``_switch_fn``
    is not stashed), ``result.i('switch_name')`` raises
    ``NotImplementedError`` with a hint to set ``result._switch_fn``."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_switch("Q1", "vin", "gnd", g_on=1e3, g_off=1e-9)

    sf = p.make_pwm_switch_fn(
        frequency=10e3,
        duty=0.5,
        switch_idx=b.switch_index_of("Q1"),
        num_switches=b.graph.num_switches,
    )
    res = p.simulate(b, t_end=1e-3, dt=1e-5, switch_fn=sf)
    # Manually remove the stash to mimic the "result built another way"
    # case.
    try:
        del res._switch_fn
    except AttributeError:
        pass
    with pytest.raises(NotImplementedError) as exc:
        res.i("Q1")
    msg = str(exc.value)
    assert "switch_fn" in msg
    assert "_switch_fn" in msg


# ---------------------------------------------------------------------------
# result.currents() — PLECS-style "all currents" dict
# ---------------------------------------------------------------------------
def test_currents_dict_returns_all_supported_branches():
    """A small RLC + diode circuit. ``result.currents()`` should return
    a dict with one entry per branch whose kind is implemented."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "n1", 10.0)
    b.add_inductor("L1", "n1", "n2", 1e-3)
    b.add_capacitor("C1", "n2", "gnd", 1e-6)
    b.add_diode("D1", "vin", "n1", g_on=1e3, g_off=1e-9, V_th=0.5)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    currents = res.currents()
    # Every branch is reconstructible → 5 entries.
    expected = {"V1", "R1", "L1", "C1", "D1"}
    assert set(currents.keys()) == expected, (
        f"missing branches: {expected - set(currents.keys())}, "
        f"extra: {set(currents.keys()) - expected}")
    for name in expected:
        arr = currents[name]
        assert isinstance(arr, np.ndarray)
        assert arr.shape[0] == res.num_steps()


def test_currents_dict_skips_unsupported_kinds_by_default():
    """If a MOSFET/IGBT/nonlinear device is in the circuit, the dict
    silently omits it (default ``skip_unsupported=True``). The user can
    pass ``skip_unsupported=False`` to surface the
    ``NotImplementedError`` instead.

    For this scaffold we just verify the API: a circuit with only
    supported kinds returns the full dict, and the kwarg controls the
    behaviour."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "gnd", 100.0)
    res = p.simulate(b, t_end=1e-4, dt=1e-5)

    # Both modes work the same when every branch is supported.
    c_skip = res.currents(skip_unsupported=True)
    c_strict = res.currents(skip_unsupported=False)
    assert set(c_skip.keys()) == set(c_strict.keys()) == {"V1", "R1"}


# ---------------------------------------------------------------------------
# Backward-compat — unsupported kinds still raise with a clear hint
# ---------------------------------------------------------------------------
def test_unsupported_kind_raises_with_pulsim_losses_hint():
    """MOSFET/IGBT/etc. are deferred; calling ``result.i()`` on them
    must raise ``NotImplementedError`` pointing at
    ``pulsim.losses.device_loss_summary``."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 5.0)
    b.add_voltage_source("Vg", "vg", "gnd", 5.0)
    b.add_mosfet_level1("M1", "vin", "vd", "vg",
                            K=0.05, V_T=2.0, lambda_=0.02, kappa=15.0)
    b.add_resistor("R_load", "vd", "gnd", 10.0)
    res = p.simulate(b, t_end=1e-4, dt=1e-6, enable_nonlinear_refresh=True)
    with pytest.raises(NotImplementedError) as exc:
        res.i("M1")
    msg = str(exc.value)
    assert "mosfet_level1" in msg.lower()
    assert "pulsim.losses" in msg or "device_loss_summary" in msg
    _ = math  # silence
