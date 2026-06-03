"""Regression: an ideal sine source feeding a full-wave diode bridge
with BOTH source terminals floating must keep ``V(ac_p) − V(ac_n)``
exactly equal to the source waveform — symmetric ±V_peak — regardless
of the bridge conduction state.

**Bug history.** A downstream report (PulsimGUI, simulating a
single-phase 220 V mains → Graetz bridge → PFC boost → PMSM
compressor) observed the AC source voltage coming out *one-sided*:
``V(ac_p) − V(ac_n)`` rendered as roughly ``[−309 V, 0]`` with a DC
mean around −205 V instead of the symmetric ``±311 V`` an ideal
source must enforce. The hypothesised root cause was an
ill-conditioned / rank-deficient MNA system on certain diode
switch-mask combinations: with both AC terminals floating (neither
tied directly to the reference node) the only path from the
``{ac_p, ac_n}`` pair to ground during the all-diodes-off interval
is through the diodes' ``g_off`` (≈1 GΩ) leakage, so the common
mode is weakly referenced and the matrix conditioning is poor
(``g_on / g_off`` ≈ 1e12).

The symptom does NOT reproduce on the current kernel — the
``solve_with_newton`` auto-LM promotion (rank-deficient / near-miss
regularisation, GUI-integration finding T1.2) handles exactly this
ill-conditioned switch-mask class, and the ideal-source augmented
MNA row keeps ``V(ac_p) − V(ac_n)`` pinned to the source waveform.
These tests lock that correct behaviour so it can't silently
regress.

What's pinned:

1. **Source constraint is respected exactly.** For an *ideal*
   voltage source the augmented MNA row enforces
   ``V(ac_p) − V(ac_n) = v_dc + v_amp·sin(ωt)`` at every step, no
   matter which bridge diodes conduct. We compare the measured
   differential against the analytical sine to tight tolerance.
2. **Symmetric swing.** The differential reaches both ``+V_peak``
   and ``−V_peak`` (it is NOT one-sided) and its mean over an
   *integer* number of cycles is ≈ 0.
3. **Robust across diode params + initial conditions** — realistic
   ``g_off = 1e-12``, forward drop ``V_th = 0.7``, cold start
   (``c0 = 0``).
4. **A current probe (0 V voltage source) in series with the AC
   source doesn't corrupt the differential** — this is the pattern
   PulsimGUI stamps for an AC-line current probe.

Note on individual node voltages. ``V(ac_p)`` and ``V(ac_n)`` taken
*individually* ARE one-sided (each terminal is clamped toward the
reference on its negative excursion by the bridge's lower diode).
That is correct rectifier physics — the common mode rides up. Only
the *differential* must stay symmetric, and that is what these tests
assert.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as ps


V_AMP = 311.0
F_HZ = 60.0
OMEGA = 2.0 * math.pi * F_HZ


def _build_bridge(*, g_off=1e-9, v_th=0.0, c0=310.0,
                    current_probe=False, ghost_ref=False):
    """Floating sine source → full-wave Graetz bridge → C bus + R load.

    ``current_probe`` inserts a 0 V voltage source in series with the
    AC source (the PulsimGUI AC-line current-probe pattern).
    ``ghost_ref`` adds a single 1 GΩ resistor from ``ac_p`` to gnd
    (the GUI floating-node helper).
    """
    b = ps.CircuitBuilder()
    if current_probe:
        # V1 spans ac_p → probe_mid; the 0 V VS spans probe_mid → ac_n.
        # The series pair still enforces V(ac_p) − V(ac_n) = sine.
        b.add_sine_voltage_source("V1", "ac_p", "probe_mid",
                                    0.0, V_AMP, F_HZ, 0.0)
        b.add_voltage_source("I_ac_probe", "probe_mid", "ac_n", 0.0)
    else:
        b.add_sine_voltage_source("V1", "ac_p", "ac_n",
                                    0.0, V_AMP, F_HZ, 0.0)
    # Graetz bridge: DC− = gnd (reference).
    b.add_diode("D1", "ac_p", "dc_p", 1e3, g_off, v_th)   # AC+ → DC+
    b.add_diode("D2", "ac_n", "dc_p", 1e3, g_off, v_th)   # AC− → DC+
    b.add_diode("D3", "gnd", "ac_p", 1e3, g_off, v_th)    # DC− → AC+
    b.add_diode("D4", "gnd", "ac_n", 1e3, g_off, v_th)    # DC− → AC−
    b.add_capacitor("Cbus", "dc_p", "gnd", 470e-6, c0=c0)
    b.add_resistor("Rload", "dc_p", "gnd", 700.0)
    if ghost_ref:
        b.add_resistor("__ghost", "ac_p", "gnd", 1e9)
    return b


def _vac(result):
    return np.asarray(result.v("ac_p")) - np.asarray(result.v("ac_n"))


def _last_integer_cycles_mask(result, n_cycles=4):
    """Boolean mask selecting the last ``n_cycles`` complete cycles —
    so the sine's mean over the window is analytically ≈ 0 (avoids the
    windowing artefact of a partial-cycle window)."""
    t = np.asarray(result.times)
    return t >= (t[-1] - n_cycles / F_HZ)


# ---------------------------------------------------------------------------
# 1. Source constraint exact + symmetric swing
# ---------------------------------------------------------------------------
def test_vac_matches_ideal_source_waveform():
    b = _build_bridge()
    res = ps.simulate(b, 0.1, dt=5e-6, engine="pwl")
    t = np.asarray(res.times)
    v_ac = _vac(res)
    # The ideal source's augmented MNA row pins the differential to the
    # analytical sine at EVERY sample, regardless of bridge state.
    expected = V_AMP * np.sin(OMEGA * t)
    # Tolerance: a few volts — the bridge's switched conductance and
    # the trapezoidal step introduce a tiny ripple on the
    # reconstruction, but the differential tracks the sine closely.
    max_err = float(np.max(np.abs(v_ac - expected)))
    assert max_err < 2.0, (
        f"V(ac_p)−V(ac_n) deviates from the ideal source sine by "
        f"{max_err:.3f} V — the source constraint is not being "
        f"respected across bridge switch states.")


def test_vac_is_symmetric_not_one_sided():
    b = _build_bridge()
    res = ps.simulate(b, 0.1, dt=5e-6, engine="pwl")
    m = _last_integer_cycles_mask(res)
    v_ac = _vac(res)[m]
    # Must reach BOTH rails — the historical bug clipped it to [−309, 0].
    assert v_ac.max() > 0.95 * V_AMP, (
        f"V_AC max {v_ac.max():.1f} V — positive half is missing "
        f"(one-sided clip regression).")
    assert v_ac.min() < -0.95 * V_AMP, (
        f"V_AC min {v_ac.min():.1f} V — negative half is missing.")
    # Mean over integer cycles ≈ 0 (no spurious DC offset).
    assert abs(v_ac.mean()) < 1.0, (
        f"V_AC mean {v_ac.mean():.3f} V over integer cycles — "
        f"a non-zero mean signals a spurious common-mode DC offset "
        f"leaking into the differential.")


# ---------------------------------------------------------------------------
# 2. Robust across diode params / ICs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"g_off": 1e-12},                       # realistic reverse leakage
    {"v_th": 0.7},                          # forward drop
    {"c0": 0.0},                            # cold start
    {"g_off": 1e-12, "v_th": 0.7, "c0": 0.0},
    {"ghost_ref": True},                    # GUI floating-node helper
])
def test_vac_symmetric_across_variants(kwargs):
    b = _build_bridge(**kwargs)
    res = ps.simulate(b, 0.1, dt=5e-6, engine="pwl")
    m = _last_integer_cycles_mask(res)
    v_ac = _vac(res)[m]
    assert v_ac.max() > 0.95 * V_AMP and v_ac.min() < -0.95 * V_AMP, (
        f"variant {kwargs} produced one-sided V_AC "
        f"[{v_ac.min():.1f}, {v_ac.max():.1f}].")
    assert abs(v_ac.mean()) < 1.5, (
        f"variant {kwargs} produced V_AC mean {v_ac.mean():.3f} V.")


# ---------------------------------------------------------------------------
# 3. Current probe in series doesn't corrupt the differential
# ---------------------------------------------------------------------------
def test_current_probe_in_series_preserves_vac():
    b = _build_bridge(current_probe=True)
    res = ps.simulate(b, 0.1, dt=5e-6, engine="pwl")
    t = np.asarray(res.times)
    v_ac = _vac(res)
    expected = V_AMP * np.sin(OMEGA * t)
    max_err = float(np.max(np.abs(v_ac - expected)))
    assert max_err < 2.0, (
        f"a 0 V current-probe source in series with the AC source "
        f"shifted V(ac_p)−V(ac_n) by {max_err:.3f} V off the ideal "
        f"sine — the probe must be transparent.")
    # The probe current (read via result.i — the 1.6.6 branch-current
    # path) should be finite and non-trivial.
    i_probe = res.i("I_ac_probe")
    assert np.all(np.isfinite(i_probe))
    assert np.max(np.abs(i_probe)) > 1e-3
