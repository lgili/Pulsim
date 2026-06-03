"""Regression: ``SimulationResult.i(name)`` now also works on
**resistor** branches by reconstructing the current from the two
terminal node voltages and the resistor's stored ``R_ohms``
(``i = (V_from − V_to) / R``).

**Why this exists.** PulsimGUI integrates ``current_probe`` virtual
components by stamping a low-value (``1e-4 Ω`` default) bypass
resistor named ``__IP_BYPASS_<probe>`` in series with the measured
branch. Pre-fix the GUI had to reconstruct the probe current
manually because ``result.i(<bypass_name>)`` only worked for
inductors / voltage sources and threw ``NotImplementedError`` on
resistors. The reconstruction path
(``_repair_current_probe_channels_from_bypass`` in the GUI)
duplicated work the kernel could do trivially.

Extending ``result.i`` covers the case in one place:

* Inductor / voltage-source branches keep reading the state
  variable in O(1).
* **Resistor branches** reconstruct ``i = (V_from − V_to) / R``
  from the result's per-step node voltages.
* Capacitor / diode / MOSFET / switch branches still raise
  ``NotImplementedError`` pointing the caller at
  ``device_loss_summary`` (their reconstruction needs a kind-specific
  stamp evaluation that lives in ``pulsim.losses``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# 1. Trivial DC — V_in / R loop
# ---------------------------------------------------------------------------
def test_dc_loop_resistor_current():
    """5 V across a 100 Ω resistor → 0.05 A everywhere in time.

    Pins the steady-state reconstruction: pure resistive load, no
    reactive transient — ``i_R`` is constant 50 mA from the first
    sample.
    """
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "gnd", 100.0)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    i_R = res.i("R1")
    assert isinstance(i_R, np.ndarray)
    assert i_R.shape[0] == res.num_steps()
    # 5 V / 100 Ω = 50 mA. Skip the t=0 sample — pulsim records the
    # provided ``initial_state`` at sample 0 (all-zeros unless the
    # caller seeds ICs), and the steady-state DC value lands from
    # sample 1 onward.
    np.testing.assert_allclose(i_R[1:], 0.05, atol=1e-9)
    # Time-indexed access also works.
    assert float(res.i("R1", -1)) == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Sign convention follows the add_resistor(from, to) order
# ---------------------------------------------------------------------------
def test_resistor_current_sign_follows_from_to_order():
    """``i_R > 0`` when conventional current flows from `from_node`
    to `to_node`. Same convention as inductors and voltage sources."""
    # 10 V on "p" → R → "gnd". add_resistor("Rfwd", "p", "gnd", ...) →
    # current 1 A flows p → gnd, so result.i("Rfwd") > 0.
    b_fwd = p.CircuitBuilder()
    b_fwd.add_voltage_source("V_p", "p", "gnd", 10.0)
    b_fwd.add_resistor("Rfwd", "p", "gnd", 10.0)
    res_fwd = p.simulate(b_fwd, t_end=1e-4, dt=1e-5)
    i_fwd = float(res_fwd.i("Rfwd", -1))
    assert i_fwd == pytest.approx(1.0, rel=1e-6)

    # Swap node order — physically same current but result.i reports
    # opposite sign because add_resistor's from/to is flipped.
    b_rev = p.CircuitBuilder()
    b_rev.add_voltage_source("V_p", "p", "gnd", 10.0)
    b_rev.add_resistor("Rrev", "gnd", "p", 10.0)
    res_rev = p.simulate(b_rev, t_end=1e-4, dt=1e-5)
    i_rev = float(res_rev.i("Rrev", -1))
    assert i_rev == pytest.approx(-1.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. RC charging transient — exponential approach
# ---------------------------------------------------------------------------
def test_rc_transient_resistor_current_decays_exponentially():
    """RC: ``V_in = 5 V``, ``R = 1 kΩ``, ``C = 1 µF`` → τ = 1 ms.

    The current through R decays as ``i_R(t) = (V_in / R)·exp(−t/τ)``.
    Check the analytical envelope at a few sample points.
    """
    V_in = 5.0
    R = 1e3
    C = 1e-6
    tau = R * C   # 1 ms

    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", V_in)
    b.add_resistor("R1", "vin", "n1", R)
    b.add_capacitor("C1", "n1", "gnd", C)

    t_end = 5 * tau
    dt = tau / 100
    res = p.simulate(b, t_end=t_end, dt=dt)
    times = np.asarray(res.times)
    i_R = res.i("R1")

    # Analytical: i_R(t) = (V_in / R) · exp(−t/τ). Numerical kernel
    # error stays under 1 mA after the first 5 samples (warm-up).
    expected = (V_in / R) * np.exp(-times / tau)
    np.testing.assert_allclose(i_R[5:], expected[5:],
                                  atol=5e-5, rtol=2e-3)


# ---------------------------------------------------------------------------
# 4. PulsimGUI use case — current_probe via bypass resistor
# ---------------------------------------------------------------------------
def test_bypass_resistor_pattern_recovers_probed_current():
    """Mimics PulsimGUI's `current_probe` virtual component pattern:
    a tiny bypass resistor named ``__IP_BYPASS_<probe>`` sits in
    series with the measured branch. The user wants
    ``result.i("__IP_BYPASS_<probe>")`` to return that branch's
    current; the rest of the pipeline (downstream R load) sees the
    bypass as nearly transparent (``1e-4 Ω`` vs ``100 Ω`` load).

    Plant: 10 V source → __IP_BYPASS → R_load=100 Ω → gnd. Probe
    current should be 10/100 ≈ 0.1 A (bypass adds ~1 µV of drop).
    """
    R_load = 100.0
    R_bypass = 1e-4

    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 10.0)
    b.add_resistor("__IP_BYPASS_I_L", "vin", "n_probe", R_bypass)
    b.add_resistor("R_load", "n_probe", "gnd", R_load)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    i_probe = res.i("__IP_BYPASS_I_L")
    i_load = res.i("R_load")
    # Both should agree on the loop current (series circuit).
    np.testing.assert_allclose(i_probe, i_load, atol=1e-9)
    # Expected ≈ 10 V / (100 + 1e-4) Ω ≈ 0.0999999 A
    assert float(i_probe[-1]) == pytest.approx(
        10.0 / (R_load + R_bypass), rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Backward compat — non-resistor non-inductor still raises
# ---------------------------------------------------------------------------
def test_capacitor_branch_still_raises_not_implemented():
    """Capacitor current reconstruction needs node-voltage
    differentiation (``i_C = C · dv/dt``), which the .i() fast path
    doesn't implement. Calls on capacitor names still raise with the
    documented hint pointing at `pulsim.losses`."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "n1", 1e3)
    b.add_capacitor("C1", "n1", "gnd", 1e-6)

    res = p.simulate(b, t_end=1e-4, dt=1e-5)
    with pytest.raises(NotImplementedError) as exc:
        res.i("C1")
    msg = str(exc.value)
    assert "C1" in msg
    assert "capacitor" in msg.lower()
    assert "pulsim.losses" in msg or "device_loss_summary" in msg


# ---------------------------------------------------------------------------
# 6. Helpful fuzzy-match still works on typos
# ---------------------------------------------------------------------------
def test_typo_on_resistor_name_raises_fuzzy_hint():
    """The unknown-name path goes through the same fuzzy-match that
    inductors/sources used. Verifying it still fires when the typo'd
    name would have resolved to a resistor."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R_load", "vin", "gnd", 100.0)

    res = p.simulate(b, t_end=1e-4, dt=1e-5)
    with pytest.raises(p.NameNotFoundError) as exc:
        res.i("R_lood")     # typo
    assert "R_load" in exc.value.suggestions


# ---------------------------------------------------------------------------
# 7. Mixed circuit — inductor + resistor both work via the same .i()
# ---------------------------------------------------------------------------
def test_RL_circuit_inductor_and_resistor_currents_match():
    """RL step response: i_R == i_L at every step (series loop).

    Pins that the resistor path didn't regress the existing inductor
    path AND that both readings agree (they MUST in a series loop —
    physical constraint, KCL)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", 5.0)
    b.add_resistor("R1", "vin", "n1", 10.0)
    b.add_inductor("L1", "n1", "gnd", 1e-3)

    res = p.simulate(b, t_end=1e-3, dt=1e-5)
    i_R = res.i("R1")
    i_L = res.i("L1")
    np.testing.assert_allclose(i_R, i_L, atol=1e-9)
    # Final value approaches V/R = 0.5 A.
    assert float(i_L[-1]) == pytest.approx(0.5, rel=1e-2)
    _ = math   # silence unused import (kept for symmetry with neighbours)
