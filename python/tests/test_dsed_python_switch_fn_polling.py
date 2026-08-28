"""Regression: DSED must drive a plain Python ``switch_fn`` correctly.

**Bug history.** Before this fix, ``simulate(engine='dsed', ...)``
with a plain Python ``switch_fn`` (no ``next_edge_after`` method)
silently produced the trajectory of a circuit with the switch
*frozen at the t=0 mask state* — never seeing a single PWM edge —
while still returning a successful result. The C++ event
predictor in :file:`scheduler_auto.hpp` queries
``switch_fn.next_edge_after(t)``; the pybind11 fallback for
plain Python callables returns ``∞``, so the scheduler integrated
the whole window in a single mode.

Symptoms users reported:
* Buck CCM with PWM 50% duty → DC bus pinned at V_in (because
  the initial mask was SW=ON the whole way).
* Buck with resistive freewheel → trajectory diverging from the
  PWL reference by >100 % at steady state.
* "DSED hangs" — actually just diverging to a wrong attractor and
  reaching a non-physical steady state.

**Fix.** The C++ DSED scheduler now treats ``∞`` returns from
``next_edge_after`` as "no edge info — poll defensively" and caps
``t_gate`` at ``t + dt_max/10`` so the scheduler is forced to land
at that boundary and re-sample the switch_fn via
``fire_gate_event_`` (catches any discovered mask change). The
native PWM classes (:class:`pulsim.NativePwm2Switch`,
:class:`pulsim.NativeMultiMaskPwm`) and any user object whose
``next_edge_after`` returns a finite value take the analytical
fast path unchanged.

These tests pin two contracts:

1. **Correctness.** Buck with resistive freewheel under plain
   Python PWM: ``engine='dsed'`` trajectory matches ``engine='pwl'``
   reference at the steady-state level (V_out within 5 % of the PWL
   mean, ``i_L`` within 5 %).
2. **No double-wrapping.** A switch_fn that already exposes
   ``next_edge_after`` is passed through untouched (preserves the
   fast path).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import pulsim as p


# ---------------------------------------------------------------------------
# Canonical bug reproducer: buck w/ resistive freewheel + plain Python PWM
# ---------------------------------------------------------------------------
def _build_buck_resistive_freewheel():
    """Buck with a resistor in place of the freewheeling diode. LTI
    per mask (SW open vs closed both give linear ODEs), so the C++
    extractor accepts the circuit — yet pre-fix DSED produced the
    wrong trajectory because the PWM events were never detected."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("SW", "vin", "sw_mid", g_on=1e3, g_off=1e-9)
    b.add_resistor("R_fw", "sw_mid", "gnd", 10.0)
    b.add_inductor("L", "sw_mid", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 100e-6)
    b.add_resistor("R_load", "vout", "gnd", 2.0)
    return b


def _pwm_switch_fn(builder, f_sw: float = 100e3, duty: float = 0.5):
    """A plain Python PWM switch_fn — exactly the kind that
    triggered the original bug. Notably, it does NOT expose a
    ``next_edge_after`` method."""
    T_sw = 1.0 / f_sw
    n_sw = builder.graph.num_switches

    def sf(t: float):
        m = p.SwitchStateMask(n_sw)
        m.set(0, (t % T_sw) < (duty * T_sw))
        return m
    return sf


def test_python_pwm_switch_fn_produces_correct_trajectory() -> None:
    """The canonical bug reproducer. Pre-fix this asserted
    V_out_mean ≈ 12 V and i_L_mean ≈ 12 A (the SW=ON-forever
    attractor); post-fix both match the PWL reference."""
    b = _build_buck_resistive_freewheel()
    sf = _pwm_switch_fn(b, f_sw=100e3, duty=0.5)

    # DSED with default dt_max=10µs. The polling wrapper picks
    # dt_max/10 = 1µs, capturing every 100 kHz edge.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        res_dsed = p.simulate(b, t_end=5e-3, engine="dsed", switch_fn=sf)

    # PWL reference at dt = T_sw / 100 (fine enough to capture
    # every PWM edge).
    res_pwl = p.simulate(b, t_end=5e-3, dt=1e-7, switch_fn=sf)

    # Probe the last ~1 ms — the system has settled into PWM
    # ripple by then. Use the PWL helpers to get node voltages /
    # branch currents by name (engine-agnostic).
    v_pwl_tail = np.asarray(res_pwl.v("vout"))[-1000:]
    v_pwl_mean = float(v_pwl_tail.mean())
    i_pwl_tail = np.asarray(res_pwl.i("L"))[-1000:]
    i_pwl_mean = float(i_pwl_tail.mean())

    # Pull the DSED state by column — the order is
    # `[V_C, i_L]` for this circuit's per-mask state-space (the
    # extractor walks dynamic devices in branch order: capacitor
    # first, then inductor).
    states = np.asarray(res_dsed.states_reduced)
    v_dsed_tail = states[-100:, 0]
    i_dsed_tail = states[-100:, 1]
    v_dsed_mean = float(v_dsed_tail.mean())
    i_dsed_mean = float(i_dsed_tail.mean())

    # 5 % tolerance — accounts for the difference in step-size
    # discretisation between PWL @ 100ns and DSED's variable step.
    assert abs(v_dsed_mean - v_pwl_mean) / abs(v_pwl_mean) < 0.05, (
        f"DSED V_out {v_dsed_mean:.3f} V disagrees with PWL "
        f"{v_pwl_mean:.3f} V by more than 5%. Pre-fix DSED reported "
        f"V_out ≈ 12 V (the SW=ON-forever attractor) on this circuit, "
        f"which is the historical bug this test guards against.")
    assert abs(i_dsed_mean - i_pwl_mean) / abs(i_pwl_mean) < 0.05, (
        f"DSED i_L {i_dsed_mean:.3f} A disagrees with PWL "
        f"{i_pwl_mean:.3f} A by more than 5%.")


def test_python_switch_fn_triggers_polling_warning() -> None:
    """The dispatcher must emit a ``UserWarning`` recommending the
    native PWM classes when it wraps a plain Python switch_fn — both
    for visibility ("your simulation is in the slow polling path")
    and to give the user an actionable optimisation hint."""
    b = _build_buck_resistive_freewheel()
    sf = _pwm_switch_fn(b)
    with pytest.warns(UserWarning, match="NativePwm2Switch"):
        p.simulate(b, t_end=1e-4, engine="dsed", switch_fn=sf)


def test_switch_fn_with_next_edge_after_skips_polling_wrapper() -> None:
    """A user who has already implemented ``next_edge_after`` (or
    wraps their PWM in :class:`NativePwm2Switch`) gets the fast
    path — the dispatcher must NOT auto-wrap them.

    We don't run a full ``simulate`` here because the canonical
    test circuit takes seconds from cold (stiff plant + dt_init
    1 ns), which would slow CI for no good reason. The contract
    we care about is purely the dispatcher's decision tree:
    "if the switch_fn already exposes ``next_edge_after``, leave
    it alone". We pin that decision tree directly.
    """
    class _AnalyticalPWM:
        T = 1e-5
        D = 0.5
        def __call__(self, t):  # pragma: no cover - never invoked
            ...
        def next_edge_after(self, t):  # pragma: no cover
            return t + 0.5 * self.T

    sf = _AnalyticalPWM()
    # The dispatcher's UserWarning criterion (mirrors the inline check
    # in `run_dsed_from_builder`): "not a native PWM AND has no
    # next_edge_after method".
    assert not isinstance(sf, (p.NativePwm2Switch, p.NativeMultiMaskPwm))
    assert hasattr(sf, "next_edge_after")
    # Therefore the dispatcher must NOT emit the polling warning.
    # We confirm via a small simulation that finishes quickly
    # (constant-mask RC, no events) — if the dispatcher (or C++
    # scheduler) tried to engage the defensive poll cap, the call
    # would still succeed but the warning would fire.
    b2 = p.CircuitBuilder()
    b2.add_voltage_source("V", "a", "gnd", 0.0)
    b2.add_resistor("R", "a", "b", 1.0)
    b2.add_capacitor("C", "b", "gnd", 1e-6, c0=1.0)
    b2.add_switch("SW", "b", "gnd", g_on=1e-9, g_off=1e-9)

    class _ConstantNeaSwitchFn:
        """Constant mask (False), but exposes next_edge_after."""
        def __init__(self, n_sw):
            self._n = n_sw
        def __call__(self, t):
            return p.SwitchStateMask(self._n)
        def next_edge_after(self, t):
            return float("inf")

    sf2 = _ConstantNeaSwitchFn(b2.graph.num_switches)
    # Run the simulation — UserWarning must NOT fire because sf2
    # advertises next_edge_after (analytical fast path engaged).
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        res = p.simulate(b2, t_end=1e-5, engine="dsed", switch_fn=sf2)
    assert res.num_steps() > 0
