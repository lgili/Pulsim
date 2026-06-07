"""Regression: closed-loop switch P_cond desync in the loss/thermal summaries.

A stateful (closed-loop PI) ``switch_fn`` carries controller state that
evolves across the run. Re-evaluating it post-hoc returns the *converged*
mask, not the one in effect at time ``t``; paired with the saved
off-state terminal voltage that produced an unphysical ``v²·g_on``
conduction spike (``P_cond`` ~ 1e4 W, ``T_j`` ~ 1e4 °C) on a device
dissipating milliwatts.

Fix: :func:`pulsim._result_views.resolve_switch_closed_trace` prefers an
exact mask recorded at simulate-time (:class:`pulsim.SwitchMaskRecorder`)
and otherwise re-evaluates ``switch_fn`` with a voltage-consistency guard
that drops samples claiming ON while the device is clearly blocking.
"""
from __future__ import annotations

import warnings

import numpy as np

import pulsim as p


def _build_buck_switch(vin: float = 12.0, R_L: float = 8.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "vin", "gnd", vin)
    b.add_switch("Q", "vin", "sw", 1e3, 1e-9)              # controlled high-side
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)    # freewheel
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", R_L)
    return b


def _stages():
    return [
        p.FosterStage(R_th_K_per_W=0.18, tau_s=1e-3),
        p.FosterStage(R_th_K_per_W=0.24, tau_s=1e-2),
        p.FosterStage(R_th_K_per_W=0.18, tau_s=1e-1),
    ]


def _make_loop(b):
    pi = p.PIController(Kp=0.08, Ki=40.0, output_min=0.05, output_max=0.95)
    return p.bind_pi_to_switch(
        b, pi=pi, measured=lambda x: x[b.node_id_of("vout")],
        setpoint=5.0, switch="Q", freq=10e3,
    )


def _desync_warnings(record):
    return [w for w in record if "desync" in str(w.message)]


def test_closed_loop_conduction_is_physical():
    """The reported bug: closed-loop cold-start → ~1e4 W / 1e4 °C.

    After the fix the conduction is physical and the desync is flagged.
    """
    b = _build_buck_switch()
    loop = _make_loop(b)
    res = p.simulate(b, t_end=20e-3, dt=2e-6,
                     switch_fn=loop.switch_fn,
                     step_observer=loop.step_observer)
    # The *simulation* is correct — the bug was only in the summary.
    assert abs(float(np.mean(res.v("vout")[-1000:])) - 5.0) < 0.5

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        summ = p.device_thermal_summary(
            b, res, thermal_specs={"Q": {"stages": _stages()}},
            switch_fn=loop.switch_fn, T_ambient_C=25.0)
    # The guard must announce the desync it corrected.
    assert _desync_warnings(rec), "expected a post-hoc mask desync warning"

    q = next(d for d in summ if d["name"] == "Q")
    # Pre-fix: P_cond_avg ~ 6.8e3 W, T_j_peak ~ 5.9e3 °C.
    assert q["P_cond_avg"] < 50.0, q["P_cond_avg"]
    assert q["T_j_peak"] < 200.0, q["T_j_peak"]


def test_recorder_path_is_exact_and_silent():
    """Wrapping switch_fn in SwitchMaskRecorder uses the true historical
    masks — physical result, and no guard / no desync warning."""
    b = _build_buck_switch()
    loop = _make_loop(b)
    rec_fn = p.SwitchMaskRecorder(loop.switch_fn)
    res = p.simulate(b, t_end=20e-3, dt=2e-6,
                     switch_fn=rec_fn, step_observer=loop.step_observer)
    assert rec_fn.n_records > 0

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        summ = p.device_thermal_summary(
            b, res, thermal_specs={"Q": {"stages": _stages()}},
            switch_fn=rec_fn, T_ambient_C=25.0)
    assert not _desync_warnings(rec), "recorder path must not trip the guard"

    q = next(d for d in summ if d["name"] == "Q")
    assert q["P_cond_avg"] < 50.0, q["P_cond_avg"]
    assert q["T_j_peak"] < 200.0, q["T_j_peak"]


def test_stateless_pwm_is_untouched():
    """Open-loop fixed-duty PWM is stateless → post-hoc re-eval is already
    correct, so the guard must be a silent no-op (no desync warning)."""
    b = _build_buck_switch()
    Q_idx = b.switch_index_of("Q")
    n = b.graph.num_switches
    T_sw = 100e-6

    def switch_fn(t):
        m = p.SwitchStateMask(n)
        if (t % T_sw) / T_sw < 0.42:
            m.set(Q_idx, True)
        return m

    res = p.simulate(b, t_end=20e-3, dt=2e-6, switch_fn=switch_fn)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        summ = p.device_thermal_summary(
            b, res, thermal_specs={"Q": {"stages": _stages()}},
            switch_fn=switch_fn, T_ambient_C=25.0)
    assert not _desync_warnings(rec), (
        "stateless PWM must not trip the desync guard")
    q = next(d for d in summ if d["name"] == "Q")
    assert q["P_cond_avg"] < 50.0, q["P_cond_avg"]
