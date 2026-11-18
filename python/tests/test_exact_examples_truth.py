"""The exact engine on the shipped examples, against a truth independent of it
(add-exact-propagator-strategy, exact-engine session 7: benchmarks/EXACT_ROUTE_EVIDENCE.md).

* rc_pulse: the closed form of the RC step response.
* flyback with Q1 held ON: the same reduced mode integrated by expm in the
  test is −9.2e-8 V at the output; without balancing the eigensolver put
  1.3e-3 V there (a stiff winding row leaked into the slow modes).
* flyback driven as its header documents (100 kHz, D = 0.5): the guard read
  the clamp diode's reverse current as rounding (the 1 µΩ ground link set the
  current scale) and the run was refused by the diode-law check."""
import pathlib
import warnings

import numpy as np
import pytest

import pulsim as p

EX = pathlib.Path(__file__).resolve().parents[2] / "examples"


def _load(name):
    return p.load_yaml_file(str(EX / f"{name}.yaml"))


def test_rc_pulse_matches_the_closed_form():
    r = p.simulate(_load("rc_pulse"), engine="exact")
    t = np.asarray(r.times)
    v = np.asarray(r.v("out"))
    tau, w = 1e3 * 1e-7, 5e-4
    k = np.floor(t / w + 1e-12).astype(int)          # interval index (left limits at edges)
    k = np.where(np.isclose(t, k * w, rtol=0, atol=1e-15) & (k > 0), k - 1, k)
    v_ref = np.empty_like(t)
    v0 = 0.0
    for j in range(int(k.max()) + 1):
        u = 5.0 if j % 2 == 0 else 0.0
        m = k == j
        v_ref[m] = u + (v0 - u) * np.exp(-(t[m] - j * w) / tau)
        v0 = u + (v0 - u) * np.exp(-w / tau)
    assert np.max(np.abs(v - v_ref)) < 1e-9 * 5.0


def _q1_on(lc):
    m = p.SwitchStateMask(len(lc.switch_names))
    m.set(lc.builder.switch_index_of("Q1"), True)
    return lambda _t: m


def test_flyback_with_q1_on_keeps_the_output_at_its_leakage_level():
    out = {}
    for eng, kw in (("exact", {}), ("pwl", dict(dt=1e-8, discontinuity_restart=True,
                                                enable_substep_state_correction=True))):
        lc = _load("flyback")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(lc, t_end=2e-4, engine=eng, switch_fn=_q1_on(lc), **kw)
        out[eng] = float(np.asarray(r.v("vout"))[-1] - np.asarray(r.v("sec_neg"))[-1])
    # D1 is reverse-biased: only its 1 nS leak reaches the output (~-1e-8 V here).
    # The defect this guards put 8.3e-5 V there, ~1e6x the margin.
    assert abs(out["pwl"]) < 1e-7
    assert abs(out["exact"] - out["pwl"]) < 1e-10


def test_flyback_as_documented_runs_and_matches_the_fixed_engine():
    """Two clamp repairs (9.94 and 18.4 µs). The reference is pwl+split at 0.1 ns:
    at gate-off D1's right-limit voltage is 0.013 V, inside its 0.7 V threshold,
    and from 1 ns up the fixed engine's event iteration turns it ON there (a mode
    the hysteretic law also allows); at 0.1 ns it keeps it OFF until 0.7 V, as
    the exact engine does. Both reach the same steady state (19.808 V at 5 ms)."""
    t_end, res = 2e-5, {}
    for eng, kw in (("exact", {}), ("pwl", dict(dt=1e-10, discontinuity_restart=True,
                                                enable_substep_state_correction=True))):
        lc = _load("flyback")
        b = lc.builder
        sw = p.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=b.switch_index_of("Q1"),
                                  num_switches=len(lc.switch_names))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(lc, t_end=t_end, engine=eng, switch_fn=sw, **kw)
        t = np.asarray(r.times)
        v = np.asarray(r.v("vout")) - np.asarray(r.v("sec_neg"))
        res[eng] = p.time_average(t, v, t_from=1e-5)
    assert res["pwl"] > 0.1
    assert res["exact"] == pytest.approx(res["pwl"], rel=1e-5)


def test_phase_shift_full_bridge_output_with_balancing_where_it_hurts():
    """Balancing is kept per mode only when its eigenpairs have the smaller
    residual: on two of this bridge's modes the scaling back (up to 4e6)
    amplified the balanced eigenvectors' error, the propagator was 1.5e-3 off
    and V_out settled at 24.33 V. Truth: pwl with event splitting at dt/32,
    24.0662 V (mean over 1.8-2.0 ms, benchmarks/EXACT_ROUTE_EVIDENCE.md)."""
    lc = _load("phase_shift_full_bridge")
    si, n = lc.builder.switch_index_of, len(lc.switch_names)
    sw = p.make_phase_shift_full_bridge_fn(100e3, np.pi / 2, si("HS_A"), si("LS_A"),
                                           si("HS_B"), si("LS_B"), n, 100e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.simulate(lc, engine="exact", switch_fn=sw)
    v = p.time_average(np.asarray(r.times), np.asarray(r.v("vout")), t_from=1.8e-3)
    assert v == pytest.approx(24.0662, abs=1e-3)
