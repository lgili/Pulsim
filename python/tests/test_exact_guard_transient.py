"""The default-mode guard judges what the projection skips (exact engine).

The slow-manifold projection jumps over a mode's sub-ns transient, so a diode law broken
only inside it is invisible from the projected state; and a cut-set re-initialisation is
where the value settle cannot see a diode's bias at all. Two wrong answers came from it:

* a freewheel diode with g_off = 0 stayed OFF at every gate-off: the switch cut the
  inductor (a cut-set record, the current destroyed) instead of handing it to the diode;
* the forward converter kept its rectifier ON at gate-off and drained L_o's 2.7 A through
  2 MΩ of g_off in 23 ps; repaired, the open transformer's projection then left the
  rectifier forward-biased by the freewheel drop, settled only after the projection.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]


def _freewheel(series):
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "vin", "gnd", 10.0)
    b.add_switch("S", "vin", "a", 1e6, 0.0)
    if series is not None:
        # g_off = 0: all OFF leaves node a with no conductance (the settle reached a mode
        # the extraction cannot pose); 1e-9: the settle's cut-set trial mode left records
        # that the pass loop drops (settle_and_project_), the guard then repairs
        b.add_diode("Ds", "a", "a2", 1e3, series, 0.0)
        b.add_resistor("R", "a2", "b", 10.0)
    else:
        b.add_resistor("R", "a", "b", 10.0)
    b.add_inductor("L", "b", "gnd", 1e-3)
    b.add_diode("D", "gnd", "a", 1e3, 0.0, 0.0)
    return b, p.make_pwm_switch_fn(1e3, 0.5, b.switch_index_of("S"), b.graph.num_switches)


def _i_closed_form(t, series):
    # ON: 10 V across 10 Ω + 1 µΩ (+ 1 mΩ); OFF: the diode's 1 mΩ freewheels the current
    r_on, r_off, L, T = 10.0 + 1e-6, 10.0 + 1e-3, 1e-3, 1e-3
    if series is not None:
        r_on, r_off = r_on + 1e-3, r_off + 1e-3
    i, t0 = 0.0, 0.0
    while t0 + T / 2 <= t + 1e-15:
        i = 10.0 / r_on + (i - 10.0 / r_on) * np.exp(-(T / 2) * r_on / L)
        dt_off = min(T / 2, t - t0 - T / 2)
        i *= np.exp(-dt_off * r_off / L)
        if dt_off < T / 2:
            return i
        t0 += T
    return 10.0 / r_on + (i - 10.0 / r_on) * np.exp(-(t - t0) * r_on / L)


@pytest.mark.parametrize("series", [None, 0.0, 1e-9])
@pytest.mark.parametrize("mode", [None, "lcp"])
def test_a_g_off_zero_freewheel_diode_takes_the_inductor_current(mode, series):
    b, fn = _freewheel(series)
    kw = {} if mode is None else {"mode_selection": mode}
    r = p.simulate(b, 3e-3, engine="exact", dt=1e-6, switch_fn=fn, **kw)
    assert len(r.reinit_records) == 0          # was one cut-set record per gate-off
    t, i = np.asarray(r.times), np.asarray(r.i("L"))
    for tq in (4e-4, 9e-4, 1.25e-3, 2.95e-3):
        k = int(np.argmin(np.abs(t - tq)))
        assert i[k] == pytest.approx(_i_closed_form(t[k], series), rel=1e-7, abs=1e-12)


def test_a_trial_cut_set_does_not_annihilate_the_freewheel_current():
    # At gate-off the settle's first trial is the all-OFF cut-set (i_L -> 0); the
    # re-settle may then turn D ON on that projection's 1e-18 A residue, a sign
    # left to rounding (Windows CI at R = 10; here R a few ulp off 10: 6 of these 12
    # failed a gate-off on macOS). The settled mode kept the trial's i_L = 0, no record.
    for m in (1 - 2**-53, 1 + 1e-14, 1 - 1e-14, 1 + 3e-15, 1 - 3e-15, 1 + 7e-15,
              1 - 7e-15, 1 + 2e-14, 1 - 2e-14, 1 + 5e-14, 1 - 5e-14, 1 + 1e-13):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "vin", "gnd", 10.0)
        b.add_switch("S", "vin", "a", 1e6, 0.0)
        b.add_resistor("R", "a", "b", 10.0 * m)
        b.add_inductor("L", "b", "gnd", 1e-3)
        b.add_diode("D", "gnd", "a", 1e3, 0.0, 0.0)
        fn = p.make_pwm_switch_fn(1e3, 0.5, b.switch_index_of("S"), b.graph.num_switches)
        r = p.simulate(b, 3e-3, engine="exact", dt=1e-6, switch_fn=fn)
        assert len(r.reinit_records) == 0
        t, i = np.asarray(r.times), np.asarray(r.i("L"))
        for tq in (9e-4, 1.9e-3, 2.9e-3):   # 0.4 ms after each gate-off
            k = int(np.argmin(np.abs(t - tq)))
            # R moves the closed form by ~1e-13 relative: far inside rel=1e-7
            assert i[k] == pytest.approx(_i_closed_form(t[k], None), rel=1e-7), (m, tq)


def test_the_forward_converter_rectifier_follows_the_transformer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    case = v1_reader.load(ROOT / "benchmarks/circuits/switching/isolated/forward_converter.yaml")
    case.t_end = 3e-5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = case.run("trbdf2", rtol=1e-9, atol=1e-12)
        r = case.run("exact")
    assert r.diagnostics["mode_guard"]["repairs"] >= 1
    # the end value to the reference's own accuracy (measured 2.8e-8), the trace within
    # the interpolation between sparse event-driven samples (5.5e-6; was err_rms 0.648)
    assert r.v("out")[-1] == pytest.approx(ref.v("out")[-1], rel=1e-6)
    assert r.i("L_o")[-1] == pytest.approx(ref.i("L_o")[-1], rel=1e-6)
    tr, t = np.asarray(ref.times), np.asarray(r.times)
    v = np.interp(tr, t, np.asarray(r.v("out")))
    assert np.max(np.abs(v - np.asarray(ref.v("out")))) < 5e-5 * np.max(np.abs(ref.v("out")))


def _gated_bridge():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "s", "gnd", 0.0, 100.0, 50.0)
    b.add_resistor("Rs", "s", "ac", 1e-3)
    for nm, an, ca in (("D1", "ac", "p"), ("D3", "gnd", "p"), ("D2", "n", "ac"), ("D4", "n", "gnd")):
        b.add_diode(nm, an, ca, 1e3, 1e-9, 0.0, 0.0)
    b.add_capacitor("C", "p", "n", 1e-4)
    b.add_resistor("R", "p", "n", 100.0)
    b.add_switch("S", "p", "q", 1e3, 1e-9)
    b.add_resistor("Rl", "q", "n", 5.0)
    return b, p.make_pwm_switch_fn(1e4, 0.5, b.switch_index_of("S"), b.graph.num_switches)


def test_a_gated_bridge_from_rest_turns_its_tie_on_under_the_lcp():
    # From rest the LCP pinned D4 ON and left the tie D1 OFF; D1's signal then rose
    # from exactly 0 (its event side) and never fired: OFF and forward-biased by
    # 0.314 V at 1e-5 s (DiodeModeInconsistentError). release_band_ties_ decides it.
    res = {}
    for mode in (None, "lcp"):
        b, fn = _gated_bridge()
        kw = {} if mode is None else {"mode_selection": mode}
        res[mode] = p.simulate(b, 2e-3, engine="exact", dt=1e-6, switch_fn=fn, **kw)
    for r in res.values():
        S = np.asarray(r.switch_states)
        assert S[:, b.switch_index_of("D1")].any()  # D1 conducts
    vb = {m: np.asarray(r.v("p")) - np.asarray(r.v("n")) for m, r in res.items()}
    assert vb["lcp"][-1] == pytest.approx(vb[None][-1], rel=1e-9)


@pytest.mark.filterwarnings("ignore:simulate\\(engine='dsed'\\). no switch_fn")
def test_a_tied_bridge_with_a_forward_drop_conducts_after_the_guard_repairs():
    # The guard's second repair left D1 OFF 3e-7 V above V_f, rising: on its
    # predicate's event side, inside the re-settle's 1e-6·scale band, so D1 stayed
    # OFF while forward-biased by 1.1 V (DiodeModeInconsistentError).
    res = {}
    for mode in (None, "lcp"):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 325.0, 50.0)
        b.add_resistor("Rs", "s", "a", 0.5)
        b.add_inductor("Ls", "a", "ac1", 1e-3)
        for nm, an, ca in (("D1", "ac1", "dp"), ("D2", "gnd", "dp"), ("D3", "dn", "ac1"), ("D4", "dn", "gnd")):
            b.add_diode(nm, an, ca, 1e3, 1e-9, 0.0, 0.7)
        b.add_capacitor("C", "dp", "dn", 470e-6)
        b.add_resistor("RL", "dp", "dn", 100.0)
        b.add_resistor("Rtie", "dn", "gnd", 1e6)
        kw = {} if mode is None else {"mode_selection": mode}
        res[mode] = p.simulate(b, 0.01, engine="dsed", **kw)
    vb = {m: np.asarray(r.v("dp")) - np.asarray(r.v("dn")) for m, r in res.items()}
    for v in vb.values():
        assert v.max() > 250.0                     # the bus charged through D1, D4
    assert vb["lcp"][-1] == pytest.approx(vb[None][-1], rel=1e-9)


@pytest.mark.filterwarnings("ignore:simulate\\(engine='dsed'\\). no switch_fn")
@pytest.mark.parametrize("mode", [None, "lcp"])
def test_the_phase_shift_full_bridge_starts_inside_the_zero_band(mode):
    # At t = 0 the leak divider puts 5e-15 V across an OFF diode at 100 V: inside the
    # engine's zero band (1e-12 of the largest node voltage), not a breach. The check
    # used an absolute 1e-12 V floor and refused the run. Run as the route survey runs
    # the shipped example (no switch_fn: every switch closed).
    kw = {} if mode is None else {"mode_selection": mode}
    # The breach was at t = 0, so a short run covers it (the full run takes 3 s per mode).
    r = p.simulate(p.load_yaml_file(str(ROOT / "examples/phase_shift_full_bridge.yaml")),
                   2e-5, engine="exact", **kw)
    assert np.all(np.isfinite(np.asarray(r.states)))
