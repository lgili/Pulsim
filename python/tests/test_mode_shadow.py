"""add-complementarity-mode-selection task 4: ``mode_selection='shadow'`` on dsed.

The settle still decides (states bit-identical to the default run); the LCP is posed on the same
held state at every event and its disagreements land in ``result.diagnostics['mode_shadow']``,
each judged by propagating both candidate modes exactly from the held state (DECISIONS.md #9).
"""
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import lcp_pulsim_bridge as B

T = B.T


def _run(fam, V_th, t_end, mode=None, D=None):
    b = B.FAMILIES[fam].builder(V_th)
    kw = {}
    sws = [a[0] for k, *a in B.FAMILIES[fam].parts if k == "S"]
    if sws:
        D = D if D is not None else (0.5 if fam.startswith("boost") else 0.3)
        kw["switch_fn"] = p.make_pwm_switch_fn(1e5, D, b.switch_index_of(sws[0]), b.graph.num_switches, 0.0)
    if mode:
        kw["mode_selection"] = mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end=t_end, engine="dsed", **kw)


@pytest.mark.parametrize("fam,V_th,t_end", [("dcm_buck", 0.0, 40 * T), ("bridge1_vf07", 0.0, 0.04),
                                           ("boost_500", 0.0, 100 * T)])
def test_shadow_is_bit_identical_to_the_default(fam, V_th, t_end):
    # legacy subject: the shadow's decider is the settle (mode_selection='legacy'), no longer the default
    r0, r1 = _run(fam, V_th, t_end, "legacy"), _run(fam, V_th, t_end, "shadow")
    assert np.array_equal(np.asarray(r0.times), np.asarray(r1.times))
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))
    assert np.array_equal(np.asarray(r0.switch_states), np.asarray(r1.switch_states))
    assert np.array_equal(np.asarray(r0.event_times), np.asarray(r1.event_times))
    assert "mode_shadow" not in (getattr(r0, "diagnostics", None) or {})
    ms = r1.diagnostics["mode_shadow"]
    assert ms["events"] > 0 and ms["lcp_count"] >= ms["events"] and ms["errors"] == 0


@pytest.mark.parametrize("fam,V_th,t_end", [("dcm_buck", 1e-2, 100 * T), ("boost_50", 1e-2, 100 * T),
                                           ("bridge1_vf0", 1e-2, 0.06)])
def test_banded_circuits_report_no_disagreement(fam, V_th, t_end):
    ms = _run(fam, V_th, t_end, "shadow").diagnostics["mode_shadow"]
    assert ms["events"] > 0 and ms["errors"] == 0
    assert ms["disagreements"] == []
    assert ms["hysteresis_released"] > 0      # the band is exercised: held OFF, then released


def test_zero_threshold_boost_settle_agrees_with_the_lcp():
    """dsed's boost with V_th = 0 stayed ON at the DCM current zero (1.01 V low after 400 T);
    the shadow found those events, at each the LCP's mode consistent and the settle's not.
    Since the settle decides a located event on the settled state with the significance ladder
    (exact engine sessions 10-13) it agrees with the LCP at all 258 events of 100 T.
    Re-pinned (was: every disagreement is the settle's error)."""
    ms = _run("boost_500", 0.0, 100 * T, "shadow").diagnostics["mode_shadow"]
    assert ms["errors"] == 0 and ms["disagreements"] == []
    assert ms["events"] > 200
    assert ms["max_pivots_per_event"] <= 2


@pytest.mark.parametrize("V_th", (0.0, 1e-2))
def test_three_phase_c_filter_leak_pins_agree_with_the_settle(V_th):
    """Slice 1 measured 30 LCP errors here: at each pair turn-off the LCP's own solution keeps one
    diode ON with ~49 nA, the leak current that holds the floating DC bus; slice 1's D5 called it a
    tie and took it OFF, and the leak divider then forward-biased an OFF diode by 8.14 V. The slice-2
    D5 (DECISIONS.md #13): decided ports stay fixed while the ties are widened, and a remaining tie
    whose current pins the island (i * R_jj > tol_w) is ON when the all-ties-OFF mode fails the exact
    judge. Now 0 disagreements, 18 pins over 0.06 s (one per turn-off after the first period)."""
    ms = _run("bridge3_cfilter", V_th, 0.06, "shadow").diagnostics["mode_shadow"]
    assert ms["errors"] == 0 and ms["events"] == 37
    assert ms["disagreements"] == []
    assert ms["leak_pins"] == 18 and ms["ties"] == 0
    assert ms["pin_judgements"] >= ms["leak_pins"]


def test_shadow_counters_and_keys():
    ms = _run("dcm_buck", 0.0, 10 * T, "shadow").diagnostics["mode_shadow"]
    assert [d.split()[0] for d in ms["diodes"]] == ["D1"] and ms["h_e"] == pytest.approx(1e-7 * T)
    for k in ("lcp_pivots_total", "ties", "widenings", "settle_rounds_total", "port_model_problems"):
        assert ms[k] >= 0
    assert ms["port_model_problems"] >= ms["lcp_count"] > 0


@pytest.mark.parametrize("mode", ("shadow", "lcp"))
def test_mode_selection_runs_on_trbdf2(mode):
    """trbdf2's zero-time cascades are its decision sites (task 4.2 there):
    'shadow' is the default run byte for byte; 'lcp' decides the sites whose
    settled mode is not a clear, consistent answer."""
    b = B.FAMILIES["bridge1_vf07"].builder(0.0)
    r0 = p.simulate(b, t_end=1e-3, engine="trbdf2")
    r = p.simulate(B.FAMILIES["bridge1_vf07"].builder(0.0), t_end=1e-3, engine="trbdf2",
                   mode_selection=mode)
    key = "mode_shadow" if mode == "shadow" else "mode_selection"
    ms = r.diagnostics[key]
    assert ms["errors"] == 0 and ms["inconsistent"] == 0
    assert sum(ms["sites"].values()) > 0
    if mode == "shadow":
        assert np.asarray(r.states).tobytes() == np.asarray(r0.states).tobytes()
        assert np.asarray(r.times).tobytes() == np.asarray(r0.times).tobytes()
        assert ms["lcp_count"] == sum(ms["sites"].values())
    else:
        def bus(x):
            return np.asarray(x.v("p"))[-1] - np.asarray(x.v("n"))[-1]
        np.testing.assert_allclose(bus(r), bus(r0), rtol=1e-3)


def test_shadow_without_the_event_integrator_is_refused_on_dsed():
    """A diode-free circuit routes to the 'auto' integrator, which has no shadow hook."""
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "o", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine="dsed", mode_selection="shadow")
    assert (ei.value.engine, ei.value.feature) == ("dsed", "mode_selection")


def test_unknown_mode_selection_is_a_value_error():
    with pytest.raises(ValueError, match="mode_selection"):
        p.simulate(B.FAMILIES["bridge1_vf07"].builder(0.0), t_end=1e-3, engine="dsed", mode_selection="greedy")


def test_legacy_is_the_default():
    r0 = _run("dcm_buck", 0.0, 10 * T)
    r1 = _run("dcm_buck", 0.0, 10 * T, "legacy")
    assert np.array_equal(np.asarray(r0.states), np.asarray(r1.states))
