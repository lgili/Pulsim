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
    r0, r1 = _run(fam, V_th, t_end), _run(fam, V_th, t_end, "shadow")
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


def test_zero_threshold_boost_disagreements_are_the_settles_errors():
    """dsed's boost with V_th = 0 stays ON at the DCM current zero (1.01 V low after 400 T,
    test_mode_selection_today.py). The shadow finds those events; at each the LCP's mode is
    consistent over 100 h_e and 1e4 h_e and the settle's is not."""
    ms = _run("boost_500", 0.0, 100 * T, "shadow").diagnostics["mode_shadow"]
    dis = ms["disagreements"]
    assert len(dis) > 0 and ms["errors"] == 0
    for d in dis:
        assert d["lcp_consistent"] is True and d["settle_consistent"] is False, d
        assert d["settle_mode"] == (True,) and d["lcp_mode"] == (False,)
    assert ms["max_pivots_per_event"] <= 2


@pytest.mark.parametrize("V_th", (0.0, 1e-2))
def test_three_phase_c_filter_tie_rule_is_the_lcps_error(V_th):
    """Measured finding (not the design's prediction of 0): at each turn-off of the three-phase
    bridge into C || R the LCP's own solution keeps the settle's diode ON with a leak-scale
    current (~49 nA, under tol_i), so it is a tie; D5's "a remaining tie is OFF" then takes all
    diodes OFF, and the leak divider puts the floating bus where an OFF diode is forward biased by
    8.1 V. The settle's mode is consistent; this is the gauge of D10 (task 7)."""
    ms = _run("bridge3_cfilter", V_th, 0.06, "shadow").diagnostics["mode_shadow"]
    dis = ms["disagreements"]
    assert len(dis) == ms["ties"] > 0 and ms["errors"] == 0
    for d in dis:
        assert d["settle_consistent"] is True and d["lcp_consistent"] is False, d
        assert not any(d["lcp_mode"]) and sum(d["settle_mode"]) == 1
        on = d["settle_mode"].index(True)
        assert 0.0 < d["lcp_current"][on] < 1e-7
        assert d["lcp_violation"] == pytest.approx(8.14, abs=0.01)


def test_shadow_counters_and_keys():
    ms = _run("dcm_buck", 0.0, 10 * T, "shadow").diagnostics["mode_shadow"]
    assert [d.split()[0] for d in ms["diodes"]] == ["D1"] and ms["h_e"] == pytest.approx(1e-7 * T)
    for k in ("lcp_pivots_total", "ties", "widenings", "settle_rounds_total", "port_model_problems"):
        assert ms[k] >= 0
    assert ms["port_model_problems"] >= ms["lcp_count"] > 0


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "auto"))
def test_shadow_refused_by_name_off_dsed(engine):
    b = B.FAMILIES["bridge1_vf07"].builder(0.0)
    kw = {"dt": 1e-6} if engine == "pwl" else {}
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine=engine, mode_selection="shadow", **kw)
    assert ei.value.feature == "mode_selection" and ei.value.engine in ("pwl", "trbdf2")


@pytest.mark.parametrize("engine", ("pwl", "trbdf2", "dsed"))
def test_lcp_refused_by_name_on_every_engine(engine):
    b = B.FAMILIES["bridge1_vf07"].builder(0.0)
    with pytest.raises(p.UnsupportedError) as ei:
        p.simulate(b, t_end=1e-3, engine=engine, mode_selection="lcp", **({"dt": 1e-6} if engine == "pwl" else {}))
    assert (ei.value.engine, ei.value.feature) == (engine, "mode_selection")


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
