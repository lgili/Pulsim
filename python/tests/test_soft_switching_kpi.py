"""Soft-switching KPI columns against the exact truth (OpenSpec
``add-soft-switching-validation`` 2.2, 2.3, 4.2, 5.2, 5.3).

Truth: ``benchmarks/tools/soft_switching_truth.py`` on ``exact_truth.py``
(mpmath, 30 digits, never imports pulsim): per switch, the left limit of
v_DS at every turn-on, the ZVS fraction, and the switching energy the KPI
defines -- the sum over every gate edge of int v_DS i_D dt = int g v_DS^2 dt
over ``window_s`` after it, exact (window mean and RMS). The baselines the
runner compares with (``benchmarks/baselines/<id>.csv.gz``) are the same
truth on the netlist grid.

Measured: the runner's ``kpi__*`` columns (``benchmark_runner.run_benchmarks``
on ``benchmarks/manifests/soft_switching.yaml``: ``default`` = the plain
fixed engine, ``variable`` = trbdf2) and the same KPI functions on the fixed
engine with the restart rule and on the exact engine, read through
``pulsim.with_event_limits`` (both limits of every event).

Bounds:
* ZVS fraction: equal (a classification). The thresholds leave a margin on
  the truth itself: soft edges at |v_DS| <= 5 mV against 5 V (LLC, PSFB),
  the buck's lowest hard edge 1.42 V against 1 V.
* Turn-on count: equal over the edges before t_end - dt. An edge in the last
  step (the LLC's S_hi at 40 us, the PSFB's S1 at 30 us: the exact rational
  and the float t_end differ by ~3e-21 s) is in or out of the truth and of
  each engine's trace depending on that rounding (trbdf2 counts S1's).
* v_DS at each turn-on: dsed 1e-9 V (the exact engine; 2.5e-14 V seen),
  trbdf2 1e-4 V (default rtol 1e-3 on ~5 A at 200 S..2 kS: <= 2.5e-5 V;
  1.3e-5 V seen), pwl+restart 1e-3 V (second order at h = dt; 1.3e-4 V seen).
* Switching energy: the KPI is the trapezoid of v i over the engine's own
  samples, so it carries the quadrature error of that grid even with exact
  states (trbdf2 on the PSFB's S3: +15 % against the exact energy, all of it
  quadrature: about two samples in a 50 ns window). The truth measures that
  part (the same windows, trapezoid over the truth sampled at the engine's
  times, its first point extrapolated to the edge from 10 and 20 ps after
  it -- the 0.2-0.4 ps C_oss collapse at a turn-on is a jump to every trace
  with event rows), and the
  KPI is held to it: |E_kpi - E_trapz| <= tol E_exact, tol = dsed 3e-5
  (its states are exact to ~1e-10 V; what is left is the truth's first point,
  extrapolated over the collapse: 8.5e-6 seen on the PSFB's S3), trbdf2 1e-3
  (its rtol; 1.6e-4 seen), pwl+restart 1e-2 (the ON window integrates
  g_on v^2 with v ~ 1 mV: 2 dv/v at the 5.4e-6 V second-order state error
  seen = 1e-2; 3.7e-3 seen); the
  plain fixed engine is held to the restart rule's 1e-2 (the buck meets it,
  3.1e-3; LLC and PSFB do not).
The plain fixed engine (no restart rule) is trap 9: strict xfails with the
numbers it gives.
"""
from __future__ import annotations

import functools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim
import pulsim._result_views as rv

mp = pytest.importorskip("mpmath")
pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import benchmark_runner as br  # noqa: E402
import exact_truth as et  # noqa: E402
import soft_switching_truth as sst  # noqa: E402
import v1_reader  # noqa: E402
from kpi import compute_switching_loss_per_event, compute_zvs_fraction  # noqa: E402

SW = ROOT / "circuits" / "switching"
# id -> (netlist, [(switch, gate column, V_DS column)], window_s, threshold_v)
CASES = {
    "llc_half_bridge_zvs": (SW / "isolated" / "llc_half_bridge_zvs.yaml",
                            [("S_hi", "V(g_hi)", "V(vin,mid)"), ("S_lo", "V(g_lo)", "V(mid)")], 100e-9, 5.0),
    "psfb_full_bridge_zvs": (SW / "isolated" / "psfb_full_bridge_zvs.yaml",
                             [("S1", "V(g1)", "V(vin,a)"), ("S2", "V(g2)", "V(a)"),
                              ("S3", "V(g3)", "V(vin,b)"), ("S4", "V(g4)", "V(b)")], 50e-9, 5.0),
    "buck_hard_switching_reference": (SW / "buck" / "buck_hard_switching_reference.yaml",
                                      [("SH", "V(gate)", "V(vcc,sw)")], 1e-6, 1.0),
}
ENGINES = {"pwl": "runner default", "trbdf2": "runner variable",
           "pwl_restart": dict(engine="pwl", discontinuity_restart=True), "dsed": dict(engine="dsed"),
           # the plain fixed engine (trap 9) is no longer the runner default: its own cell
           "pwl_plain": dict(engine="pwl", discontinuity_restart=False)}
V_TOL = {"dsed": 1e-9, "trbdf2": 1e-4, "pwl_restart": 1e-3}
E_TOL = {"dsed": 3e-5, "trbdf2": 1e-3, "pwl_restart": 1e-2, "pwl": 1e-2, "pwl_plain": 1e-2}
TRAP9 = ("known: trap 9 -- the plain fixed engine books each commutation one step early and rings on "
         "the switch node; measured: {}")
KNOWN_PWL = {   # (ZVS / turn-on voltage, switching energy): the first assertion each fails
    "llc_half_bridge_zvs": (TRAP9.format("v_DS at S_lo's 4 us turn-on 0.918 V vs the truth's -0.0011 V"),
                            TRAP9.format("S_hi energy 9.250e-11 J vs exact 7.022e-11 J (+32 %; S_lo +62 %)")),
    "psfb_full_bridge_zvs": (TRAP9.format("v_DS at S1's 10 us turn-on -0.00059 V vs -0.00462 V; ZVS fraction "
                                          "of S3 and S4 2/3 (8.53 V at S3's 8 us turn-on)"),
                             TRAP9.format("S1 energy 1.046e-9 J vs exact 5.702e-10 J (+84 %)")),
    "buck_hard_switching_reference": (TRAP9.format("ZVS fraction 0.04 (1 of 25 turn-ons read soft) vs the "
                                                   "truth's 0 of 24"), None),
}


# The restart rule's right-limit row at a turn-on is the model's own x(t_e+) (C_oss voltage continuous,
# the ~0.3 ps collapse recorded); _engine_trace adds the record's post-collapse row one ulp later
# (with_right_limits(collapse=True)), so the trapezoid no longer carries the pre-collapse row over the next step (LLC S_hi was +12 %,
# PSFB S1 +29 %; now -0.34 % / -0.03 %). The PSFB's lagging leg: see the xfail's numbers.
KNOWN_RESTART = {
    "psfb_full_bridge_zvs": ("known: the restart rule's lagging-leg energy reads low; measured: S3 energy "
                             "3.051e-11 J vs the truth's trapezoid 3.380e-11 J (-10 %; exact 3.178e-11, -4 %). "
                             "The engine's grid rows meet the truth to <= 5 uV; the reader's collapse row "
                             "holds the step's END state (-3.63e-4 V at 8.00 us+, truth's post-collapse "
                             "~-5.6e-4 V) over the first step after each turn-on: S3 turn-ons -2.8 / -4.1 / "
                             "-8.6 % per event at 8 / 18 / 28 us; the post-collapse x(t_e+) + J is not exported"),
}


@functools.lru_cache(maxsize=None)
def _sol(case):
    return et.Solver(et.Circuit(str(CASES[case][0]), "conductance"), 30).run()


@functools.lru_cache(maxsize=None)
def _truth(case, switch, grid=None):
    _, _, w, thr = CASES[case]
    return sst.switch_kpi_truth(_sol(case), switch, w, thr, grid=None if grid is None else list(grid))


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    out = tmp_path_factory.mktemp("softsw")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows = br.run_benchmarks(ROOT / "manifests" / "soft_switching.yaml", out)
    br.write_results(out, rows)
    return out, {(r.benchmark_id, r.scenario): r for r in rows}


@functools.lru_cache(maxsize=None)
def _engine_trace(case, engine):
    """(t, {switch: (gate, v_DS, i_D)}) as the runner reads its CSV: the first
    sample dropped (an algebraic node's first sample is not solved), every
    event with both limits."""
    path, sws, _, _ = CASES[case]
    kw = dict(ENGINES[engine])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = v1_reader.load(path).run(kw.pop("engine"), **kw)
        # a reinit record's collapse at t_e+, as the engine ran the step (not a line from x(t_e+))
        r = rv.with_right_limits(rv.with_inner_events(r), collapse=True)
    obs = lambda c: np.asarray(v1_reader.observable(r, c), dtype=float)[1:]  # noqa: E731
    return (np.asarray(r.times, dtype=float)[1:],
            {s: (obs(g) > 0.5, obs(v), obs(f"I({s})")) for s, g, v in sws})


def _measured(runner, case, engine):
    """(t, traces, {switch: kpis}) for one engine; the runner's own columns for
    its two scenarios."""
    _, sws, w, thr = CASES[case]
    if engine in ("pwl", "trbdf2"):
        out, rows = runner
        row = rows[(case, "default" if engine == "pwl" else "variable")]
        t, s = br.load_csv_series(out / "outputs" / case / row.scenario / "pulsim.csv")
        t = np.asarray(t[1:])
        tr = {sw: (np.asarray(s[g][1:]) > 0.5, np.asarray(s[v][1:]), np.asarray(s[f"I({sw})"][1:]))
              for sw, g, v in sws}
        k = {sw: {key.split("__")[1]: val for key, val in row.kpis.items() if key.endswith(f"__{sw}")}
             for sw, _, _ in sws}
        return t, tr, k
    t, tr = _engine_trace(case, engine)
    k = {}
    for sw, (g, v, i) in tr.items():
        k[sw] = {**compute_zvs_fraction(list(g), list(v), thr, 1),
                 **compute_switching_loss_per_event(list(g), list(v), list(i), list(t), w)}
    return t, tr, k


# The runner default under the release (restart rule + event split). Its CSV carries the grid rows
# only, no record collapse row, so the trapezoid holds each turn-on's pre-collapse row over the next
# step (the engine itself, read with collapse rows as _engine_trace reads it: LLC S_hi -0.4 %, PSFB S3
# -9.7 %, S1 -0.0 %: the restart cell's numbers). Measured on the runner's rows:
_CSV = "known: the runner's CSV has no collapse rows (with_right_limits(collapse=True)); measured: {}"
KNOWN_DEFAULT = {
    "llc_half_bridge_zvs": (None, _CSV.format("S_hi energy 7.444e-11 J vs the truth's trapezoid 7.025e-11 J (+6.0 %)")),
    "psfb_full_bridge_zvs": (None, _CSV.format("S3 energy 5.672e-11 J vs the truth's trapezoid 3.396e-11 J (+67 %)")),
}


KNOWN_DSED = {"psfb_full_bridge_zvs": (
    "release finding F-Z-9: the exact engine's LCP mode selection ('auto' -> 'lcp') on branch-form diodes (the "
    "release's form default) leaves D7 of the parallel D6/D7 pair OFF at the 7.92 us rectifier commutation; v_DS at "
    "S1's 10 / 20 us turn-on 1.99e-9 / 3.12e-9 V off the truth (V_TOL 1e-9); the settle or form='conductance' "
    "7.3e-11 / 1.6e-10 V")}


def _cells(which):
    for case in CASES:
        for eng in ENGINES:
            why = (KNOWN_PWL[case][which] if eng == "pwl_plain"
                   else KNOWN_DEFAULT.get(case, (None, None))[which] if eng == "pwl"
                   else KNOWN_RESTART.get(case) if eng == "pwl_restart" and which == 1
                   else KNOWN_DSED.get(case) if eng == "dsed" and which == 0 else None)
            marks = [pytest.mark.xfail(strict=True, reason=why)] if why else []
            yield pytest.param(case, eng, marks=marks, id=f"{case}-{eng}")


def test_truth_meets_the_design_targets():
    """The spec's targets on the truth itself: LLC and PSFB ZVS on every
    edge with |v_DS| <= 5 mV (the body diode's -i/g_on), the buck hard on
    every edge (>= 1.42 V); the LLC's average switching loss (both switches)
    below 1 % of its output power (exact: int V(p)^2 / R_ac dt / t_end)."""
    for case, (_, sws, _, _) in CASES.items():
        for sw, _, _ in sws:
            tr = _truth(case, sw)
            v = np.abs(tr["v_ds_at_turn_on"])
            if case.startswith("buck"):
                assert tr["zvs_fraction"] == 0.0 and v.min() > 1.4 and tr["turn_on_events"] == 24
            else:
                assert tr["zvs_fraction"] == 1.0 and v.max() < 5e-3, (case, sw, v)
    # the numbers docs/SOFT_SWITCHING.md publishes
    exact = {("llc_half_bridge_zvs", "S_hi"): 7.022e-11, ("llc_half_bridge_zvs", "S_lo"): 7.280e-11,
             ("psfb_full_bridge_zvs", "S1"): 5.702e-10, ("psfb_full_bridge_zvs", "S2"): 8.527e-10,
             ("psfb_full_bridge_zvs", "S3"): 3.178e-11, ("psfb_full_bridge_zvs", "S4"): 3.306e-11,
             ("buck_hard_switching_reference", "SH"): 7.406e-6}
    for (case, sw), e in exact.items():
        assert _truth(case, sw)["switching_energy_j"] == pytest.approx(e, rel=1e-3), (case, sw)
    assert max(np.abs(_truth("llc_half_bridge_zvs", s)["v_ds_at_turn_on"]).max() for s in ("S_hi", "S_lo")) < 1.2e-3
    vb = np.asarray(_truth("buck_hard_switching_reference", "SH")["v_ds_at_turn_on"])
    assert np.all((vb[:10] > 24.01) & (vb[:10] < 24.09)) and np.all((vb[10:] > 1.41) & (vb[10:] < 6.02)), vb
    sol = _sol("llc_half_bridge_zvs")
    t_end = mp.mpf(40e-6)
    p_out = sol.window(["V(p)"], mp.mpf(0), t_end)["V(p)"]["rms"] ** 2 / 150
    p_sw = sum(_truth("llc_half_bridge_zvs", s)["switching_energy_j"] for s in ("S_hi", "S_lo")) / 40e-6
    assert p_sw < 0.01 * float(p_out), (p_sw, float(p_out))


@pytest.mark.parametrize("case,engine", list(_cells(0)))
def test_zvs_kpi_and_turn_on_voltage(runner, case, engine):
    _, sws, _, _ = CASES[case]
    t, traces, kpis = _measured(runner, case, engine)
    dt = v1_reader.load(CASES[case][0]).dt
    for sw, _, _ in sws:
        tr = _truth(case, sw)
        gate, v, _ = traces[sw]
        on = [k for k in range(1, len(t)) if gate[k] and not gate[k - 1]]
        assert kpis[sw]["total_turn_on_events"] == len(on), sw
        assert kpis[sw]["zvs_fraction"] == tr["zvs_fraction"], sw
        on = [k for k in on if t[k] < tr["t_end"] - dt]
        assert len(on) == sum(1 for x in tr["t_on"] if x < tr["t_end"] - dt), (sw, len(on))
        for k, te, want in zip(on, tr["t_on"], tr["v_ds_at_turn_on"]):
            # the same edge (an edge on a grid point is first read on the next row: t[k] - te = dt + 1.6e-21 s)
            assert abs(t[k] - te) <= dt * (1 + 1e-9), (sw, t[k], te)
            assert abs(v[k - 1] - want) <= V_TOL.get(engine, 1e-3), (sw, te, v[k - 1], want)


@pytest.mark.parametrize("case,engine", list(_cells(1)))
def test_switching_energy_kpi(runner, case, engine):
    _, sws, _, _ = CASES[case]
    t, _, kpis = _measured(runner, case, engine)
    for sw, _, _ in sws:
        tr = _truth(case, sw, tuple(t))
        exact, trapz = tr["switching_energy_j"], tr["switching_energy_trapz_j"]
        got = kpis[sw]["switching_energy_j"]
        tol = E_TOL[engine]
        assert abs(got - trapz) <= tol * exact, (sw, got, trapz, exact)
        # the average column: the runner divides by the trace's span (its first sample dropped)
        span = t[-1] - t[0]
        assert abs(kpis[sw]["switching_loss_w_avg"] - trapz / span) <= tol * exact / span, sw


def test_runner_writes_the_kpi_columns_and_the_benches_pass(runner):
    """5.3: results.csv (the dashboard's table) carries a kpi__zvs_fraction__,
    kpi__switching_energy_j__ and kpi__switching_loss_w_avg__ column per switch,
    and every row the engines get right passes its reference validation
    against the truth baseline."""
    out, rows = runner
    head = (out / "results.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    for case, (_, sws, _, _) in CASES.items():
        for sw, _, _ in sws:
            for m in ("zvs_fraction", "switching_energy_j", "switching_loss_w_avg"):
                assert f"kpi__{m}__{sw}" in head, (case, sw, m)
    status = {k: (r.status, r.max_error) for k, r in rows.items()}
    assert status[("buck_hard_switching_reference", "default")][0] == "passed", status
    assert status[("buck_hard_switching_reference", "variable")][0] == "passed", status
    # the LLC / PSFB max_error = the baseline's linear interpolation, max (h^2/8)|v''| over its
    # rows from 50 ns, + the 0.5 V per-sample bound of the dead-time tests
    for case in ("llc_half_bridge_zvs", "psfb_full_bridge_zvs"):
        bm = br.load_yaml(CASES[case][0])["benchmark"]
        t, ser = br.load_csv_series(ROOT / "baselines" / f"{case}.csv.gz")
        t, v = np.asarray(t), np.asarray(ser[bm["validation"]["observable"]])
        m = t >= 50e-9
        t, v = t[m], v[m]
        h1, h2 = np.diff(t)[:-1], np.diff(t)[1:]
        ok = (h1 > 1e-12) & (h2 > 1e-12)            # not across an event's two limits
        d2 = 2 * ((v[2:] - v[1:-1]) / h2 - (v[1:-1] - v[:-2]) / h1) / (h1 + h2)
        interp = float(np.max((np.maximum(h1, h2) ** 2 / 8 * np.abs(d2))[ok]))
        bound = float(bm["expectations"]["metrics"]["max_error"])
        assert interp + 0.5 <= bound < interp + 0.55, (case, interp, bound)
    for case in ("llc_half_bridge_zvs", "psfb_full_bridge_zvs"):
        assert status[(case, "variable")][0] == "passed", status
        # the runner default is the restart rule + split under the release (trap 9 was the plain
        # engine's, > 200 V seen): it passes (LLC max_error 0.0806 V measured)
        assert status[(case, "default")][0] == "passed", status


@pytest.mark.xfail(strict=True, reason="F-S-5: trbdf2 starts V(b) of the PSFB at 599.8 V (3e-16 s), 200 V off "
                                       "the consistent 400 V start that the exact truth and dsed keep")
def test_trbdf2_keeps_the_psfb_consistent_start():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = v1_reader.load(CASES["psfb_full_bridge_zvs"][0]).run("trbdf2")
    t, v = np.asarray(r.times), np.asarray(r.v("b"))
    assert np.max(np.abs(v[t < 1e-9] - 400.0)) < 1.0


def test_collapse_rows_follow_each_record_step():
    """with_right_limits(collapse=True): after a record's step start (with or
    without a right-limit row there) one row holding the step's end sample;
    times strictly increasing, every per-row array the same length."""
    import types
    from pulsim._result_views import with_right_limits
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    st = np.arange(10.0).reshape(5, 2)
    sw = np.array([[0], [1], [0], [1], [0]], dtype=bool)
    rec = lambda k: types.SimpleNamespace(sample=k, t=t[k - 1])  # noqa: E731
    r = types.SimpleNamespace(times=t, states=st, switch_states=sw, right_limit_index=np.array([1]),
                              right_limit_states=np.array([[-1.0, -1.0]]),
                              right_limit_switch_states=np.array([[1]], dtype=bool),
                              reinit_records=[rec(2), rec(4)])   # steps (1, 2] (right limit at 1) and (3, 4]
    v = with_right_limits(r, collapse=True)
    assert np.all(np.diff(v.times) > 0) and len(v.times) == len(v.states) == len(v.switch_states) == 8
    assert v.times[2] == np.nextafter(1.0, np.inf) and v.states[2].tolist() == [-1.0, -1.0]
    assert v.times[3] == np.nextafter(v.times[2], np.inf) and v.states[3].tolist() == st[2].tolist()
    assert v.times[6] == np.nextafter(3.0, np.inf) and v.states[6].tolist() == st[4].tolist()
    assert v.switch_states[3].tolist() == sw[2].tolist()
    assert len(with_right_limits(r).times) == 6   # without collapse: one row per right limit


# ---- F-Z-9: the LCP on branch-form rectifier diodes ------------------------------------------
# At the PSFB rectifier commutations (7.92 / 17.92 / 27.92 us, D5 and D8 ON) D6 and D7 turn on
# together and share the current equally (equal R_on, symmetric bridge). The selector's widened
# pass was posed with every tie OFF: the output inductor's 13 A into g_off put the port voltages
# at ~7e9 V, and the 7.8e-7 V gap it computed for D7 (rounding of that network) was read as a
# decision: D7 OFF for 55 ns, I(D6) 4.66 A where 2.33 A is true.

def _psfb_exact_lcp(form, monkeypatch):
    import pulsim
    for m in ("add_diode", "add_switch"):   # the reader takes the library's default form
        f = getattr(pulsim.CircuitBuilder, m)
        monkeypatch.setattr(pulsim.CircuitBuilder, m,
                            lambda self, *a, _f=f, **k: _f(self, *a, **{"form": form, **k}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return v1_reader.load(CASES["psfb_full_bridge_zvs"][0]).run("dsed", mode_selection="lcp")


def _rectifier_pair_split(r):
    """Per commutation: (samples, worst |I(D6) - I(D7)| / I(D6)) from 5 to 50 ns after it."""
    t = np.asarray(r.times, dtype=float)
    i6, i7 = (np.asarray(v1_reader.observable(r, f"I({d})"), dtype=float) for d in ("D6", "D7"))
    out = []
    for t0 in (7.92053e-6, 17.92049e-6, 27.92046e-6):
        w = (t > t0 + 5e-9) & (t < t0 + 50e-9)
        out.append((int(w.sum()), float(np.max(np.abs(i6[w] - i7[w]) / i6[w])) if w.any() else np.inf,
                    float(np.min(i6[w])) if w.any() else 0.0))
    return out


@pytest.mark.parametrize("form", ["conductance", "branch"])
def test_exact_lcp_turns_on_both_parallel_rectifier_diodes(form, monkeypatch):
    r = _psfb_exact_lcp(form, monkeypatch)
    for n, rel, i_min in _rectifier_pair_split(r):
        assert n >= 3 and i_min > 1e-3 and rel <= 1e-6, (form, n, rel, i_min)
    ms = r.diagnostics["mode_selection"]
    assert ms["leak_pins"] == 0 and ms["capped"] == 0, ms
    if form == "branch":
        assert ms["rounding_ties"] >= 1, ms     # the floor is what decided it (3 measured)


def test_exact_lcp_branch_form_psfb_turn_on_voltage(monkeypatch):
    """The turn-on KPI that exposed F-Z-9 (+1.991e-9 / +3.121e-9 V against the 1e-9 V bar): on
    branch-form devices with 'lcp' it is the conductance form's again."""
    import pulsim
    for m in ("add_diode", "add_switch"):
        f = getattr(pulsim.CircuitBuilder, m)
        monkeypatch.setattr(pulsim.CircuitBuilder, m,
                            lambda self, *a, _f=f, **k: _f(self, *a, **{"form": "branch", **k}))
    monkeypatch.setitem(ENGINES, "dsed_branch_lcp", dict(engine="dsed", mode_selection="lcp"))
    t, tr = _engine_trace("psfb_full_bridge_zvs", "dsed_branch_lcp")
    g, v, _ = tr["S1"]
    truth = _truth("psfb_full_bridge_zvs", "S1")["v_ds_at_turn_on"]
    on = [k for k in range(1, len(t)) if g[k] and not g[k - 1]]
    err = [v[k - 1] - w for k, w in zip(on, truth)]
    assert len(err) == 2 and max(abs(e) for e in err) <= V_TOL["dsed"], err


def test_exact_lcp_without_the_rounding_floor_brings_f_z_9_back():
    """The floor is the fix: with it off (a test hook, fresh process: it is read once) the
    selector decides D7 OFF on its rounding gap again at each commutation (3 leak pins) and
    D6 carries the pair's current (measured |I6 - I7| / I6 ~ 1)."""
    import json
    import os
    import subprocess
    code = (
        "import json, sys, warnings\n"
        f"sys.path[:0] = {[str(ROOT / 'tools'), str(ROOT)]!r}\n"
        "import numpy as np, pulsim, v1_reader\n"
        "for m in ('add_diode', 'add_switch'):\n"
        "    f = getattr(pulsim.CircuitBuilder, m)\n"
        "    setattr(pulsim.CircuitBuilder, m, lambda self, *a, _f=f, **k: _f(self, *a, **{'form': 'branch', **k}))\n"
        "warnings.simplefilter('ignore')\n"
        f"r = v1_reader.load({str(CASES['psfb_full_bridge_zvs'][0])!r}).run('dsed', mode_selection='lcp')\n"
        "t = np.asarray(r.times)\n"
        "i6, i7 = (np.asarray(v1_reader.observable(r, f'I({d})'), dtype=float) for d in ('D6', 'D7'))\n"
        "w = (t > 7.92053e-6 + 5e-9) & (t < 7.92053e-6 + 50e-9)\n"
        "ms = r.diagnostics['mode_selection']\n"
        "print(json.dumps({k: ms[k] for k in ('rounding_ties', 'leak_pins')}"
        " | {'rel': float(np.max(np.abs(i6[w] - i7[w]) / i6[w]))}))\n")
    env = dict(os.environ, PULSIM_TEST_LCP_NO_ROUNDING_FLOOR="1", PYTHONPATH=os.pathsep.join(sys.path))
    p = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stderr[-2000:]
    d = json.loads(p.stdout.strip().splitlines()[-1])
    assert d["rounding_ties"] == 0, d                # the floor is off ...
    assert d["leak_pins"] >= 1, d                    # ... the pre-fix selection is back
    assert d["rel"] > 0.5, d                         # D7 OFF: D6 carries the pair's current


@pytest.mark.parametrize("form", [
    "conductance",
    pytest.param("branch", marks=pytest.mark.xfail(strict=True, reason=(
        "release finding F-Z-9: on branch-form diodes the LCP puts D6 ON with 1.3e-6 A and D7 OFF at a "
        "7.8e-7 V gap at the 7.92 us rectifier commutation; D7 stays OFF to 7.9755 us (D6 4.66 A, truth 2.33 A)"))),
])
def test_exact_lcp_rectifier_pair_shares_at_7_95_us(form, monkeypatch):
    """F-Z-9's cause, at the mode (this branch's pin): at the PSFB rectifier
    commutation (7.9205 us, D5 and D8 ON) D6 and D7 are in parallel (0 -> r
    through equal R_on) and share the current equally; the LCP must turn both
    ON on either device form (the settle does)."""
    r = _psfb_exact_lcp(form, monkeypatch)
    k = int(np.searchsorted(np.asarray(r.times), 7.95e-6))
    i6, i7 = (float(np.asarray(v1_reader.observable(r, f"I({d})"))[k]) for d in ("D6", "D7"))
    assert i6 > 0.5 and abs(i6 - i7) <= 1e-6 * i6, (form, i6, i7)
