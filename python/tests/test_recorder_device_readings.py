"""Device readings in the loop (add-probes-and-recorders item 1, DECISIONS 20).

A device's current is read under the mode the engine COMMITTED, at the
committed state; a capacitor's is the companion's committed current (the
current the solver enforced), so at every recorded sample the currents of
every node sum to zero to the solver's own floor, and the trapezoid of i_C
over a step is C·Δv on a trapezoidal step. The energy balance below uses
recorder outputs only.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pytest

import pulsim as p


EPS = np.finfo(float).eps
_RUNS = {"pwl": dict(engine="pwl", dt=1e-6),
         "trbdf2": dict(engine="trbdf2", rtol=1e-7, atol=1e-10)}
# the companion readings on integrator='esdirk43' too (the KCL and charge
# tests; the per-kind tests keep _RUNS). Before the interpolated-landing fix
# the per-sample KCL check failed on esdirk43 at exactly those landings
# (7e-5 of scale on the buck, 7e-6 on the rectifier): it exercises them.
_RUNS_C = {**_RUNS, "esdirk43": dict(_RUNS["trbdf2"], integrator="esdirk43")}
A, B = 0.21e-3 + 1.234e-9, 0.87e-3 + 4.321e-9   # mid-step on every grid


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 10.0)
    b.add_resistor("R1", "in", "out", 1e3)
    b.add_capacitor("C1", "out", "gnd", 1e-6)
    return b, None, 3e-3, {"in": ["V1", "R1"], "out": ["-R1", "C1"]}, {"C1": ("out", 1e-6)}


def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 24.0)
    b.add_switch("S1", "in", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("R", "out", "gnd", 5.0)
    fn = p.make_pwm_switch_fn(20e3, 0.4, b.switch_index_of("S1"), b.graph.num_switches)
    kcl = {"in": ["V", "S1"], "sw": ["-S1", "-D1", "L"], "out": ["-L", "C", "R"]}
    return b, fn, 1e-3, kcl, {"C": ("out", 10e-6)}


def _rectifier():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 100.0, 50.0)
    for nm, a, c in (("D1", "s", "dp"), ("D2", "gnd", "dp"),
                     ("D3", "dn", "s"), ("D4", "dn", "gnd")):
        b.add_diode(nm, a, c, 1e3, 1e-9)
    b.add_capacitor("Cf", "dp", "dn", 100e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    kcl = {"s": ["Vac", "D1", "-D3"], "dp": ["-D1", "-D2", "Cf", "RL"],
           "dn": ["D3", "D4", "-Cf", "-RL", "Rtie"]}
    return b, None, 40e-3, kcl, {"Cf": ("dp,dn", 100e-6)}


_CIRCUITS = {"rc": _rc, "buck": _buck, "rectifier": _rectifier}


def _run(fac, eng, probes, **rec):
    b, fn, t_end, kcl, caps = fac()
    extra = {} if fn is None else {"switch_fn": fn}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end, recorders=[p.Recorder("r", probes, **rec)],
                         **_RUNS_C[eng], **extra)
    return res, kcl, caps


@pytest.mark.parametrize("eng", sorted(_RUNS_C))
@pytest.mark.parametrize("circ", sorted(_CIRCUITS))
def test_kcl_holds_at_every_recorded_sample(circ, eng):
    _, _, _, kcl, caps = _CIRCUITS[circ]()
    devs = sorted({t.lstrip("-") for ts in kcl.values() for t in ts})
    vs = sorted({f"v({n})" for n, _ in caps.values()})
    res, kcl, caps = _run(_CIRCUITS[circ], eng, [f"i({d})" for d in devs] + vs)
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    a = res.recorded["r"]
    t = a.times
    h = np.diff(t)
    # the solver's own floor: the companion law turns one ulp of a capacitor
    # voltage into C·eps·|v| / h of current (femtosecond steps near events)
    floor = sum(8 * c * EPS * np.max(np.abs(a[f"v({n})"])) for n, c in caps.values())
    for node, terms in kcl.items():
        cols = [(-1.0 if x[0] == "-" else 1.0, f"i({x.lstrip('-')})") for x in terms]
        scale = max(np.max(np.abs(a[q])) for _, q in cols)
        s = sum(sg * a[q] for sg, q in cols)
        tol = 1e-9 * scale + floor / h
        bad = np.flatnonzero(np.abs(s[1:]) > tol)
        assert bad.size == 0, (node, t[1 + bad[:5]], s[1 + bad[:5]])
        if a.event_t.size:
            # an event row's right side (a settled probe, or the end of the
            # first step under the new mode) obeys KCL too
            se = sum(sg * a.event_right[q] for sg, q in cols)
            assert np.all(np.abs(se) <= 1e-9 * scale + floor / 1e-15), (node, se)


@pytest.mark.parametrize("eng", sorted(_RUNS_C))
@pytest.mark.parametrize("circ", ["rc", "buck", "rectifier"])
def test_capacitor_charge_over_mid_step_windows(circ, eng, request):
    _, _, t_end, _, caps = _CIRCUITS[circ]()
    (cap, (nodes, c)), = caps.items()
    wins = [(0.07 * t_end + 1.234e-9, 0.29 * t_end + 4.321e-9),
            (0.31 * t_end + 2.2e-9, 0.97 * t_end)]
    v = f"v({nodes})"
    res, _, _ = _run(_CIRCUITS[circ], eng, [f"i({cap})", v], windows=wins)
    res2, _, _ = _run(_CIRCUITS[circ], eng, [v], keep="times",
                      times=[x for w in wins for x in w])
    w = res.recorded["r"].windows[f"i({cap})"]
    ends = res2.recorded["r"][v]
    for k in range(len(wins)):
        dq = c * (ends[2 * k + 1] - ends[2 * k])
        q = w["integral"][k]
        # trapezoidal steps: exact per step; the residue is the two partial
        # steps at the window ends (linear i vs linear v) and, on trbdf2,
        # the BDF2 stage (O(h^3) per step)
        rel = 2e-4 if eng == "pwl" else 3e-3   # >= 5x the residue seen here
        assert abs(q - dq) <= rel * abs(dq) + 1e-12, (k, q, dq)
        print(f"{circ} {eng} window {k}: rel {(q - dq) / dq:.2e}", file=sys.stderr)


_BUCK_P = ["p(V)", "p(S1)", "p(D1)", "p(R)", "p(L)", "p(C)"]


@pytest.mark.parametrize("eng", ["pwl", "trbdf2", "exact"])
def test_energy_balance_from_recorder_outputs(eng, request):
    # the exact engine is read from its stored samples (linear between
    # them): sampled every 1 us, as the pwl grid
    run = dict(_RUNS.get(eng, {"engine": "exact", "dt": 1e-6}))
    b, fn, t_end, _, _ = _buck()
    probes = _BUCK_P if eng != "exact" else _BUCK_P[:4]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end, switch_fn=fn, **run, recorders=[
            p.Recorder("e", probes, keep="none", windows=[(A, B)]),
            p.Recorder("s", ["v(out)", "i(L)"], keep="times", times=[A, B])])
    e = {q: res.recorded["e"].windows[q]["integral"][0] for q in probes}
    s = res.recorded["s"]
    stored = 0.5 * 100e-6 * np.diff(s["i(L)"] ** 2)[0] + 0.5 * 10e-6 * np.diff(s["v(out)"] ** 2)[0]
    supplied = -e["p(V)"]
    dissipated = e["p(S1)"] + e["p(D1)"] + e["p(R)"]
    resid = supplied - dissipated - stored
    print(f"{eng}: supplied {supplied:.6e} J, residual {resid:.2e} "
          f"({resid / supplied:.2e})", file=sys.stderr)
    # residues seen: pwl 1.4e-5, trbdf2 3.8e-7, exact 1.1e-5 (relative)
    assert abs(resid) <= 1e-4 * supplied, (supplied, dissipated, stored)
    if eng != "exact":
        # Tellegen on the same records: the absorbed powers sum to zero
        tel = sum(e.values())
        assert abs(tel) <= 1e-9 * supplied, tel
        # the reactive powers integrate to the stored-energy change
        react = e["p(L)"] + e["p(C)"] - stored
        print(f"{eng}: reactive residual {react / supplied:.2e}", file=sys.stderr)
        assert abs(react) <= 1e-4 * supplied, react


def test_device_readings_refused_on_the_stored_array_pass():
    b, *_ = _rc()
    with pytest.raises(ValueError, match=r"i\(C1\) of a capacitor .*in the loop"):
        p.simulate(b, 1e-3, engine="exact",
                   recorders=[p.Recorder("r", ["i(C1)"])])


def _series(add, dc=0.0):
    """V (sine) -> Rs -> device(mid, gnd): KCL at mid is i(Rs) = i(device)."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "in", "gnd", dc, 5.0 - dc, 1e3)
    b.add_resistor("Rs", "in", "mid", 10.0)
    add(b)
    return b


def _gated(add):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "in", "gnd", 10.0)
    b.add_resistor("Rs", "in", "mid", 10.0)
    b.add_sine_voltage_source("Vg", "g", "gnd", 5.0, 5.0, 1e3)
    add(b)
    return b


_NEWTON = {
    "shockley_diode": lambda: _series(lambda b: b.add_shockley_diode("X", "mid", "gnd")),
    "lauritzen_diode": lambda: _series(lambda b: b.add_lauritzen_diode("X", "mid", "gnd")),
    "saturable_inductor": lambda: _series(
        lambda b: b.add_saturable_inductor("X", "mid", "gnd", 1e-3, 0.2, 1e-4)),
    "nonlinear_capacitor": lambda: _series(
        lambda b: b.add_nonlinear_capacitor("X", "mid", "gnd", 1e-6, 1.0), dc=3.0),
    "mosfet_level1": lambda: _gated(lambda b: b.add_mosfet_level1(
        "X", "mid", "gnd", "g", 2.0, 3.0, with_body_diode=False)),
    "igbt_level1": lambda: _gated(lambda b: b.add_igbt_level1("X", "mid", "gnd", "g")),
}


@pytest.mark.parametrize("eng", sorted(_RUNS))
@pytest.mark.parametrize("kind", sorted(_NEWTON))
def test_newton_kind_current_equals_its_series_resistor(kind, eng):
    # KCL at the device's terminal against a series resistor, at every
    # sample, and p() = v(mid)·i() for a two-terminal law
    run = dict(_RUNS[eng], dt=2e-6) if eng == "pwl" else _RUNS[eng]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(_NEWTON[kind](), 2e-3, **run, recorders=[
            p.Recorder("r", ["i(Rs)", "i(X)", "p(X)", "v(mid)"])])
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    a = res.recorded["r"]
    scale = np.max(np.abs(a["i(Rs)"]))
    assert scale > 1e-3, kind                     # the device really conducts
    h = np.diff(a.times)
    floor = 8 * 1e-6 * EPS * 5.0 / h if kind == "nonlinear_capacitor" else 0.0
    d = np.abs(a["i(Rs)"] - a["i(X)"])[1:]
    assert np.all(d <= 1e-6 * scale + floor), (kind, np.max(d))
    if kind not in ("mosfet_level1", "igbt_level1"):
        np.testing.assert_allclose(a["p(X)"], a["v(mid)"] * a["i(X)"],
                                   rtol=1e-9, atol=1e-12 * scale * 5.0)


@pytest.mark.parametrize("eng", sorted(_RUNS))
def test_vcvs_and_current_source_readings(eng):
    b = p.CircuitBuilder()
    b.add_sine_current_source("I1", "a", "gnd", 0.1, 0.05, 1e3)
    b.add_resistor("Ra", "a", "gnd", 20.0)
    b.add_vcvs("E1", "a", "gnd", "o", "gnd", 3.0)
    b.add_resistor("Ro", "o", "gnd", 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, 2e-3, **_RUNS[eng], recorders=[
            p.Recorder("r", ["i(I1)", "i(Ra)", "i(E1)", "i(Ro)", "p(E1)", "p(Ro)"])])
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    a = res.recorded["r"]
    # a current source's i() is positive out of n_pos (as res.i); sample 0
    # is the zero start state, not an operating point
    np.testing.assert_allclose(a["i(Ra)"][1:], a["i(I1)"][1:], rtol=1e-9)
    np.testing.assert_allclose(a["i(E1)"], -a["i(Ro)"], rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(a["p(E1)"], -a["p(Ro)"], rtol=1e-9, atol=1e-15)


def test_device_readings_on_esdirk43_agree_with_trbdf2():
    # every reading, the capacitor's companion current included (its KCL:
    # test_kcl_holds_at_every_recorded_sample), in the loop on
    # integrator='esdirk43'; the window energies agree with the TR-BDF2 run's
    probes = list(_BUCK_P)
    e = {}
    for integ, extra in (("trbdf2", {}), ("esdirk43", {"integrator": "esdirk43"})):
        b, fn, t_end, _, _ = _buck()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = p.simulate(b, t_end, switch_fn=fn, **_RUNS["trbdf2"], **extra,
                             recorders=[p.Recorder("e", probes, keep="none", windows=[(A, B)])])
        assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
        e[integ] = {q: res.recorded["e"].windows[q]["integral"][0] for q in probes}
    supplied = -e["trbdf2"]["p(V)"]
    for q in probes:
        # both runs at rtol 1e-7; seen <= 2e-5 of the supplied energy
        print(f"{q}: esdirk43 - trbdf2 = {(e['esdirk43'][q] - e['trbdf2'][q]) / supplied:.2e} "
              "of the supplied energy", file=sys.stderr)
        assert abs(e["esdirk43"][q] - e["trbdf2"][q]) <= 2e-4 * supplied, q


def test_diode_right_side_is_exactly_zero_after_turn_off():
    # Analytic value of an event row's right side (DECISIONS 24): a PWL
    # diode with zero off conductance carries exactly 0 A once off, so on
    # the right of every turn-off the row reads 0.0 bit for bit (the right
    # side is read under the settled mode, from the settled state). A
    # bridge rectifier on TR-BDF2 (settled event rows at the landings).
    t_end = 40e-3
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", 0.0, 100.0, 50.0)
    diodes = (("D1", "s", "dp"), ("D2", "gnd", "dp"), ("D3", "dn", "s"), ("D4", "dn", "gnd"))
    for nm, an, ca in diodes:
        b.add_diode(nm, an, ca, 1e3, 0.0)
    b.add_capacitor("Cf", "dp", "dn", 100e-6)
    b.add_resistor("RL", "dp", "dn", 100.0)
    b.add_resistor("Rtie", "dn", "gnd", 1e6)
    probes = [f"i({nm})" for nm, _, _ in diodes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end, **_RUNS["trbdf2"], recorders=[p.Recorder("r", probes)])
    assert res.recorder_feed["mode"] == "in-loop", res.recorder_feed
    a = res.recorded["r"]
    settled = a.event_resolution == "settled"
    assert np.count_nonzero(settled) >= 8
    n_off = 0
    for q in probes:
        left, right = a.event_left[q][settled], a.event_right[q][settled]
        # a diode the settle turned off: it carried reverse current on the
        # left (the step that crossed zero; ~1e-6 A here, far above the
        # 1e-13 A roundoff an off diode reads), and reads exactly 0 on the right
        off = left < -1e-9
        n_off += int(np.count_nonzero(off))
        assert np.all(right[off] == 0.0), (q, a.event_t[settled][off], right[off])
        # no row reads reverse current on the right: a conducting diode
        # carries forward current, an off one exactly none
        assert np.all(right >= 0.0), (q, right[right < 0])
    assert n_off >= 4


def _half_wave_rl():
    # benchmarks/circuits/diodes/diode_half_wave_rl_load.yaml: a series diode
    # into R + L (its turn-off carries the inductor's residual current)
    from pathlib import Path
    root = Path(__file__).resolve().parents[2] / "benchmarks"
    sys.path[:0] = [str(root / "tools"), str(root)]
    import v1_reader
    case = v1_reader.load(root / "circuits/diodes/diode_half_wave_rl_load.yaml")
    kcl = {"in": ["Vac", "D1"], "out": ["-D1", "R1"], "n_l": ["-R1", "L1"]}
    return case.build()[0], case.t_end, 20e-6, kcl, {"D1": ("in", "out", 200.0, 1e-6)}, {}


def _rectifier_laws():
    b, _, t_end, kcl, _ = _rectifier()
    dio = {nm: (a, c, 1e3, 1e-9) for nm, a, c in
           (("D1", "s", "dp"), ("D2", "gnd", "dp"), ("D3", "dn", "s"), ("D4", "dn", "gnd"))}
    return b, t_end, 1e-6, kcl, dio, {"Cf": 100e-6}


@pytest.mark.parametrize("split", [False, True], ids=["rule_off", "split"])
@pytest.mark.parametrize("circ", ["rectifier", "half_wave_rl"])
def test_every_row_obeys_kcl_and_the_diode_law(circ, split):
    # Every recorder row (kept rows, both sides of every event row) on the
    # fixed engine, with the split (discontinuity_restart +
    # enable_substep_state_correction) and without. Before the fix the split's
    # located V_f = 0 turn-ons (continuity points) published the left state
    # read under the ON law: i(D1) = g_on·v_D(left) = 2.06 mA on the rectifier
    # (KCL at s/dp off by 2 mA; at t = 0.030004 D3 off by 17 mA).
    b, t_end, dt, kcl, dio, caps = (_rectifier_laws if circ == "rectifier" else _half_wave_rl)()
    devs = sorted({x.lstrip("-") for ts in kcl.values() for x in ts})
    nodes = sorted({n for a, c, _, _ in dio.values() for n in (a, c) if n != "gnd"} | set(kcl))
    probes = [f"i({d})" for d in devs] + [f"v({n})" for n in nodes]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end, engine="pwl", dt=dt, discontinuity_restart=split,
                         enable_substep_state_correction=split,
                         recorders=[p.Recorder("r", probes)])
    a = res.recorded["r"]
    ev = a.event_t
    if split:   # the located crossings are exercised (off the dt grid)
        assert np.count_nonzero(np.abs(ev / dt - np.round(ev / dt)) > 1e-6) >= 4, ev
    vmax = max(np.max(np.abs(a[f"v({n})"])) for n in nodes if n != "gnd")
    # the right limit's backward-Euler companion at eps = 1e-3·γ·dt turns one
    # ulp of a capacitor voltage into C·eps_mach·|v| / eps of current
    floor = sum(8 * c * EPS * vmax / (1e-3 * (2 - np.sqrt(2)) * dt) for c in caps.values())
    sides = [("rows", lambda q: np.asarray(a[q]), a.times)]
    if ev.size:
        sides += [("left", lambda q: np.asarray(a.event_left[q]), ev),
                  ("right", lambda q: np.asarray(a.event_right[q]), ev)]
    for side, g, t in sides:
        def v(n):
            return 0.0 if n == "gnd" else g(f"v({n})")
        for node, terms in kcl.items():
            cols = [(-1.0 if x[0] == "-" else 1.0, f"i({x.lstrip('-')})") for x in terms]
            scale = max(np.max(np.abs(a[q])) for _, q in cols)   # the run's currents
            s = sum(sg * g(q) for sg, q in cols)
            bad = np.flatnonzero(np.abs(s) > 1e-12 * scale + floor)
            assert bad.size == 0, (side, node, t[bad[:5]], s[bad[:5]])
        for d, (an, ca, gon, goff) in dio.items():
            # an ON diode reads g_on·v_D, an OFF one g_off·v_D (V_th = 0)
            vd, i = v(an) - v(ca), g(f"i({d})")
            err = np.minimum(np.abs(i - gon * vd), np.abs(i - goff * vd))
            bad = np.flatnonzero(err > 64 * EPS * gon * vmax + 1e-12 * np.abs(i))
            assert bad.size == 0, (side, d, t[bad[:5]], i[bad[:5]], vd[bad[:5]])
