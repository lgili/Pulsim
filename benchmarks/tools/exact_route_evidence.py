"""Evidence for the exact engine's default-route gate (add-exact-propagator-strategy D11, 13.3).

    python3 benchmarks/tools/exact_route_evidence.py [--out benchmarks/EXACT_ROUTE_EVIDENCE.md]
                                                     [--circuits GLOB,...] [--refs DIR]

For every reference circuit the v1 reader maps (benchmarks/manifests/v2_triage.yaml, status
'runs') it runs the exact engine (``engine='exact'``) and today's default (``engine='auto'``)
the way the reference harness runs them (``v1_reader.BenchCase.run``: netlist dt, DC start,
switch_fn, step_observer) and grades both against the harness's reference (trbdf2 rtol 1e-8,
else pwl dt/4) with its strict measure. For every shipped example (``examples/*.yaml``) it
runs ``pulsim.exact_route`` and both engines; the examples have no reference, so their rows
give the route, the wall times, the events and the largest exact-vs-auto difference.

Nothing here changes a route: ``engine='auto'`` is not flipped (task 19.1 belongs to the
number-changing release). Wall times are single runs on whatever machine generated the file.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools"), str(ROOT / "benchmarks")]
import differential_runner as dr  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402


def _ref(rel, case, cols, tol, ref_dir):
    f = None
    if ref_dir is not None:
        ref_dir.mkdir(parents=True, exist_ok=True)
        f = ref_dir / (rel.replace("/", "__") + ".pkl")
        if f.exists():
            return pickle.loads(f.read_bytes())
    out = dr.reference(case, cols, tol)
    if f is not None:
        f.write_bytes(pickle.dumps(out))
    return out


def _events(res):
    for name in ("event_times",):
        v = getattr(res, name, None)
        if v is not None:
            return int(len(v))
    n = getattr(res, "n_events", None)
    return int(n) if n is not None else None


def _cell(case, engine, cols, ref, tol):
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = case.run(engine)
    except dr.REFUSAL as e:
        return dict(outcome="refused", wall_s=time.perf_counter() - t0,
                    why=f"{getattr(e, 'feature', '')}: {str(e).splitlines()[0][:90]}")
    except Exception as e:  # noqa: BLE001 - every outcome is a row
        return dict(outcome="error", wall_s=time.perf_counter() - t0,
                    why=f"{type(e).__name__}: {str(e).splitlines()[0][:90]}")
    rec = dict(wall_s=time.perf_counter() - t0, events=_events(res),
               engine_used=getattr(res, "engine_used", engine))
    if ref is None:
        rec.update(outcome="unverified", why="no converged reference")
    else:
        dr.classify(rec, res, case, cols, ref, tol, None)
    ex = getattr(res, "exact", None) or {}
    if ex:
        rec["auto_would_route"] = ex.get("auto_would_route")
    return rec


def reference_rows(patterns, ref_dir):
    rows = []
    triage = dr.load_triage()
    for rel, entry in sorted(triage.items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(dr.ROOT / rel)
        _, sw, _ = case.build()
        blob = json.dumps(case._components)
        switched = sw is not None or any(k in blob for k in ('"diode"', '"pwm"', '"pulse"'))
        cols = dr.observables_of(case)
        tol = entry.get("tol", {}).get("dsed", dr.default_tol("dsed", switched))
        ref_name, ref = _ref(rel, case, cols, tol, ref_dir)
        if entry.get("suspect"):
            ref_name, ref = None, None
        row = dict(circuit=rel.replace("circuits/", ""), reference=ref_name, tol=tol,
                   exact=_cell(case, "exact", cols, ref, tol),
                   auto=_cell(case, "auto", cols, ref, tol))
        rows.append(row)
        print(f"{row['circuit']:60s} exact={row['exact']['outcome']:10s} "
              f"auto={row['auto']['outcome']}", flush=True)
    return rows


# --- shipped examples against an independent truth (exact-engine session 7) -------------
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)


def _trace(r):
    from pulsim._result_views import with_right_limits
    r2 = with_right_limits(r)          # dsed stores LEFT limits at events: both sides
    return np.asarray(r2.times, float), np.asarray(r2.states, float)


def _l2(ta, xa, tb, xb, t0, t1):
    """Time-weighted RMS of a - b on [t0, t1], piecewise-linear traces, exact on the union grid."""
    tg = np.union1d(ta, tb)
    tg = np.union1d(tg[(tg > t0) & (tg < t1)], [t0, t1])
    d = np.interp(tg, ta, xa) - np.interp(tg, tb, xb)
    a, b = d[:-1], d[1:]
    return float(np.sqrt(np.sum(np.diff(tg) * (a * a + a * b + b * b) / 3.0) / (t1 - t0)))


def _tavg(t, y, a, b):
    m = (t > a) & (t < b)
    tt = np.r_[a, t[m], b]
    return float(np.trapezoid(np.r_[np.interp(a, t, y), y[m], np.interp(b, t, y)], tt) / (b - a))


def _ex_file(name):
    return str(ROOT / "examples" / f"{name}.yaml")


def _swfn(name, setting):
    if setting != "doc":
        return None            # the survey's setting: no switch_fn, every switch closed
    lc = pulsim.load_yaml_file(_ex_file(name))
    si, n = lc.builder.switch_index_of, len(lc.switch_names)
    if name == "half_bridge_pwm":
        return pulsim.make_dead_time_pwm_pair_fn(frequency=20e3, duty=0.5, hs_switch_idx=si("S_hi"),
                                                 ls_switch_idx=si("S_lo"), num_switches=n,
                                                 dead_time=1e-6)
    if name == "flyback":
        return pulsim.make_pwm_switch_fn(frequency=100e3, duty=0.5, switch_idx=si("Q1"),
                                         num_switches=n)
    if name == "phase_shift_full_bridge":   # phi = pi/2 (the header's D_eff = 0.5), 100 ns dead time
        return pulsim.make_phase_shift_full_bridge_fn(100e3, np.pi / 2, si("HS_A"), si("LS_A"),
                                                      si("HS_B"), si("LS_B"), n, 100e-9)
    return None


def _run(name, setting, engine, **kw):
    f = _swfn(name, setting)
    if f is not None:
        kw["switch_fn"] = f
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pulsim.simulate(pulsim.load_yaml_file(_ex_file(name)), engine=engine, **kw)
    return _trace(r) + (time.perf_counter() - t0,)


def _rc_pulse_truth(name, _setting):
    """Closed form of the RC step response through every pulse edge (both sides)."""
    nl = yaml.safe_load(open(_ex_file(name)))["circuit"]["devices"]
    d = {x["name"]: x for x in nl}
    R, C, src = d["R1"]["R"], d["C1"]["C"], d["Vpulse"]
    lc = pulsim.load_yaml_file(_ex_file(name))
    names = list(lc.builder.state_var_names())
    w, T, tau, vh = src["pulse_width"], lc.t_end, R * C, src["v_pulsed"]
    ts, X, v = [], [], 0.0
    for k in range(int(round(T / w))):
        u = vh if k % 2 == 0 else src["v_initial"]
        tt = np.linspace(k * w, (k + 1) * w, 20001)
        if k:
            tt[0] = np.nextafter(k * w, np.inf)
        vo = u + (v - u) * np.exp(-(tt - k * w) / tau)
        row = {"V(in)": np.full_like(tt, u), "V(out)": vo, "Is(Vpulse)": -(u - vo) / R}
        ts.append(tt)
        X.append(np.c_[[row[n] for n in names]].T)
        v = u + (v - u) * np.exp(-w / tau)
    return np.concatenate(ts), np.concatenate(X), "closed form", None


def _rc_ladder_truth(name, _setting):
    """Eigendecomposition of the RC ladder's symmetric (C^-1/2 G C^-1/2) matrix from the netlist."""
    nl = yaml.safe_load(open(_ex_file(name)))["circuit"]["devices"]
    src = next(x for x in nl if x["type"] == "voltage_source")
    free = sorted({x[k] for x in nl for k in ("from", "to")} - {"gnd", src["from"]})
    ix = {n: i for i, n in enumerate(free)}
    N = len(free)
    G, b, Cd = np.zeros((N, N)), np.zeros(N), np.zeros(N)
    for x in nl:
        if x["type"] == "resistor":
            g = 1 / x["R"]
            for p_, q in ((x["from"], x["to"]), (x["to"], x["from"])):
                if p_ in ix:
                    G[ix[p_], ix[p_]] += g
                    if q in ix:
                        G[ix[p_], ix[q]] -= g
                    elif q == src["from"]:
                        b[ix[p_]] += g * src["V"]
        elif x["type"] == "capacitor":
            Cd[ix[x["from"]]] += x["C"]
    s = 1 / np.sqrt(Cd)
    lam, Q = np.linalg.eigh((s[:, None] * G) * s[None, :])
    vinf = np.linalg.solve(G, b)
    coef = Q.T @ (-vinf / s)
    lc = pulsim.load_yaml_file(_ex_file(name))
    names = list(lc.builder.state_var_names())
    def at(t):   # evaluated at the engines' own samples: no interpolation error in the truth
        V = vinf[None, :] + s[None, :] * ((np.exp(-np.outer(t, lam)) * coef[None, :]) @ Q.T)
        X = np.zeros((len(t), len(names)))
        for n, i in ix.items():
            X[:, names.index(f"V({n})")] = V[:, i]
        X[:, names.index(f"V({src['from']})")] = src["V"]
        i_src = np.zeros(len(t))        # the source delivers what its resistors carry away
        for x in nl:
            if x["type"] == "resistor" and src["from"] in (x["from"], x["to"]):
                other = x["to"] if x["from"] == src["from"] else x["from"]
                i_src -= (src["V"] - (V[:, ix[other]] if other in ix else 0.0)) / x["R"]
        X[:, names.index(f"Is({src['name']})")] = i_src
        return X
    return at, None, "closed form (eigendecomposition)", None


def _numeric_truth(cfg):
    """Two runs of a converging engine: the first is the truth, its distance
    to the second the truth's own uncertainty. dt entries are fractions of the
    example's dt."""
    def f(name, setting):
        lc = pulsim.load_yaml_file(_ex_file(name))
        out = []
        for eng, kw in cfg[1:]:
            kw = {k: (v * lc.dt if k == "dt" else v) for k, v in kw.items()}
            out.append(_run(name, setting, eng, **kw)[:2])
        return out[0][0], out[0][1], cfg[0], out[1]
    return f


TRBDF2 = ("trbdf2 rtol 1e-10 (vs 1e-9)", ("trbdf2", dict(rtol=1e-10, atol=1e-13)),
          ("trbdf2", dict(rtol=1e-9, atol=1e-12)))
SPLIT32 = ("pwl+split dt/32 (vs dt/8)", ("pwl", dict(dt=1 / 32, **SPLIT)),
           ("pwl", dict(dt=1 / 8, **SPLIT)))
SPLIT8 = ("pwl+split dt/8 (vs dt)", ("pwl", dict(dt=1 / 8, **SPLIT)), ("pwl", dict(dt=1.0, **SPLIT)))


def _claims_half_bridge(t, X, names):
    i, vm = X[:, names.index("I(Lload)")], X[:, names.index("V(mid)")]
    T, P = 5e-4, 5e-5
    m = (t >= T - P) & (t <= T)
    return [("v(mid) @ 500 us = 50.23 V", f"{np.interp(T, t, vm):.3f}"),
            ("i_L offset = -0.02 A", f"{_tavg(t, i, T - P, T):+.4f}"),
            ("i_L amplitude ~0.62 A", f"{i[m].max():.3f} / {i[m].min():.3f}")]


def _claims_rectifier(t, X, names):
    v = X[:, names.index("V(vdc_p)")] - X[:, names.index("V(vdc_n)")]
    T, P = 0.8, 0.02
    m = (t >= T - P) & (t <= T)
    return [("V_dc ~171 V", f"{_tavg(t, v, T - P, T):.2f}"),
            ("ripple <= 5.7 V pp", f"{v[m].max() - v[m].min():.2f}")]


def _claims_flyback(t, X, names):
    c = {n: X[:, i] for i, n in enumerate(names)}
    T, W = 5e-3, 1e-4
    v = c["V(vout)"] - c["V(sec_neg)"]
    vc = c["V(clamp)"] - c["V(vin)"]
    pin = _tavg(t, 48.0 * -c["Is(Vin)"], T - W, T)
    pout, pcl = _tavg(t, v * v / 5.0, T - W, T), _tavg(t, vc * vc / 2e3, T - W, T)
    return [("V_out = 19.8 V", f"{_tavg(t, v, T - W, T):.4f}"),
            ("clamp ~17 W", f"{pcl:.3f}"),
            ("balance closes to 0.1 %", f"{(pin - pout - pcl) / pin:.2%}")]


def _claims_psfb(t, X, names):
    v = X[:, names.index("V(vout)")] - X[:, names.index("V(rect_neg)")]
    return [("V_out ~25 V less drops", f"{_tavg(t, v, 1.8e-3, 2e-3):.3f}"),
            ("settled at 2 ms (mean 1.0-1.2 vs 1.8-2.0 ms)",
             f"{_tavg(t, v, 1.0e-3, 1.2e-3):.3f} vs {_tavg(t, v, 1.8e-3, 2e-3):.3f}")]


EXAMPLE_STUDY = [
    ("rc_pulse", "survey", _rc_pulse_truth, None),
    ("stress_large_rc_ladder", "survey", _rc_ladder_truth, None),
    ("half_bridge_pwm", "survey", _numeric_truth(SPLIT32), None),
    ("half_bridge_pwm", "doc", _numeric_truth(TRBDF2), _claims_half_bridge),
    ("three_phase_diode_rectifier", "survey", _numeric_truth(TRBDF2), _claims_rectifier),
    ("flyback", "survey", _numeric_truth(SPLIT8), None),
    ("flyback", "doc", _numeric_truth(SPLIT8), _claims_flyback),
    ("phase_shift_full_bridge", "survey", _numeric_truth(SPLIT32), None),
    ("phase_shift_full_bridge", "doc", _numeric_truth(SPLIT32), _claims_psfb),
]


def _worst(truth, run, names, t0, t1, skip=()):
    t, X = truth
    pk = {n: float(np.max(np.abs(X[t >= t0, j]))) for j, n in enumerate(names)}
    unit = {u: max([v for n, v in pk.items() if n.startswith(u)] + [1e-300]) for u in ("V", "I")}
    best = (0.0, "")
    for j, n in enumerate(names):
        if n in skip:
            continue
        e = _l2(t, X[:, j], run[0], run[1][:, j], t0, t1) / max(pk[n], 1e-6 * unit[n[0]])
        best = max(best, (e, n))
    return best


def example_study():
    rows = []
    for name, setting, truth_fn, claims_fn in EXAMPLE_STUDY:
        lc = pulsim.load_yaml_file(_ex_file(name))
        names = list(lc.builder.state_var_names())
        row = dict(example=name, setting=setting, claims=[])
        t, X, row["truth"], second = truth_fn(name, setting)
        t0, t1 = lc.dt, lc.t_end            # from the first output step: sample 0 is a start convention
        runs = {}
        for eng in ("exact", "auto"):
            try:
                runs[eng] = _run(name, setting, eng)
            except Exception as ex:  # noqa: BLE001 - a refusal is a row
                row[eng] = dict(why=f"{type(ex).__name__}: {str(ex).splitlines()[0][:70]}")
        at = t if callable(t) else None
        if at is not None:                  # a closed form: judged on each engine's own samples
            t = np.union1d(np.r_[0.0, np.geomspace(1e-12 * t1, t1, 4000)],
                           np.concatenate([v[0] for v in runs.values()]))
            X = at(t)
        row["truth_unc"] = _worst((t, X), second, names, t0, t1)[0] if second is not None else 0.0
        for eng, (te, Xe, w) in runs.items():
            # a closed form is exact between any two samples; an engine's trace is
            # linear there, so a denser truth grid would charge it interpolation error
            e, col = _worst((te, at(te)) if at is not None else (t, X), (te, Xe), names, t0, t1)
            row[eng] = dict(err=e, col=col, wall_s=w)
            if claims_fn:
                row["claims"].append((eng, claims_fn(te, Xe, names)))
        if claims_fn:
            row["claims"].insert(0, ("truth", claims_fn(t, X, names)))
        rows.append(row)
        print(f"{name:30s} {setting:6s} exact={row['exact'].get('err')} auto={row['auto'].get('err')}",
              flush=True)
    return rows


def example_rows():
    rows = []
    for f in sorted((ROOT / "examples").glob("*.yaml")):
        try:
            lc = pulsim.load_yaml_file(str(f))
        except Exception as e:  # noqa: BLE001 - a v1 file the 2.0 loader refuses
            rows.append(dict(example=f.stem, load=f"{type(e).__name__}"))
            continue
        row = dict(example=f.stem)
        t_end = getattr(lc, "t_end", 0.0) or None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rt = pulsim.exact_route(lc.builder, t_end) if t_end else None
            row["route"] = (rt["exact"], rt["reason"]) if rt else (None, "no t_end")
        except Exception as e:  # noqa: BLE001
            row["route"] = (False, f"{type(e).__name__}: {str(e).splitlines()[0][:60]}")
        for eng in ("exact", "auto"):
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = pulsim.simulate(pulsim.load_yaml_file(str(f)), engine=eng)
                row[eng] = dict(outcome="ran", wall_s=time.perf_counter() - t0,
                                events=_events(r), engine_used=getattr(r, "engine_used", eng),
                                n=len(r.times))
            except Exception as e:  # noqa: BLE001
                row[eng] = dict(outcome="refused" if isinstance(e, dr.REFUSAL) else "error",
                                wall_s=time.perf_counter() - t0,
                                why=f"{type(e).__name__}: {str(e).splitlines()[0][:80]}")
        rows.append(row)
        print(f"{f.stem:40s} {row.get('route')}", flush=True)
    return rows


def _fmt(c):
    if c is None:
        return "", "", "", ""
    o = c["outcome"]
    err = f"{c['err_rms']:.2g}" if "err_rms" in c else c.get("why", "")
    ev = "" if c.get("events") is None else str(c["events"])
    return o, err, f"{c['wall_s']:.3g}", ev


def write(path, refs, exs, study):
    L = ["# Exact engine route evidence", "",
         "Generated by `benchmarks/tools/exact_route_evidence.py`; do not edit by hand.", "",
         "Exact = `engine='exact'`; auto = today's default `engine='auto'` (trbdf2 on these "
         "runs: the v1 harness passes no dt to auto). Error = the strict gate's err_rms "
         "against the harness reference (agree at <= tol). Wall times are one run each on a "
         "shared machine: an indication, not a benchmark. Nothing here changes a route.", ""]
    cnt = {}
    for r in refs:
        for k in ("exact", "auto"):
            cnt.setdefault(k, {}).setdefault(r[k]["outcome"], 0)
            cnt[k][r[k]["outcome"]] += 1
    L += ["## Totals (reference circuits)", "", "| engine | " + " | ".join(
        sorted({o for v in cnt.values() for o in v})) + " |"]
    outs = sorted({o for v in cnt.values() for o in v})
    L += ["|---|" + "---|" * len(outs)]
    for k in ("exact", "auto"):
        L += [f"| {k} | " + " | ".join(str(cnt[k].get(o, 0)) for o in outs) + " |"]
    both = [r for r in refs if r["exact"]["outcome"] == "agree" and "wall_s" in r["auto"]
            and r["auto"]["outcome"] in ("agree", "disagree")]
    if both:
        ratio = np.median([r["exact"]["wall_s"] / max(r["auto"]["wall_s"], 1e-9) for r in both])
        L += ["", f"Median wall-time ratio exact / auto where both ran and exact agrees: "
              f"{ratio:.3g} over {len(both)} circuits."]
    L += ["", "## Reference circuits", "",
          "| circuit | tol | exact | exact err | exact s | exact events | auto would route | "
          "auto | auto err | auto s | auto events |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in refs:
        e, a = _fmt(r["exact"]), _fmt(r["auto"])
        L += [f"| {r['circuit']} | {r['tol']:.0e} | {e[0]} | {e[1]} | {e[2]} | {e[3]} | "
              f"{r['exact'].get('auto_would_route', '')} | {a[0]} | {a[1]} | {a[2]} | {a[3]} |"]
    L += ["", "## Shipped examples against an independent truth", "",
          "Each example is run in the route survey's setting (`survey`: no switch_fn, every "
          "switch closed) and, where the file documents a switch_fn, in that one (`doc`: the "
          "header's frequency and duty; the PSFB at phi = pi/2 with 100 ns dead time). Error = "
          "the worst column's time-weighted RMS over [dt, t_end] of the piecewise-linear "
          "traces, relative to that column's peak (a column at ~0 is judged against 1e-6 of "
          "its unit's largest peak); exact's trace carries both sides of every jump "
          "(`with_right_limits`). A closed form is evaluated on each engine's own samples, "
          "so an engine is not charged for drawing straight lines between them. The truth's uncertainty is its distance to the second, "
          "coarser truth run. auto = the example's own dt on pwl (the one rule: a dt in the "
          "file is a fixed step). The old column (the largest pointwise difference from sample "
          "1 divided by the column's peak) compared a left limit with a right limit at every "
          "edge and divided noise by ~0 peaks; it is gone.", "",
          "| example | setting | truth | truth unc. | exact err (column) | exact s | "
          "auto err (column) | auto s |", "|---|---|---|---|---|---|---|---|"]

    def cell(c):
        if "err" not in c:
            return c.get("why", ""), ""
        return f"{c['err']:.2g} ({c['col']})", f"{c['wall_s']:.3g}"
    for r in study:
        e, a = cell(r["exact"]), cell(r["auto"])
        L += [f"| {r['example']} | {r['setting']} | {r['truth']} | {r['truth_unc']:.2g} | "
              f"{e[0]} | {e[1]} | {a[0]} | {a[1]} |"]
    L += ["", "### The examples' stated numbers", "",
          "| example | claim | truth | exact | auto |", "|---|---|---|---|---|"]
    for r in study:
        if not r["claims"]:
            continue
        by = {k: dict(v) for k, v in r["claims"]}
        for claim in by["truth"]:
            L += [f"| {r['example']} | {claim} | {by['truth'][claim]} | "
                  f"{by.get('exact', {}).get(claim, 'refused')} | "
                  f"{by.get('auto', {}).get(claim, '')} |"]
    L += ["", "## Every shipped example: route and wall time", "",
          "`simulate(load_yaml_file(f), engine=...)` as the route survey runs it.", "",
          "| example | exact_route | exact | exact s | exact samples | auto | auto s | "
          "auto samples |", "|---|---|---|---|---|---|---|---|"]
    for r in exs:
        if "load" in r:
            L += [f"| {r['example']} | not loadable ({r['load']}) | | | | | | |"]
            continue
        rt = r.get("route", (None, ""))
        e, a = r.get("exact", {}), r.get("auto", {})
        L += [f"| {r['example']} | {'exact' if rt[0] else rt[1]} | "
              f"{e.get('outcome', '')} {e.get('why', '')} | {e.get('wall_s', 0):.3g} | "
              f"{e.get('n', '')} | {a.get('outcome', '')} {a.get('why', '')} | "
              f"{a.get('wall_s', 0):.3g} | {a.get('n', '')} |"]
    Path(path).write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "EXACT_ROUTE_EVIDENCE.md"))
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--refs", default=None, help="cache directory for the references")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args(argv)
    refs = reference_rows(a.circuits.split(","), Path(a.refs) if a.refs else None)
    exs = [] if a.no_examples else example_rows()
    study = [] if a.no_examples else example_study()
    write(a.out, refs, exs, study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
