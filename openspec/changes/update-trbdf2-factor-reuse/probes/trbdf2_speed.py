"""Work-precision of engine='trbdf2' against the fixed-step engine (pwl, pwl+split).

    PYTHONPATH=<wt>/python:<wt>/python/tests python3 trbdf2_speed.py [--reps 3] [--only buck_ccm,...]
        [--json out.json]

Per circuit, every engine runs a tolerance (trbdf2: rtol, atol = 1e-3 rtol) or step (pwl: dt = T/N)
sweep. Accuracy is measured against an independent truth, never against the engine itself:

* buck CCM / DCM, boost: exact piecewise-LTI propagation (expm, diode events by brentq) at every
  switching-period start of the last 20 periods; every engine lands a sample there (a gate rise).
  err = max |x - x*| / max |x*| over iL and vC.
* 325 V diode bridge: the mpmath truth of benchmarks/tools/bridge_ground_truth.py (30-40 ms
  time-weighted mean of dp - dn = 311.343881428885841697 V). err = relative.
* 12-switch three-phase VSI: trbdf2 at rtol 1e-11 (1.7e-8 from rtol 1e-10), the limit pwl+split
  converges to at first order (2.3e-4, 6.3e-5, 4.5e-5, 1.6e-5 at dt 20, 10, 5, 2.5 ns).
  Shockley rectifier: pwl at dt 1e-8 (2.7e-10 from dt 2e-8). err = max over windows of the
  exactly clipped time average (p.time_average) of each output (VSI: the three phase currents;
  rectifier: the capacitor voltage, a state: the algebraic diode node between sparse samples
  measures the output density, not the solve), per carrier / source period, over max |ref|.

Cost per switching period: accepted / rejected steps, linear-cache factorizations (segments built
+ (mask, dt) builds + numeric refactors), Newton iterations and factorizations, wall time
(median of --reps). Equal accuracy: cost at a common error, log-log interpolated along each sweep.
Timing is reported, never asserted.
"""
import argparse
import json
import math
import statistics
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

import pulsim as p

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "update-trbdf2-post-event-restart" / "probes"))
import trbdf2_baseline as B  # noqa: E402

GON, GOFF = 1e3, 1e-9

# ---------------------------------------------------------------- capture the run's cache
_CACHES = []
_ORIG_CACHE = p.PwlStateSpaceCache


def _recording_cache(*a, **k):
    c = _ORIG_CACHE(*a, **k)
    _CACHES.append(c)
    return c


p.PwlStateSpaceCache = _recording_cache

# switch_fn calls, counted in one extra (untimed) run per point
_COUNT = {"on": False, "n": 0}
_ORIG_SIM = p.simulate


def _counting_simulate(*a, **k):
    fn = k.get("switch_fn")
    if _COUNT["on"] and fn is not None:
        def counted(t):
            _COUNT["n"] += 1
            return fn(t)
        k["switch_fn"] = counted
    return _ORIG_SIM(*a, **k)


p.simulate = _counting_simulate


# ---------------------------------------------------------------- exact 2-state references
def _exact_two_state(AB, vdiode, phases, nper, x0=(0.0, 0.0)):
    """phases: [(duration, gs)], one diode decided by the sign of vdiode(x, gs, gd)."""
    def prop(A, Bv, x, dt):
        M = np.zeros((3, 3)); M[:2, :2] = A; M[:2, 2] = Bv
        E = expm(M * dt); return E[:2, :2] @ x + E[:2, 2]

    def seg(x, rem, gs):
        on = vdiode(x, gs, GOFF) >= 0
        gd = GON if on else GOFF
        A, Bv = AB(gs, gd)
        f = lambda dt: vdiode(prop(A, Bv, x, dt), gs, gd)  # noqa: E731
        f0, f1 = f(1e-15), f(rem)
        if f0 * f1 < 0:
            dtx = brentq(f, 1e-15, rem, xtol=1e-18, rtol=1e-15)
            x = prop(A, Bv, x, dtx)
            A2, B2 = AB(gs, GOFF if on else GON)
            return prop(A2, B2, x, rem - dtx)
        return prop(A, Bv, x, rem)
    x = np.array(x0, float); out = [x.copy()]
    for _ in range(nper):
        for dur, gs in phases:
            x = seg(x, dur, gs)
        out.append(x.copy())
    return np.array(out)  # x at every period start, k = 0..nper


def buck_exact_orbit(R, D, nper):
    Vin, L, C, T = B.Vin, B.Lb, B.Cb, B.T

    def AB(gs, gd):
        k = 1 / (gs + gd)
        return (np.array([[-k / L, -1 / L], [1 / C, -1 / (R * C)]]), np.array([gs * Vin * k / L, 0.0]))

    def vd(x, gs, gd):  # diode anode gnd, cathode sw
        return -((gs * Vin - x[0]) / (gs + gd))
    return _exact_two_state(AB, vd, [(D * T, GON), ((1 - D) * T, GOFF)], nper)


def boost_exact_orbit(nper):
    Vin, L, C, R, T = 12.0, 100e-6, 47e-6, 10.0, B.T

    def vsw(x, gs, gd):
        return (x[0] + gd * x[1]) / (gs + gd)

    def AB(gs, gd):
        k = 1 / (gs + gd)
        A = np.array([[-k / L, -gd * k / L], [gd * k / C, (gd * gd * k - gd - 1 / R) / C]])
        return A, np.array([Vin / L, 0.0])

    def vd(x, gs, gd):
        return vsw(x, gs, gd) - x[1]
    return _exact_two_state(AB, vd, [(0.5 * T, GON), (0.5 * T, GOFF)], nper)


def _at(times, y, tk):
    """Sample at tk (every engine lands a step boundary there: a gate rise or a multiple of dt)."""
    i = int(np.argmin(np.abs(times - tk)))
    assert abs(times[i] - tk) <= 1e-9 * B.T + 1e-15, (tk, times[i])
    return y[i]


def _orbit_err(r, exact, t_start, nper, sigs, last=20):
    t = np.asarray(r.times)
    worst = 0.0
    for j, (kind, name) in enumerate(sigs):
        y = np.asarray(r.i(name) if kind == "i" else r.v(name))
        scale = np.max(np.abs(exact[:, j]))
        for k in range(nper - last, nper + 1):
            worst = max(worst, abs(_at(t, y, t_start + k * B.T) - exact[k, j]) / scale)
    return worst


def _window_err(r, ref, outs, windows):
    worst = 0.0
    t = np.asarray(r.times)
    tr = np.asarray(ref.times)
    for name in outs:
        y, yr = np.asarray(_sig(r, name)), np.asarray(_sig(ref, name))
        scale = np.max(np.abs(yr))
        for a, b in windows:
            worst = max(worst, abs(p.time_average(t, y, a, b) - p.time_average(tr, yr, a, b)) / scale)
    return worst


def _sig(r, name):
    return r.i(name[2:]) if name.startswith("i:") else r.v(name)


# ---------------------------------------------------------------- circuits
NPER = 200


def _vsi(engine, **kw):
    import test_branch_form_switch as TB
    b, base, _ = TB._inverter(False)
    base.pop("dt"); base.pop("engine")
    base.update(kw)
    return p.simulate(b, engine=engine, **base)


CIRCUITS = {
    "buck_ccm": dict(run=lambda e, **k: B.buck(1.0, 0.5, NPER, e, **k), T=B.T, nper=NPER,
                     truth=lambda: buck_exact_orbit(1.0, 0.5, NPER), t0=B.T0,
                     sigs=[("i", "L"), ("v", "vout")]),
    "buck_dcm": dict(run=lambda e, **k: B.buck(50.0, 0.3, NPER, e, **k), T=B.T, nper=NPER,
                     truth=lambda: buck_exact_orbit(50.0, 0.3, NPER), t0=B.T0,
                     sigs=[("i", "L"), ("v", "vout")]),
    "boost": dict(run=B.boost, T=B.T, nper=200, truth=lambda: boost_exact_orbit(200), t0=0.0,
                  sigs=[("i", "L"), ("v", "vout")]),
    "vsi12": dict(run=_vsi, T=1e-4, nper=40, ref=("trbdf2", 1e-11),
                  outs=["i:La", "i:Lb", "i:Lc"], windows=[(k * 1e-4, (k + 1) * 1e-4) for k in range(40)]),
    "bridge": dict(run=B.bridge_c, T=20e-3, nper=2, bridge_truth=311.343881428885841697),
    "shockley": dict(run=B.shockley_rect, T=1e-3, nper=5, ref=("pwl", 1e-8),
                     outs=["out"], windows=[(k * 1e-3, (k + 1) * 1e-3) for k in range(5)]),
}
RTOLS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
NS = [20, 50, 100, 200, 500, 1000, 2000]
BRIDGE_DT = [1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
SHOCKLEY_DT = [2e-6, 1e-6, 5e-7, 2e-7, 1e-7]


def _kw(engine, val):
    if engine == "trbdf2":
        return {"rtol": val, "atol": 1e-3 * val}
    kw = {"dt": val}
    if engine == "pwl+split":
        kw.update(discontinuity_restart=True, enable_substep_state_correction=True)
    return kw


def run_one(c, engine, val, reps):
    eng = "pwl" if engine.startswith("pwl") else engine
    walls = []
    for _ in range(reps):
        _CACHES.clear()
        t0 = time.perf_counter()
        r = c["run"](eng, **_kw(engine, val))
        walls.append(time.perf_counter() - t0)
    cache = _CACHES[-1]
    m = cache.metrics()
    _COUNT.update(on=True, n=0)
    c["run"](eng, **_kw(engine, val))
    _COUNT["on"] = False
    gate_calls = _COUNT["n"]
    st = getattr(r, "_trbdf2_stats", None) or {}
    nper = c["nper"]
    row = dict(engine=engine, val=val, wall=statistics.median(walls),
               steps=st.get("n_accept", len(r.times) - 1), rejects=st.get("n_reject", 0),
               factors=int(cache.num_built_segments() + m.event_builds + m.event_refactors),
               factor_hits=int(m.event_hits), newton_it=int(r.newton.iterations),
               newton_fact=int(r.newton.factorizations), solves=int(st.get("n_solves", 0)),
               gate_calls=gate_calls)
    for k in ("steps", "rejects", "factors", "newton_it", "newton_fact", "gate_calls"):
        row[k + "_per"] = row[k] / nper
    row["wall_per_ms"] = 1e3 * row["wall"] / nper
    return r, row


def accuracy(c, r, ref_run):
    if "truth" in c:
        return _orbit_err(r, c["_exact"], c["t0"], c["nper"], c["sigs"])
    if "bridge_truth" in c:
        v = np.asarray(r.v("dp")) - np.asarray(r.v("dn"))
        return abs(p.time_average(np.asarray(r.times), v, 0.030, 0.040) - c["bridge_truth"]) / c["bridge_truth"]
    return _window_err(r, ref_run, c["outs"], c["windows"])


def at_error(rows, target):
    """log-log interpolated cost at `target` along a sweep (None if the sweep does not bracket it)."""
    pts = sorted(((r["err"], r) for r in rows if r["err"] > 0), key=lambda q: q[0])
    for (e0, r0), (e1, r1) in zip(pts, pts[1:]):
        if e0 <= target <= e1:
            w = (math.log(target) - math.log(e0)) / (math.log(e1) - math.log(e0)) if e1 > e0 else 0
            return {k: math.exp((1 - w) * math.log(max(r0[k], 1e-30)) + w * math.log(max(r1[k], 1e-30)))
                    for k in ("steps_per", "factors_per", "newton_it_per", "wall_per_ms")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--engines", default="trbdf2,pwl,pwl+split")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(CIRCUITS)
    out = {}
    for name in names:
        c = CIRCUITS[name]
        ref_run = None
        if "truth" in c:
            c["_exact"] = c["truth"]()
        elif "ref" in c:
            eng, dt = c["ref"]
            ref_run, _ = run_one(c, eng, dt, 1)
            half, _ = run_one(c, eng, 2 * dt, 1)
            print(f"{name}: reference {eng} dt={dt:g}; vs 2dt {_window_err(half, ref_run, c['outs'], c['windows']):.2e}")
        rows = []
        for engine in a.engines.split(","):
            if engine == "trbdf2":
                vals = RTOLS
            elif name == "bridge":
                vals = BRIDGE_DT
            elif name == "shockley":
                vals = SHOCKLEY_DT
            else:
                vals = [c["T"] / n for n in NS]
            if name == "shockley" and engine == "pwl+split":
                continue  # no switch: pwl+split == pwl
            for v in vals:
                try:
                    r, row = run_one(c, engine, v, a.reps)
                    row["err"] = accuracy(c, r, ref_run)
                except Exception as ex:  # noqa: BLE001
                    print(f"{name} {engine} {v:g}: {type(ex).__name__}: {str(ex)[:120]}")
                    continue
                rows.append(row)
                print(f"{name:9s} {engine:9s} {v:9.2e} err {row['err']:.2e} steps/T {row['steps_per']:8.1f} "
                      f"rej/T {row['rejects_per']:6.1f} fact/T {row['factors_per']:8.1f} hits {row['factor_hits']:7d} "
                      f"newt/T {row['newton_it_per']:7.1f} nfact/T {row['newton_fact_per']:7.1f} "
                      f"gate/T {row['gate_calls_per']:7.1f} "
                      f"wall/T {row['wall_per_ms']:8.4f} ms", flush=True)
        out[name] = rows
        for tgt in (1e-3, 1e-4, 1e-5):
            line = [f"  at err {tgt:.0e}:"]
            for engine in a.engines.split(","):
                ae = at_error([r for r in rows if r["engine"] == engine], tgt)
                if ae:
                    line.append(f"{engine} steps {ae['steps_per']:.1f} fact {ae['factors_per']:.1f} "
                                f"wall {ae['wall_per_ms']:.4f} ms/T;")
            print(" ".join(line))
        if a.json:
            Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
