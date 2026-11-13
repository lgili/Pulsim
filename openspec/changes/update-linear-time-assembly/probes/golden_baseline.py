"""Bit-identity oracle for the Phase 1 linear-algebra changes.

    python golden_baseline.py record  [--dir D]   # save goldens (pre-change tree)
    python golden_baseline.py compare [--dir D]   # strict byte compare + counter diff
    python golden_baseline.py perf    [--reps N]  # ns/step and build cost (report only)
    python golden_baseline.py gate    [--dir D]   # rounding gate (see gate())

Result arrays (times, states, switch states, TR-BDF2 stats, error text)
are compared byte for byte: any difference, even a sign of zero, fails.
Kernel counters (cache metrics, ``res.counters``) are printed as
before -> after; they are what a speed change is supposed to move.
Used by update-linear-time-assembly, update-trbdf2-factor-reuse and
update-newton-persistent-solver.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pulsim as p

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DIR = ROOT / ".scratch" / "golden"

# ---- cache capture: every PwlStateSpaceCache built during a run ------------
_caches: list = []
_orig_cache = p.PwlStateSpaceCache


def _wrap_cache(*a, **k):
    c = _orig_cache(*a, **k)
    _caches.append(c)
    return c


p.__dict__["PwlStateSpaceCache"] = _wrap_cache


# ---- circuits ---------------------------------------------------------------
def buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def vsi():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdc", "dcp", "gnd", 600.0)
    v = p.add_three_phase_vsi(b, "INV", vdc_pos="dcp", vdc_neg="gnd", out_a="a", out_b="b", out_c="c")
    for ph in "abc":
        b.add_resistor(f"R{ph}", ph, f"m{ph}", 10.0)
        b.add_inductor(f"L{ph}", f"m{ph}", "n", 5e-3)
    legs = p.ThreePhaseLegIndices(*v.switch_indices)
    fn = p.make_three_phase_spwm_fn(10e3, 50.0, 0.8, legs, b.graph.num_switches, 1e-6)
    return b, fn


def rlc_ladder(N):
    """n = 3N + 2 unknowns."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
    for k in range(N):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
    b.add_resistor("Rload", f"n{N}", "gnd", 50.0)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def shockley_rect(N=16):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
    b.add_shockley_diode("D1", "a", "n0", 1e-9)
    for k in range(N):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-6)
    b.add_capacitor("Co", f"n{N}", "gnd", 1e-4)
    b.add_resistor("RL", f"n{N}", "gnd", 50.0)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def shockley_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_shockley_diode("D1", "gnd", "sw", 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return b, fn


def mos_l1():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vdd", "vdd", "gnd", 24.0)
    b.add_pulse_voltage_source("Vg", "g", "gnd", 0.0, 10.0, 0.0, 5e-6, 10e-6, 1e-7, 1e-7)
    b.add_mosfet_level1("M1", "sw", "gnd", "g", 5.0, 3.0)
    b.add_inductor("L", "vdd", "sw", 100e-6)
    b.add_resistor("Rs", "sw", "out", 1.0)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    b.add_resistor("RL", "out", "gnd", 20.0)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def transformer():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "p", "gnd", 0.0, 100.0, 1e3)
    b.add_resistor("Rp", "p", "p1", 0.5)
    b.add_transformer("T1", "p1", "gnd", "s", "gnd", 1e-3, 4e-3, 0.98)
    b.add_resistor("RL", "s", "gnd", 20.0)
    b.add_capacitor("Cs", "s", "gnd", 1e-7)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def saturable():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 20.0, 1e3)
    b.add_resistor("R", "a", "b", 1.0)
    b.add_saturable_inductor("Ls", "b", "gnd", 1e-3, 2.0, 1e-5)
    return b, p.make_combined_switch_fn(b.graph.num_switches, [])


def sync_buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_switch("Q2", "sw", "gnd", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    fn = p.make_dead_time_pwm_pair_fn(100e3, 0.5, b.switch_index_of("Q1"),
                                      b.switch_index_of("Q2"), b.graph.num_switches, 0.0)
    return b, fn


# name -> (circuit factory, simulate kwargs)
CASES = {
    "buck_pwl": (buck, dict(t_end=2e-3, dt=1e-7, engine="pwl")),
    "buck_trbdf2": (buck, dict(t_end=4e-3, engine="trbdf2")),
    "vsi_pwl": (vsi, dict(t_end=10e-3, dt=1e-7, engine="pwl")),
    "vsi_trbdf2": (vsi, dict(t_end=20e-3, engine="trbdf2")),
    "rlc152_pwl": (lambda: rlc_ladder(50), dict(t_end=1e-3, dt=1e-7, engine="pwl")),
    "rlc152_trbdf2": (lambda: rlc_ladder(50), dict(t_end=2e-3, engine="trbdf2")),
    "rlc1502_pwl": (lambda: rlc_ladder(500), dict(t_end=2e-4, dt=1e-7, engine="pwl")),
    "rlc1502_trbdf2": (lambda: rlc_ladder(500), dict(t_end=2e-3, engine="trbdf2")),
    "shockrect51_pwl": (shockley_rect, dict(t_end=5e-3, dt=1e-6, engine="pwl")),
    "shockrect51_trbdf2": (shockley_rect, dict(t_end=20e-3, engine="trbdf2")),
    "shockrect483_pwl": (lambda: shockley_rect(160), dict(t_end=4e-4, dt=1e-6, engine="pwl")),
    "shockbuck_pwl": (shockley_buck, dict(t_end=1e-3, dt=1e-7, engine="pwl")),
    "shockbuck_trbdf2": (shockley_buck, dict(t_end=1e-3, engine="trbdf2")),
    "mosl1_pwl": (mos_l1, dict(t_end=5e-4, dt=1e-7, engine="pwl")),
    "mosl1_trbdf2": (mos_l1, dict(t_end=2e-4, engine="trbdf2")),
    "xfmr_pwl": (transformer, dict(t_end=2e-3, dt=1e-7, engine="pwl")),
    "xfmr_trbdf2": (transformer, dict(t_end=2e-3, engine="trbdf2")),
    "satind_pwl": (saturable, dict(t_end=2e-3, dt=1e-7, engine="pwl")),
    "syncbuck_dsed": (sync_buck, dict(t_end=2e-3, engine="dsed")),
    # discontinuity_restart=True (add-discontinuity-restart-rule): the
    # SDIRK restart stages solve on the event table at 2*gamma*dt.
    "buck_pwl_restart": (buck, dict(t_end=2e-3, dt=1e-7, engine="pwl",
                                    discontinuity_restart=True)),
    "vsi_pwl_restart": (vsi, dict(t_end=10e-3, dt=1e-7, engine="pwl",
                                  discontinuity_restart=True)),
}


def _int_fields(obj) -> dict:
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            out[k] = int(v)
    return out


def run_case(name):
    factory, kw = CASES[name]
    b, fn = factory()
    _caches.clear()
    arrays: dict = {}
    counters: dict = {}
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = p.simulate(b, switch_fn=fn, **kw)
    except Exception as e:  # a refusal is a result too
        arrays["error"] = np.frombuffer(f"{type(e).__name__}: {e}".encode(), dtype=np.uint8)
        return arrays, counters, time.perf_counter() - t0, 0
    wall = time.perf_counter() - t0
    arrays["times"] = np.asarray(res.times, dtype=np.float64)
    arrays["states"] = np.asarray(res.states, dtype=np.float64)
    try:
        arrays["switch_states"] = np.asarray(res.switch_states)
    except Exception as e:
        arrays["switch_states_error"] = np.frombuffer(str(e).encode(), dtype=np.uint8)
    st = getattr(res, "_trbdf2_stats", None)
    if st:
        keys = sorted(k for k, v in st.items() if isinstance(v, (int, float)))
        arrays["trbdf2_stats"] = np.array([float(st[k]) for k in keys])
        arrays["trbdf2_stats_keys"] = np.frombuffer(",".join(keys).encode(), dtype=np.uint8)
    rc = getattr(res, "counters", None)
    if rc:
        for k, v in dict(rc).items():
            counters[f"res.{k}"] = int(v)
    for k in ("n_restart_steps", "n_restart_factorizations"):
        if getattr(res, k, 0):
            counters[f"res.{k}"] = int(getattr(res, k))

    nw = getattr(res, "newton", None)
    if nw is not None:
        # Iterations of every Newton call, in call order: the per-step
        # iteration counts the rounding gate requires to be identical.
        arrays["newton_trace"] = np.asarray(nw.iterations_per_solve, dtype=np.uint16)
        for k in ("solves", "iterations", "analyses", "factorizations",
                  "partial_refactors", "lm_attempts"):
            counters[f"newton.{k}"] = int(getattr(nw, k))
    for i, c in enumerate(_caches):
        for k, v in _int_fields(c.metrics()).items():
            counters[f"cache.{k}"] = counters.get(f"cache.{k}", 0) + v
    counters["n_caches"] = len(_caches)
    return arrays, counters, wall, len(arrays["times"])


# Counter keys that are themselves results (step sequence): byte-compared.
def record(d: Path, names):
    d.mkdir(parents=True, exist_ok=True)
    for name in names:
        arrays, counters, wall, n = run_case(name)
        np.savez(d / f"{name}.npz", **arrays,
                 counter_keys=np.array(sorted(counters)),
                 counter_vals=np.array([counters[k] for k in sorted(counters)], dtype=np.int64))
        print(f"recorded {name:20s} samples={n:7d} wall={wall*1e3:8.1f} ms {counters}")


def compare(d: Path, names) -> int:
    bad = 0
    for name in names:
        g = np.load(d / f"{name}.npz")
        arrays, counters, wall, n = run_case(name)
        keys = sorted(set(k for k in g.files if not k.startswith("counter_")) | set(arrays))
        verdict = []
        for k in keys:
            if k not in g.files and k == "newton_trace":
                continue  # golden recorded before the trace existed
            if k not in arrays or k not in g.files:
                verdict.append(f"{k}:MISSING")
                bad += 1
                continue
            a, r = arrays[k], g[k]
            eq = (a.shape == r.shape and a.dtype == r.dtype and np.array_equal(a, r)
                  and a.tobytes() == r.tobytes())
            if not eq:
                bad += 1
                extra = ""
                if a.shape == r.shape and a.dtype.kind == "f":
                    extra = f" max|d|={np.max(np.abs(a - r)):.3e}"
                verdict.append(f"{k}:DIFF{a.shape}vs{r.shape}{extra}")
        old = dict(zip(g["counter_keys"].tolist(), g["counter_vals"].tolist()))
        moved = {k: f"{old.get(k, '-')}->{counters.get(k, '-')}"
                 for k in sorted(set(old) | set(counters)) if old.get(k) != counters.get(k)}
        status = "IDENTICAL" if not verdict else "NOT IDENTICAL " + " ".join(verdict)
        print(f"{name:20s} {status}  counters moved: {moved or 'none'}")
    print("GOLDEN COMPARE:", "ALL IDENTICAL" if bad == 0 else f"{bad} DIFFERENCES")
    return 1 if bad else 0


def gate(d: Path, names, tol=1e-9) -> int:
    """Rounding gate (maintainer, 2026-11-13) against goldens recorded
    with the Newton trace: step count, switch-state sequence and the
    iteration count of every Newton call identical; states within `tol`
    relative per variable (|a - r| / range of r over the run); a case
    without Newton calls must stay byte-identical."""
    bad = 0
    for name in names:
        g = np.load(d / f"{name}.npz")
        arrays, counters, wall, n = run_case(name)
        msgs, ok = [], True
        if "error" in g.files or "error" in arrays:
            ok = "error" in g.files and "error" in arrays and np.array_equal(g["error"], arrays["error"])
            msgs.append("error text " + ("same" if ok else "DIFFERS"))
        else:
            newton = "newton_trace" in g.files and g["newton_trace"].size > 0
            for k in ("switch_states", "newton_trace"):
                if k in g.files or k in arrays:
                    same = k in g.files and k in arrays and np.array_equal(g[k], arrays[k])
                    ok &= same
                    if not same:
                        msgs.append(f"{k} DIFFERS")
            if g["times"].shape != arrays["times"].shape:
                ok = False
                msgs.append(f"steps {g['times'].shape[0]} -> {arrays['times'].shape[0]}")
            else:
                r, a = g["states"], arrays["states"]
                rng = r.max(axis=0) - r.min(axis=0)
                rng = np.where(rng > 0, rng, np.maximum(np.abs(r).max(axis=0), 1.0))
                rel = float(np.max(np.abs(a - r) / rng)) if r.size else 0.0
                trel = float(np.max(np.abs(arrays["times"] - g["times"])) / max(g["times"][-1], 1e-300))
                exact = all(np.array_equal(g[k], arrays[k]) for k in ("times", "states"))
                if not newton:
                    ok &= exact
                else:
                    ok &= rel <= tol and trel <= tol
                msgs.append(f"max rel state diff {rel:.2e}, times {trel:.2e}"
                            + (" (bit-identical)" if exact else ""))
        old = dict(zip(g["counter_keys"].tolist(), g["counter_vals"].tolist()))
        nk = [k for k in sorted(set(old) | set(counters)) if k.startswith("newton.")]
        moved = {k.split(".", 1)[1]: f"{old.get(k, '-')}->{counters.get(k, '-')}" for k in nk}
        bad += 0 if ok else 1
        print(f"{name:20s} {'PASS' if ok else 'FAIL'}  {'; '.join(msgs)}  newton {moved or '-'}")
    print("ROUNDING GATE:", "ALL PASS" if bad == 0 else f"{bad} FAIL")
    return 1 if bad else 0


def perf(names, reps):
    for name in names if reps > 0 else []:
        best, n = 1e99, 0
        for _ in range(reps):
            _, _, wall, n = run_case(name)
            best = min(best, wall)
        print(f"perf {name:20s} samples={n:7d} best={best*1e3:9.2f} ms  {best/max(n-1, 1)*1e9:10.0f} ns/sample")
    # segment build cost: eager build of the RLC ladder (one mask) and of a switched ladder
    def build_cost(b, reps_b=3, dt=1e-7):
        bb = 1e99
        for _ in range(reps_b):
            c = _orig_cache(b.graph, b.pool)
            t0 = time.perf_counter()
            c.build(dt)
            bb = min(bb, time.perf_counter() - t0)
        return bb, c
    for N in (1, 5, 50, 500, 5000):
        b, _ = rlc_ladder(N)
        w, c = build_cost(b)
        print(f"build rlc n={b.pool.state_size(b.graph):6d} {w*1e6:11.1f} us/seg  {_int_fields(c.metrics())}")
    b, _ = vsi()
    w, c = build_cost(b, 2)
    segs = c.num_built_segments()
    print(f"build vsi n={b.pool.state_size(b.graph)} masks={segs} {w/segs*1e6:9.1f} us/mask  {_int_fields(c.metrics())}")
    for N, K in ((100, 8), (1000, 6)):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
        for k in range(N):
            b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
            b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
            b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
        for j in range(K):
            node = f"n{1 + (j * (N - 1)) // max(K - 1, 1)}"
            b.add_switch(f"S{j}", node, f"s{j}", 1e3, 1e-9)
            b.add_resistor(f"RS{j}", f"s{j}", "gnd", 5.0)
        b.add_resistor("Rload", f"n{N}", "gnd", 50.0)
        w, c = build_cost(b, 1)
        segs = c.num_built_segments()
        print(f"build sw_ladder n={b.pool.state_size(b.graph):5d} masks={segs} {w/segs*1e6:9.1f} us/mask  {_int_fields(c.metrics())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "compare", "perf", "gate"])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    names = a.only or list(CASES)
    if a.mode == "record":
        record(a.dir, names)
        return 0
    if a.mode == "compare":
        return compare(a.dir, names)
    if a.mode == "gate":
        return gate(a.dir, names)
    perf(names, a.reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
