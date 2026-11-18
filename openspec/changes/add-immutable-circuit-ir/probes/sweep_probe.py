"""Baseline / after records of add-immutable-circuit-ir (tasks 0.2, 10.2).

    python sweep_probe.py components [--out F]   # components() dump digests
    python sweep_probe.py sweep [--reps N]       # 100-point sweeps, A/B
    python sweep_probe.py private                # builder private attributes

``sweep`` runs a 100-point parameter sweep on the audit's buck (L) and on
the n = 1502 RLC ladder (Rload), serially, in two forms, interleaved:

* ``factory`` — ``builder_factory(**p)`` rebuilt for every point (today's
  ``pulsim.sweep``), the baseline;
* ``circuit`` — one ``Circuit`` and ``with_params`` per point (only when
  the tree has it).

Per form it reports the COUNTERS (circuit builds, symbolic analyses,
numeric factorisations: segment + event builds + event refactors) and the
wall time, best of N. Times are reported, never asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pulsim as p

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "benchmarks" / "tools")]

# ---- cache capture ----------------------------------------------------------
_caches: list = []
_orig_cache = p.PwlStateSpaceCache


def _wrap_cache(*a, **k):
    c = _orig_cache(*a, **k)
    _caches.append(c)
    return c


p.__dict__["PwlStateSpaceCache"] = _wrap_cache


# ---- the two sweeps ---------------------------------------------------------
def buck(L=100e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_mosfet("Q1", "vin", "sw")
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", 10.0)
    return b


def ladder(Rload=50.0, N=500):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vs", "n0", "gnd", 0.0, 10.0, 1e3)
    for k in range(N):
        b.add_inductor(f"L{k}", f"n{k}", f"m{k}", 1e-5)
        b.add_resistor(f"R{k}", f"m{k}", f"n{k+1}", 0.1)
        b.add_capacitor(f"C{k}", f"n{k+1}", "gnd", 1e-7)
    b.add_resistor("Rload", f"n{N}", "gnd", Rload)
    return b


def _buck_kw():
    b = buck()
    fn = p.make_pwm_switch_fn(100e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)
    return dict(t_end=2e-4, dt=1e-7, engine="pwl", switch_fn=fn)


SWEEPS = {
    # name: (factory, field, values, simulate kwargs)
    "buck": (buck, "L.L", "L", np.geomspace(50e-6, 500e-6, 100), _buck_kw),
    "ladder1502": (ladder, "Rload.R", "Rload", np.geomspace(5.0, 500.0, 100),
                   lambda: dict(t_end=2e-6, dt=1e-7, engine="pwl")),
}


def _factor_count(c) -> int:
    m = c.metrics()
    return int(c.num_built_segments()) + int(m.event_builds) + int(m.event_refactors)


def run_form(name, form):
    factory, field, pname, values, kw_fn = SWEEPS[name]
    kw = kw_fn()
    _caches.clear()
    builds = 0
    outs = []
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if form == "factory":
            for x in values:
                b = factory(**{pname: float(x)})
                builds += 1
                outs.append(p.simulate(b, **kw).states[-1].copy())
        else:
            c = factory().build()
            builds += 1
            for x in values:
                outs.append(p.simulate(c.with_value(field, float(x)), **kw).states[-1].copy())
    wall = time.perf_counter() - t0
    return dict(
        wall_s=wall, builds=builds, runs=len(values),
        symbolic_analyses=sum(int(c.metrics().symbolic_analyses) for c in _caches),
        numeric_factorisations=sum(_factor_count(c) for c in _caches),
        final=np.array(outs))


def cmd_sweep(reps):
    forms = ["factory"] + (["circuit"] if hasattr(p.Circuit, "with_value") else [])
    for name in SWEEPS:
        best = {f: None for f in forms}
        for _ in range(reps):
            for f in forms:  # interleaved A/B
                r = run_form(name, f)
                if best[f] is None or r["wall_s"] < best[f]["wall_s"]:
                    best[f] = r
        for f in forms:
            r = best[f]
            print(f"{name:11s} {f:8s} best-of-{reps} wall {r['wall_s']*1e3:9.1f} ms "
                  f"({r['wall_s']*1e3/r['runs']:.2f} ms/point)  builds {r['builds']:3d}  "
                  f"symbolic_analyses {r['symbolic_analyses']:3d}  "
                  f"numeric_factorisations {r['numeric_factorisations']:4d}")
        if len(forms) == 2:
            same = np.array_equal(best["factory"]["final"], best["circuit"]["final"])
            print(f"{name:11s} final states bit-identical across forms: {same}")


# ---- components() dumps ------------------------------------------------------
def circuits():
    """(id, builder) for every shipped example and harness circuit."""
    import v1_reader
    out = []
    for f in sorted((ROOT / "examples").glob("*.yaml")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append((f"examples/{f.name}", p.load_yaml_file(str(f)).builder))
        except Exception as exc:  # noqa: BLE001 — recorded as unloadable
            out.append((f"examples/{f.name}", exc))
    for f in sorted((ROOT / "benchmarks" / "circuits").rglob("*.yaml")):
        rel = f.relative_to(ROOT).as_posix()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append((rel, v1_reader.load(f).build()[0]))
        except v1_reader.Unmapped as exc:
            out.append((rel, exc))
    return out


def digest(b) -> str:
    return hashlib.sha256(json.dumps(b.components(), sort_keys=True,
                                     default=repr).encode()).hexdigest()[:16]


def cmd_components(out):
    rec = {}
    for cid, b in circuits():
        rec[cid] = (f"unloadable: {type(b).__name__}" if isinstance(b, Exception)
                    else {"n": len(b.components()), "sha": digest(b)})
    Path(out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    n_ok = sum(isinstance(v, dict) for v in rec.values())
    print(f"{len(rec)} circuits, {n_ok} loaded -> {out}")


def cmd_private():
    pat = re.compile(r"(\w+)\._(c_blocks|pulsim_ic|pulsim_aliases|shockley_params)\s*=(?!=)")
    rd = re.compile(r"getattr\((\w+),\s*\"_(c_blocks|pulsim_ic|pulsim_aliases|shockley_params)\"")
    for f in sorted((ROOT / "python" / "pulsim").rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for kind, rx in (("set", pat), ("read", rd)):
                if rx.search(line):
                    print(f"{kind:4s} {f.relative_to(ROOT)}:{i}: {line.strip()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "components", "private"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).with_name("components_baseline.json")))
    a = ap.parse_args()
    {"sweep": lambda: cmd_sweep(a.reps), "components": lambda: cmd_components(a.out),
     "private": cmd_private}[a.cmd]()
