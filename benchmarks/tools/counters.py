#!/usr/bin/env python3
"""Counter-based performance gate: deterministic work counters per
circuit x engine, compared with ``benchmarks/manifests/perf_counters.yaml``.
Wall time is printed for information and never asserted.

Counters are the ones the kernel already reports (no kernel change):
pwl ``n_steps`` / ``n_dt_retries`` / ``n_event_iterations`` /
``n_commutations`` and the PWL cache metrics (``on_cache``); trbdf2 its
stats dict; dsed (``integrator='rk45'``, so the ``'auto'`` selection is not
pinned) its accept / reject / event counts.

    counters.py            # check
    counters.py --freeze   # rewrite the manifest (review the diff)
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v1_reader  # noqa: E402
from analytic import BY_ID  # noqa: E402

import pulsim  # noqa: E402

MANIFEST = ROOT / "manifests" / "perf_counters.yaml"
SLACK = {"pwl": 1.0, "trbdf2": 1.02, "dsed": 1.02}
CACHE_FIELDS = ("event_builds", "event_hits", "event_refactors", "rank1_hits",
                "multi_bit_rank1_hits", "full_refactor_hits", "fallbacks", "segment_evictions")
TRBDF2_FIELDS = ("n_accept", "n_reject", "n_solves", "n_newton_retries", "n_gate_events",
                 "n_diode_events", "n_chatter_breaks", "n_forced_accepts")
DSED_FIELDS = ("n_accept", "n_reject", "n_events", "n_rk45_steps", "n_bdf2_steps")


def _ladder(n=500):
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V", "n0", "gnd", 10.0)
    for k in range(n):
        b.add_resistor(f"R{k}", f"n{k}", f"n{k + 1}", 10.0)
        b.add_capacitor(f"C{k}", f"n{k + 1}", "gnd", 1e-9)
    return b, None


def _saturable():
    b = pulsim.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "a", 5.0)
    b.add_saturable_inductor("L", "a", "gnd", 1e-3, 1.0, 1e-4)
    return b, None


def _analytic(cid):
    c = BY_ID[cid]
    return lambda: c.build(1.0), c.t_end, c.dt


def _v1(rel):
    case = v1_reader.load(ROOT / rel)
    if not case.uic:
        raise ValueError(f"{rel}: counter circuits must start at rest")
    return lambda: case.build()[:2], case.t_end, case.dt


# name -> (build() -> (builder, switch_fn), t_end, dt)
CIRCUITS = {
    "rc_charge": _analytic("rc_charge"),
    "ladder500_rc": (lambda: _ladder(), 2e-5, 1e-8),
    "buck_ccm": _analytic("buck_ccm"),
    "buck_dcm": _analytic("buck_dcm"),
    "hard_switched_rc": _analytic("hard_switched_rc"),
    "shockley_op": _analytic("shockley_op"),
    "saturable_rl": (lambda: _saturable(), 2e-3, 1e-6),
    "half_bridge_inverter": _v1("circuits/switching/inverter/half_bridge_inverter.yaml"),
    "three_phase_diode_rectifier": _v1("circuits/three_phase/three_phase_diode_rectifier.yaml"),
    "buck_mosfet_nonlinear": _v1("circuits/switching/buck/buck_mosfet_nonlinear.yaml"),
    "flyback_converter": _v1("circuits/switching/isolated/flyback_converter.yaml"),
    "diode_bridge_rectifier": _v1("circuits/diodes/diode_bridge_rectifier.yaml"),
}


def count(name: str, engine: str) -> dict:
    """Counters of one run; raises the engine's refusal unchanged."""
    build, t_end, dt = CIRCUITS[name]
    b, sw = build()
    kw = {"switch_fn": sw} if sw is not None else {}
    if engine == "pwl":
        caches = []
        res = pulsim.simulate(b, t_end, dt, engine="pwl", on_cache=caches.append, **kw)
        m = caches[0].metrics()
        out = {f: int(getattr(m, f)) for f in CACHE_FIELDS}
        out["num_built_segments"] = int(caches[0].num_built_segments())
        out.update(n_steps=len(res.times), n_dt_retries=len(res.dt_retries),
                   n_event_iterations=int(sum(res.event_iteration_count)),
                   n_commutations=len(res.commutation_events))
        return out
    if engine == "trbdf2":
        res = pulsim.simulate(b, t_end, engine="trbdf2", **kw)
        return {f: int(res._trbdf2_stats[f]) for f in TRBDF2_FIELDS}
    res = pulsim.simulate(b, t_end, engine="dsed", integrator="rk45", **kw)
    return {f: int(getattr(res, f)) for f in DSED_FIELDS}


def check(frozen: dict, name: str, engine: str, got: dict) -> tuple:
    """(failures, reminders) for one cell."""
    bad, better = [], []
    for k, v in frozen.items():
        limit = v * SLACK[engine]
        if got[k] > limit:
            bad.append(f"{name} / {engine} / {k}: {got[k]} > frozen {v}"
                       + (f" x {SLACK[engine]}" if SLACK[engine] != 1.0 else ""))
        elif v and got[k] < 0.95 * v:
            better.append(f"{name} / {engine} / {k}: {got[k]} < 0.95 x frozen {v} - re-freeze")
    return bad, better


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    frozen = (yaml.safe_load(MANIFEST.read_text()) or {}).get("counters", {}) if MANIFEST.exists() else {}
    out, fails = {}, 0
    for name in CIRCUITS:
        out[name] = {}
        for engine in ("pwl", "trbdf2", "dsed"):
            t0 = time.perf_counter()
            try:
                got = count(name, engine)
            except Exception as ex:  # noqa: BLE001 - a refusal is recorded, not gated
                out[name][engine] = {"refused": type(ex).__name__}
                continue
            out[name][engine] = got
            wall = time.perf_counter() - t0
            if not args.freeze and "refused" not in frozen.get(name, {}).get(engine, {"refused": 1}):
                bad, better = check(frozen[name][engine], name, engine, got)
                fails += len(bad)
                for line in bad + better:
                    print(line)
            print(f"{name:28s} {engine:6s} {wall * 1e3:8.1f} ms (information only)")
    if args.freeze:
        head = [line for line in MANIFEST.read_text().splitlines() if line.startswith("#")] \
            if MANIFEST.exists() else []
        MANIFEST.write_text("\n".join(head) + "\n" + yaml.safe_dump({"counters": out}, sort_keys=False))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
