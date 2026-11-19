#!/usr/bin/env python3
"""The fixed-step engine's results against a STORED baseline: run each
circuit, reduce every public result attribute to a digest (arrays bit for
bit, every stats key) and record or compare the digests.

    pwl_identity.py record DIR --goldens --reference   # at the base commit
    pwl_identity.py compare DIR --goldens --reference  # at the tip
    [--circuits GLOB,...] [--configs pwl,pwl+restart,pwl+split]
    [--static]   # the static path (``run_transient`` on a dt = 0 cache)

The refactor-unified-transient-stepper identity gate since the legacy loops
were deleted (DECISIONS 20): with one driver left there is no twin to run
against, so the base commit's own results are the reference. Bit identity
is a LOCAL gate (this machine, this build). Prints ``IDENTICAL <name>`` or
``DIFFER <name>: <first differing key>``; exit status 1 when any case
differs or is missing from the baseline.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np


def _value(v, depth=0):
    """A comparable, exact form of a result attribute."""
    if isinstance(v, (bool, int, str, type(None))):
        return v
    if isinstance(v, float):
        return np.float64(v)
    if isinstance(v, dict):
        return {k: _value(x, depth + 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, np.ndarray):
        return np.array(v)
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(x, (int, float, bool)) for x in v):
            return np.asarray(v)
        return [_value(x, depth + 1) for x in v]
    if depth > 3:
        return repr(v)
    try:
        a = np.asarray(v)
        if a.dtype != object:
            return np.array(a)
    except Exception:  # noqa: BLE001 - not array-like
        pass
    return {k: _value(getattr(v, k), depth + 1) for k in dir(v)
            if not k.startswith("_") and not callable(getattr(v, k, None))}


def fingerprint(res) -> dict:
    """Every public, non-callable attribute of a result (arrays exact)."""
    out = {}
    for k in dir(res):
        if k.startswith("__"):
            continue
        try:
            v = getattr(res, k)
        except Exception as e:  # noqa: BLE001 - a property may refuse
            out[k] = f"<{type(e).__name__}>"
            continue
        if callable(v):
            continue
        out[k] = _value(v)
    return out


def digests(fp, path="") -> dict:
    """Flatten a fingerprint to {key path: digest}: an array by dtype,
    shape and bytes; anything else by its repr (exact for floats)."""
    if isinstance(fp, np.generic):
        fp = np.asarray(fp)
    if isinstance(fp, np.ndarray):
        a = np.ascontiguousarray(fp)
        h = hashlib.sha256(f"{a.dtype}{a.shape}".encode("utf-8"))
        h.update(a.tobytes() if a.dtype != object else repr(a.tolist()).encode("utf-8"))
        return {path: h.hexdigest()}
    if isinstance(fp, dict):
        out = {}
        for k in sorted(fp, key=str):
            out.update(digests(fp[k], f"{path}.{k}"))
        return out or {path: "{}"}
    if isinstance(fp, list):
        out = {f"{path}#len": str(len(fp))}
        for i, x in enumerate(fp):
            out.update(digests(x, f"{path}[{i}]"))
        return out
    return {path: f"{type(fp).__name__}:{fp!r}"}


def run(fn) -> dict:
    """The case's digests; an exception is its type and message (a refusal
    must stay the same refusal)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return digests(fingerprint(fn()))
        except Exception as e:  # noqa: BLE001 - recorded, not swallowed
            return {"raised": f"{type(e).__name__}: {e}"}


def first_difference(a: dict, b: dict) -> str | None:
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            return f"{k}: {'present on one side only' if k not in a or k not in b else 'differs'}"
    return None


def record_or_compare(mode: str, store: Path, cases) -> int:
    """``record``: write {name: digests} to ``store``; ``compare``: check
    each case against it."""
    if mode == "record":
        out = {}
        for name, fn in cases:
            out[name] = run(fn)
            print(f"RECORDED {name}", flush=True)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(json.dumps(out, sort_keys=True), encoding="utf-8")
        print(f"{len(out)} recorded in {store}")
        return 0
    base = json.loads(store.read_text(encoding="utf-8"))
    n = bad = 0
    for name, fn in cases:
        n += 1
        if name not in base:
            bad += 1
            print(f"DIFFER {name}: not in the baseline", flush=True)
            continue
        d = first_difference(base[name], run(fn))
        bad += d is not None
        tag = " (raised)" if "raised" in base[name] else ""
        print(f"DIFFER {name}: {d}" if d else f"IDENTICAL {name}{tag}", flush=True)
    print(f"{n - bad}/{n} identical")
    return 1 if bad else 0


ROOT = Path(__file__).resolve().parents[1]


def golden_cases():
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import goldens  # noqa: PLC0415
    for name, fn, _names, _newton in goldens.cases():
        yield name, fn


# The fixed-engine configurations this tool records (the pre-release names of
# reference_status; run them under next_defaults.legacy() for the old meaning).
CONFIGS = {"pwl": ("pwl", {}), "pwl+restart": ("pwl", {"discontinuity_restart": True}),
           "pwl+split": ("pwl", {"discontinuity_restart": True,
                                 "enable_substep_state_correction": True})}


def reference_cases(patterns, configs):
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT)]
    import differential_runner as dr  # noqa: PLC0415
    import reference_status as rs  # noqa: PLC0415  (extra_kwargs)
    import v1_reader  # noqa: PLC0415
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(ROOT / rel)
        for cfg in configs:
            eng, kw = CONFIGS[cfg]

            def fn(case=case, eng=eng, kw=kw):
                with rs.extra_kwargs(kw):
                    return case.run(eng)
            yield f"{cfg} {rel}", fn


def static_cases():
    """The static path: a rectifier with and without a PWM'd MOSFET, a
    PWM source, the step options that path reads, cancellation, one step."""
    import pulsim as p  # noqa: PLC0415

    def rect(mosfet=True):
        b = p.CircuitBuilder()
        b.add_sine_voltage_source("Vs", "a", "gnd", 0.0, 100.0, 50.0)
        b.add_diode("D1", "a", "pp", 1e3, 1e-9)
        b.add_diode("D2", "gnd", "pp", 1e3, 1e-9)
        b.add_diode("D3", "nn", "a", 1e3, 1e-9)
        b.add_diode("D4", "nn", "gnd", 1e3, 1e-9)
        b.add_resistor("R1", "pp", "nn", 10.0)
        if not mosfet:
            return b, p.make_combined_switch_fn(b.graph.num_switches, [])
        b.add_mosfet("Q1", "pp", "q")
        b.add_resistor("R2", "q", "nn", 5.0)
        return b, p.make_pwm_switch_fn(10e3, 0.5, b.switch_index_of("Q1"), b.graph.num_switches)

    def pwm_src():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 10.0)
        b.add_switch("S", "in", "a", 1e3, 1e-9)
        b.add_resistor("R", "a", "gnd", 10.0)
        b.add_pwm_voltage_source("Vp", "x", "gnd", 5.0, 0.0, 1e5, 0.5)
        b.add_resistor("Rx", "x", "gnd", 1.0)
        return b, p.make_pwm_switch_fn(3e4, 0.3, b.switch_index_of("S"), b.graph.num_switches)

    def run(factory, t_end=4e-3, cancel_after=None, **opt):
        def fn():
            b, sw = factory()
            cache = p.PwlStateSpaceCache(b.graph, b.pool)
            cache.build()
            opts = p.SimulationOptions(t_start=0.0, t_end=t_end, dt=1e-6)
            for k, v in opt.items():
                setattr(opts, k, v)
            n = [0]

            def cont():
                n[0] += 1
                return cancel_after is None or n[0] < cancel_after
            return p.run_transient(cache, b.graph, b.pool, opts, switch_fn=sw,
                                   should_continue=cont)
        return fn

    yield "static rect_mosfet", run(rect)
    yield "static rect_plain", run(lambda: rect(False))
    yield "static rect_store3", run(rect, store_every=3)
    yield "static rect_restart_rule", run(rect, discontinuity_restart=True)
    yield "static rect_shadow", run(rect, mode_selection=1)
    yield "static rect_lcp", run(rect, mode_selection=2)
    yield "static rect_strict", run(rect, strict_event_iterations=True)
    yield "static pwm_src", run(pwm_src, t_end=1e-4)
    yield "static pwm_src_restart", run(pwm_src, t_end=1e-4, discontinuity_restart=True)
    yield "static rect_cancel", run(rect, cancel_after=50)
    yield "static one_step", run(rect, t_end=1e-6)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("dir", type=Path)
    ap.add_argument("--goldens", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--configs", default="pwl,pwl+restart,pwl+split")
    ap.add_argument("--static", action="store_true")
    a = ap.parse_args(argv)
    cases = []
    if a.goldens:
        cases += list(golden_cases())
    if a.reference:
        cases += list(reference_cases(a.circuits.split(","), a.configs.split(",")))
    if a.static:
        cases += list(static_cases())
    return record_or_compare(a.mode, a.dir / "pwl_identity.json", cases)


if __name__ == "__main__":
    sys.exit(main())
