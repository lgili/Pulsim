#!/usr/bin/env python3
"""The fixed-step engine's old loop against its twin on the step skeleton
(refactor-unified-transient-stepper, slice 3a): run one circuit on both
paths and compare every array bit for bit and every stats key.

    pwl_twin_identity.py --goldens                  # benchmarks/goldens/pwl cases
    pwl_twin_identity.py --reference [--circuits GLOB,...] [--configs pwl,pwl+restart]

The twin is selected by the internal environment variable
``PULSIM_PWL_STEPPER_TWIN=1`` (read once per run by ``run_transient``; not
a public option). Bit identity between two code paths is a LOCAL gate
(this machine, this build); the CI test compares at 1e-12 of scale.
Prints ``IDENTICAL <name>`` or ``DIFFER <name>: <first differing key>``;
exit status 1 when any case differs.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import os
import sys
import warnings
from pathlib import Path

import numpy as np

ENV = "PULSIM_PWL_STEPPER_TWIN"


@contextlib.contextmanager
def twin(on: bool, env: str = ENV):
    old = os.environ.get(env)
    os.environ[env] = "1" if on else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(env, None)
        else:
            os.environ[env] = old


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


def _arrays_differ(a, b, rtol):
    if a.dtype != b.dtype or a.shape != b.shape:
        return f"dtype/shape {a.dtype}{a.shape} != {b.dtype}{b.shape}"
    if rtol == 0 or a.dtype.kind not in "fc":
        same = np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
        return None if same else "differs"
    scale = float(np.nanmax(np.abs(a))) if a.size else 0.0
    ok = np.allclose(a, b, rtol=0, atol=rtol * max(scale, 1e-300), equal_nan=True)
    return None if ok else f"differs beyond {rtol:g} of scale {scale:.3g}"


def first_difference(a, b, path="", rtol=0.0) -> str | None:
    """The first key at which two fingerprints differ: bit for bit
    (``rtol=0``) or, for float arrays, within ``rtol`` of the array's
    scale (everything else stays exact)."""
    if isinstance(a, np.generic):
        a, b = np.asarray(a), np.asarray(b) if isinstance(b, np.generic) else b
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        d = _arrays_differ(a, b, rtol)
        return f"{path}: {d}" if d else None
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                return f"{path}.{k}: present on one path only"
            d = first_difference(a[k], b[k], f"{path}.{k}", rtol)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]", rtol)
            if d:
                return d
        return None
    return None if a == b else f"{path}: differs"


def run_both(fn, env: str = ENV):
    """(old-loop fingerprint, twin fingerprint); an exception is its
    type and message (a refusal must be the same refusal on both)."""
    fps = []
    for on in (False, True):
        with twin(on, env), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fps.append(fingerprint(fn()))
            except Exception as e:  # noqa: BLE001 - compared, not swallowed
                fps.append({"raised": f"{type(e).__name__}: {e}"})
    return fps[0], fps[1]


def check(name, fn, env: str = ENV) -> bool:
    a, b = run_both(fn, env)
    d = first_difference(a, b)
    tag = " (raised)" if "raised" in a else ""
    print(f"DIFFER {name}: {d}" if d else f"IDENTICAL {name}{tag}", flush=True)
    return d is None


ROOT = Path(__file__).resolve().parents[1]


def golden_cases():
    sys.path[:0] = [str(Path(__file__).resolve().parent)]
    import goldens  # noqa: PLC0415
    for name, fn, _names, _newton in goldens.cases():
        yield name, fn


def reference_cases(patterns, configs):
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT)]
    import differential_runner as dr  # noqa: PLC0415
    import reference_status as rs  # noqa: PLC0415
    import v1_reader  # noqa: PLC0415
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not any(fnmatch.fnmatch(rel, p) for p in patterns):
            continue
        case = v1_reader.load(ROOT / rel)
        for cfg in configs:
            eng, kw = rs.CONFIGS[cfg]

            def fn(case=case, eng=eng, kw=kw):
                with rs.extra_kwargs(kw):
                    return case.run(eng)
            yield f"{cfg} {rel}", fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--goldens", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--circuits", default="*")
    ap.add_argument("--configs", default="pwl,pwl+restart,pwl+split")
    a = ap.parse_args(argv)
    cases = []
    if a.goldens:
        cases += list(golden_cases())
    if a.reference:
        cases += list(reference_cases(a.circuits.split(","), a.configs.split(",")))
    bad = [n for n, fn in cases if not check(n, fn)]
    print(f"{len(cases) - len(bad)}/{len(cases)} identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
