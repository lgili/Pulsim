"""mode_selection='shadow' on trbdf2 is bit-identical to the default
(add-complementarity-mode-selection 4.2; DECISIONS #36 of add-branch-form-ideal-switch).

    PYTHONPATH=<worktree>/python:<worktree>/benchmarks python3 trbdf2_shadow_survey.py [--circuits GLOB]

Every reference circuit of the differential matrix with a PWL diode, in both
forms (conductance; branch, R_on = 1/g_on), on trbdf2: the default run and
the 'shadow' run, times / states / switch states compared byte for byte.
Printed per cell: sites, disagreements, same.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "benchmarks"
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import differential_runner as dr  # noqa: E402
import ideal_switch_flip_evidence as ev  # noqa: E402
import v1_reader  # noqa: E402

import pulsim  # noqa: E402


def cells(pattern):
    for rel, entry in sorted(dr.load_triage().items()):
        if entry["status"] != "runs" or not fnmatch.fnmatch(rel, pattern):
            continue
        case = v1_reader.load(ROOT / rel)
        if any(c.get("type") == "diode" and "is" not in c for c in case._components):
            yield rel, case


def run(case, kw):
    orig = pulsim.simulate
    pulsim.simulate = lambda *a, **k: orig(*a, **{**k, **kw})
    try:
        return case.run("trbdf2")
    except Exception as ex:  # noqa: BLE001 - reported per cell
        return f"{type(ex).__name__}: {ex}"[:120]
    finally:
        pulsim.simulate = orig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="circuits/*")
    a = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    print(f"pulsim from {pulsim.__file__}")
    n = ndiff = 0
    for rel, case in cells(a.circuits):
        for form, ctx in (("conductance", contextlib.nullcontext), ("branch", ev.branch_form)):
            with ctx():
                r0 = run(case, {})
                r1 = run(case, {"mode_selection": "shadow"})
            name = f"{rel.removeprefix('circuits/')} [{form}]"
            if isinstance(r1, str) and "mode_selection" in r1 and not isinstance(r0, str):
                print(f"{name:64s} refused by name: {r1[:90]}")   # a Newton device
                continue
            if isinstance(r0, str) or isinstance(r1, str):
                same = isinstance(r0, str) and isinstance(r1, str) and r0 == r1
                print(f"{name:64s} error: {r0 if isinstance(r0, str) else r1} same={same}")
                n += 1
                ndiff += not same
                continue
            same = all(np.asarray(getattr(r0, k)).tobytes() == np.asarray(getattr(r1, k)).tobytes()
                       for k in ("times", "states", "switch_states"))
            ms = r1.diagnostics["mode_shadow"]
            print(f"{name:64s} sites {sum(ms['sites'].values()):6d} lcps {ms['lcps']:6d} "
                  f"disagree {len(ms['disagreements']):4d} errors {ms['errors']:3d} "
                  f"same={'yes' if same else 'NO'}", flush=True)
            n += 1
            ndiff += not same
    print(f"{n} runs, {'ALL bit-identical' if ndiff == 0 else f'{ndiff} differ'}")
    return 0 if ndiff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
