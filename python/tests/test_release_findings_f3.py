"""Release findings, group F, third session: the exact engine's leftovers.

F-X-6  a trial mode's partial projection survived the settle under the output rule
       (forward converter, I(L_o) +0.585 A in 0.2 ps).
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]


def _case(rel):
    sys.path[:0] = [str(ROOT / "benchmarks" / "tools")]
    try:
        import v1_reader
    finally:
        del sys.path[0]
    return v1_reader.load(ROOT / "benchmarks" / rel)


def _no_dt(case, **kw):
    """The netlist's run without its dt (what the release route runs)."""
    b, sw, _ = case.build()
    args = dict(engine="exact", t_start=case.t_start, **kw)
    if sw is not None:
        args["switch_fn"] = sw
    if not case.uic:
        args["initial_state"] = case._dc_start(b, sw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, case.t_end, **args)


FWD = "circuits/switching/isolated/forward_converter.yaml"


@pytest.mark.parametrize("rt", [None, 1e-4, 1e-5, 1e-6])
def test_fx6_forward_converter_keeps_the_inductor_current_at_the_switch_off(rt):
    """mpmath truth (benchmarks/truth/forward_converter.json, exact_truth.py): I(L_o) is
    5.620787 A across the 48 us switch-off (D_fw takes it 5e-14 s later) and the full-run
    mean of V(out) is 6.6073535537897 V. Before the fix the output rule's run read 6.2063 A
    right after the event and its mean was 0.131 V off (1e-4 / 1e-5 / 1e-6 alike)."""
    truth = json.loads((ROOT / "benchmarks/truth/forward_converter.json").read_text(encoding="utf-8"))
    r = _no_dt(_case(FWD), **({} if rt is None else {"output_rtol": rt}))
    t = np.asarray(r.times)
    k = np.searchsorted(t, 4.80000553e-5)
    assert abs(float(np.asarray(r.i("L_o"))[k]) - 5.620787) < 1e-5
    mean = float(np.trapezoid(np.asarray(r.v("out")), t) / (t[-1] - t[0]))
    # the samples' linear interpolation: 1.9e-3 V on the rule-free grid, ~0.17·rtol·range
    # under the rule (1.1e-5 V at 1e-5); the defect moved it by 0.131 V
    tol = 1e-2 if rt is None else 100 * rt
    assert abs(mean - float(truth["windows"]["full"]["V(out)"]["mean"])) < tol
