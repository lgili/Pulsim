"""refactor-unified-transient-stepper slice 3b: the variable-step engine's
legacy loop (selected by the internal ``PULSIM_TRBDF2_STEPPER_LEGACY=1``)
and its twin on the step skeleton (the default since slice 3c) give the same
run.

Between two code paths CI compares at 1e-12 of each float array's scale
(every count, mask and string exactly); bit identity is the LOCAL gate,
``benchmarks/tools/trbdf2_twin_identity.py --goldens --reference --extra``
on this machine (DECISIONS 15).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools"), str(ROOT / "benchmarks")]
import pwl_twin_identity as pt  # noqa: E402
import reference_status as rs  # noqa: E402
import trbdf2_twin_identity as tw  # noqa: E402
import v1_reader  # noqa: E402

RTOL = 1e-12
# Small reference circuits: diodes, a gated switch, a Newton device,
# magnetics (each runs in well under a second on trbdf2).
CIRCUITS = [
    "circuits/switching/buck/buck_switching.yaml",
    "circuits/switching/buck/buck_mosfet_nonlinear.yaml",
    "circuits/diodes/diode_bridge_rectifier.yaml",
    "circuits/magnetics/half_bridge_transformer_dc.yaml",
]


def _same(fn):
    a, b = pt.run_both(fn, tw.ENV)
    assert "raised" not in a, a.get("raised")
    d = pt.first_difference(a, b, rtol=RTOL)
    assert d is None, d


@pytest.mark.parametrize("mode", ["", "lcp"])
@pytest.mark.parametrize("rel", CIRCUITS)
def test_twin_runs_the_reference_circuit_as_the_legacy_loop(rel, mode):
    path = ROOT / "benchmarks" / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    if mode and "nonlinear" in rel:
        pytest.skip("mode_selection is refused with a Newton device (by name)")
    case = v1_reader.load(path)
    eng, kw = rs.CONFIGS["trbdf2"]
    kw = dict(kw, mode_selection=mode) if mode else kw

    def run():
        with rs.extra_kwargs(kw):
            return case.run(eng)
    _same(run)


def test_twin_runs_ticks_and_a_monitor_as_the_legacy_loop():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _same(tw._ticks_and_monitor)


@pytest.mark.parametrize("split", [1.02e-3 + 1e-9, 1.0333e-3],
                         ids=["inside_a_restart", "between_edges"])
def test_a_run_stopped_on_one_path_resumes_on_the_other(split):
    """old→old, old→twin, twin→old, twin→twin: one resumed run."""
    ends = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ft in (False, True):
            for st in (False, True):
                second, first = tw._resume(split, 2e-3, ft, st)
                ends[(ft, st)] = (pt.fingerprint(first), pt.fingerprint(second))
    base = ends[(False, False)]
    for key, (f1, f2) in ends.items():
        d = (pt.first_difference(base[0], f1, rtol=RTOL)
             or pt.first_difference(base[1], f2, rtol=RTOL))
        assert d is None, f"{key}: {d}"
