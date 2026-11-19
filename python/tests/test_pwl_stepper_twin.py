"""refactor-unified-transient-stepper slice 3a: the fixed-step engine's
legacy loop and its twin on the step skeleton (selected by the internal
``PULSIM_PWL_STEPPER_TWIN=1``) give the same run.

Between two code paths CI compares at 1e-12 of each float array's scale
(every count, mask and string exactly); bit identity is the LOCAL gate,
``benchmarks/tools/pwl_twin_identity.py --goldens --reference`` on this
machine (DECISIONS 13).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import pulsim as p

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "tools"), str(ROOT / "benchmarks")]
import pwl_twin_identity as tw  # noqa: E402
import reference_status as rs  # noqa: E402
import v1_reader  # noqa: E402

RTOL = 1e-12
# Small reference circuits covering diodes, gated switches, a Newton
# device, magnetics and a three-phase bridge (each runs in well under 1 s).
CIRCUITS = [
    "circuits/switching/buck/buck_switching.yaml",
    "circuits/switching/buck/buck_mosfet_nonlinear.yaml",
    "circuits/diodes/diode_bridge_rectifier.yaml",
    "circuits/magnetics/half_bridge_transformer_dc.yaml",
]


def _same(fn):
    a, b = tw.run_both(fn)
    assert "raised" not in a, a.get("raised")
    d = tw.first_difference(a, b, rtol=RTOL)
    assert d is None, d


@pytest.mark.parametrize("cfg", list(c for c in rs.CONFIGS if c.startswith("pwl")))
@pytest.mark.parametrize("rel", CIRCUITS)
def test_twin_runs_the_reference_circuit_as_the_legacy_loop(rel, cfg):
    path = ROOT / "benchmarks" / rel
    if not path.exists():
        pytest.skip(f"{rel} not in this checkout")
    case = v1_reader.load(path)
    eng, kw = rs.CONFIGS[cfg]

    def run():
        with rs.extra_kwargs(kw):
            return case.run(eng)
    _same(run)


def _rect():
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 50.0, 0.0)
    b.add_diode("D", "ac", "out", 1e3, 1e-9, 0.7)
    b.add_capacitor("Co", "out", "gnd", 100e-6)
    b.add_resistor("Rl", "out", "gnd", 100.0)
    return b


def _sim(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(_rect(), dt=1e-6, engine="pwl", **kw)


@pytest.mark.parametrize("restart", [False, True])
def test_a_run_stopped_on_one_path_resumes_on_the_other(restart):
    """old→old, old→twin, twin→old, twin→twin: one resumed run."""
    kw = dict(discontinuity_restart=True) if restart else {}
    ends = {}
    for first_twin in (False, True):
        with tw.twin(first_twin):
            first = _sim(t_end=12e-3, **kw)
        for second_twin in (False, True):
            with tw.twin(second_twin):
                second = _sim(t_end=25e-3, resume_from=first.final_snapshot, **kw)
            ends[(first_twin, second_twin)] = tw.fingerprint(second)
    base = ends[(False, False)]
    for key, fp in ends.items():
        d = tw.first_difference(base, fp, rtol=RTOL)
        assert d is None, f"{key}: {d}"

