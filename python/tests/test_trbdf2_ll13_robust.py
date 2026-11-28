"""F-Z-6: the ll13 active-clamp forward reference call (trbdf2, rtol 1e-8) must not
depend on the run length.

Root cause 1 (the t = 0 abort): a zero start without a stateful kind kept the init's
derivative slots (v_L = 0 for the 48 V the leakage inductor carries at t = 0+), the
first step's estimate was O(h) at every h, the step sat on h_min = span * 1e-11 and
the floor decided the run (forced accept at 5e-16 s for 50 us, Newton numerically
singular at 3e-17 s for 3 us).
"""
import dataclasses
import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "tools"))
import v1_reader  # noqa: E402

LL13 = ROOT / "benchmarks/circuits/switching/isolated/ll13_active_clamp_forward.yaml"


def _run(t_end, rtol, **kw):
    case = dataclasses.replace(v1_reader.load(LL13), t_end=t_end)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.run("trbdf2", rtol=rtol, atol=rtol * 1e-4, **kw)


@pytest.mark.parametrize("t_end,rtol", [(3e-6, 1e-8), (3e-6, 1e-10), (100e-6, 1e-10)])
def test_start_does_not_sit_on_the_floor(t_end, rtol):
    # these aborted at t = 0 ("Newton failed ... with h already at h_min")
    # before the first turn-off, with the floors of a run of t_end (h_max = t_end / 1000)
    r = _run(1.4e-6, rtol, dt=t_end / 1000)
    st = r._trbdf2_stats
    assert st["n_forced_accepts"] == 0
    assert st["n_derivative_refreshes"] >= 1   # the start took the probe's slots
    h_min = t_end * 1e-11   # the run's floor (span_ref = 1000 h_max)
    assert r.times[1] - r.times[0] > 10 * h_min   # it was the floor: forced, or Newton singular

