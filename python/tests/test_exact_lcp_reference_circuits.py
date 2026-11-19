"""dsed with mode_selection='lcp' on the five reference circuits that the ideal-switch
flip evidence (benchmarks/IDEAL_SWITCH_FLIP_EVIDENCE.md on feat/core-a-ideal-switch,
d9159fe4) listed as DiodeModeInconsistentError. That branch predates fd060121, which
holds the session-6 guard and tie fixes (add-complementarity-mode-selection
DECISIONS #34). On this tip each run agrees with the default selection, which
agrees with the harness reference (benchmarks/REFERENCE_STATUS.md). Measured
relative L2 against the default: 0 to 4.5e-5."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
sys.path[:0] = [str(BENCH), str(BENCH / "tools")]

import differential_runner as dr  # noqa: E402
import v1_reader  # noqa: E402


@pytest.mark.parametrize("rel", [
    "magnetics/half_bridge_transformer_dc.yaml",
    "switching/boost/boost_pfc_open_loop.yaml",
    "switching/isolated/active_clamp_forward.yaml",
    "switching/isolated/flyback_converter.yaml",
    "switching/isolated/forward_converter.yaml",
])
def test_lcp_selection_runs_and_agrees_with_the_default(rel):
    case = v1_reader.load(dr.ROOT / "circuits" / rel)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = case.run("dsed")
        b = case.run("dsed", mode_selection="lcp")   # raised DiodeModeInconsistentError there
    ta, xa = np.asarray(a.times), np.asarray(a.states)
    tb, xb = np.asarray(b.times), np.asarray(b.states)
    tg = np.union1d(ta, tb)
    for j in range(xa.shape[1]):
        d = np.interp(tg, ta, xa[:, j]) - np.interp(tg, tb, xb[:, j])
        rms = np.sqrt(np.trapezoid(d * d, tg) / (tg[-1] - tg[0]))
        assert rms <= 1e-3 * max(np.abs(xa[:, j]).max(), 1e-9), j   # the harness's 1e-3 tolerance
