"""Release finding F-B2-1: the exact engine's window integrals are chords.

On the buck of ``test_recorder_device_readings._buck`` the exact engine's
STATES at the window ends agree with an independent mpmath truth to 6e-15
(relative), but its recorder window integrals over [A, B] are 1.1e-4 to
2.0e-4 off: the recorder integrates each accepted step as the straight line
between its ends, and the exact engine's steps are its 10 us output steps
(101 of them on 1 ms). ``output_rtol`` is dropped without a word when
recorders are given, and ``res.integral`` / ``res.avg`` (documented
"Correct on any grid") read the same chords from the stored record. The
error is the chord's: it falls as h^2 with ``dt`` (1.1e-8 at dt = 0.1 us,
1.1e-10 at 10 ns). trbdf2 (rtol 1e-11) and the fixed engine with the
restart rule at dt/16 are within 6e-9 of the truth: they are right.

Truth (no pulsim): ``.scratch/exact_buck/truth.py`` in the worktree, two
ways that agree to 18 digits on the integrals: the repo's mpmath tool
``benchmarks/tools/exact_truth.py`` (fa4b5c71) and a hand derivation
(x = [iL, vC] per mode, x(t) = e^{Mt}(x0 + xp) - xp, the diode's turn-off
by bisection). Notes: openspec/changes/update-number-changing-release/
fixes/exact-buck-integrals.md.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

import pulsim as p

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_recorder_device_readings as m  # noqa: E402

# mpmath, 40+ digits (hand derivation == exact_truth.py)
TRUTH_STATES = {m.A: (1.8962389778368713475, 7.8493711689425254904),
                m.B: (3.4379292124049610102, 9.5072665802156611277)}
TRUTH_INT = {"i(L)": 0.0012814829296360187342,
             "v(out)": 0.0063245198776164365313,
             "p(R)": 0.012186147898259251171}


def _run(t_end, **kw):
    b, fn, _, _, _ = m._buck()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, switch_fn=fn, engine="dsed", **kw)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "finding F-B2-1: engine='exact' recorder windows integrate the chord of "
    "each 10 us step (1.1e-4..2.0e-4 off on the buck while its states are "
    "exact to 1e-14); output_rtol is dropped silently with recorders"))
def test_exact_engine_window_integrals_match_truth():
    # control: the states are exact. A RuntimeError (not the xfail's
    # AssertionError) so a regression here FAILS instead of being absorbed.
    for t, (il, vc) in TRUTH_STATES.items():
        r = _run(t)
        e = max(abs(r.i("L")[-1] / il - 1), abs(r.v("out")[-1] / vc - 1))
        if not (abs(r.times[-1] - t) < 1e-15 and e < 1e-12):
            raise RuntimeError(f"exact-engine state at {t} off by {e:.2e}")
    r = _run(1e-3, output_rtol=1e-10, recorders=[
        p.Recorder("w", list(TRUTH_INT), keep="none", windows=[(m.A, m.B)])])
    errs = {q: abs(r.recorded["w"].windows[q]["integral"][0] / v - 1)
            for q, v in TRUTH_INT.items()}
    assert max(errs.values()) < 1e-6, errs
