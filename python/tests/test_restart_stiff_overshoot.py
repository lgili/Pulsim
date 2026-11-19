"""The restart's stage-2 fallback also catches a stage-2 turn-on that undoes stage 1.

At a transformer converter's switch-off (flyback, forward; coupling 0.95, g_off = 1e-6)
the magnetising flux has nowhere to go: the exact solution drains it through g_off in
~1 ns and the rectifier turns off 58 ps (flyback) / 32 ps (forward) after the gate. The
restart's stage 1 (backward Euler) does the same; its SDIRK stage 2, from the pseudo
previous state x0* = x0 + ((1-γ)/γ)(x1 - x0), has R(z) < 0 for this stiff mode, flips
the sign of the residual flux and forward-biases the rectifier again. Kept, the
secondary trapped 0.04 A (flyback, dt), 0.03 A (dt/4), 0.12 A (dt/16) at every
switch-off, a fixed fraction of the period's energy: the `exact` error of V(out) was
0.11 at dt, 0.10 at dt/4, 0.067 at dt/16 (forward 0.092 / 0.085). Now such a stage 2
is re-tried as backward Euler like a stage-2 turn-off: 2.6e-6 (flyback), 5e-6 (forward).

Truth: the 50-digit exact engine (benchmarks/tools/exact_truth.py on the
exact-truth branch), conductance form, sampled with mpmath.
"""
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CIRCUITS = ROOT / "benchmarks" / "circuits" / "switching" / "isolated"
RULE = dict(discontinuity_restart=True)
SPLIT = dict(discontinuity_restart=True, enable_substep_state_correction=True)

# exact truth at t = 100 us (V(out) is a capacitor voltage: continuous at the gate edge there)
FLYBACK_VOUT_100US = 6.13112757797267
FORWARD_VOUT_100US = 4.11310611813393
FORWARD_ILO_100US = 5.13702824155989
# flyback, 100 ns after the first switch-off: the rectifier is off, the secondary
# carries its g_off leakage only (exact: +1.56e-6 A; before the fix -0.0401 A)
FLYBACK_IS_8_1US = 1.56023291009852e-6


def _case(monkeypatch, name):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    case = v1_reader.load(CIRCUITS / f"{name}.yaml")
    case.t_end = 1e-4
    return case


def _run(case, div, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # the g_off spike warning at the switch-off
        return case.run("pwl", dt=case.dt / div, **kw)


@pytest.mark.parametrize("kw", [RULE, SPLIT], ids=["restart", "split"])
def test_flyback_releases_its_secondary_at_every_switch_off(monkeypatch, kw):
    case = _case(monkeypatch, "flyback_converter")
    for div in (1, 2):
        r = _run(case, div, kw)
        t = np.asarray(r.times)
        k = int(np.argmin(np.abs(t - 8.1e-6)))
        # measured +1.05e-5 A (dt) / +9.0e-7 A (dt/2); before -0.040 / -0.033 A
        assert abs(r.i("T1.s")[k] - FLYBACK_IS_8_1US) < 2e-4
        # measured -3.4e-5 / -1.8e-5 V; before +0.146 / +0.134 V (not converging)
        assert r.v("out")[-1] == pytest.approx(FLYBACK_VOUT_100US, abs=5e-4)
        assert r.counters["stage2_be_kept"] >= 3   # 5 measured: one per switch-off; before 0


@pytest.mark.parametrize("kw", [RULE, SPLIT], ids=["restart", "split"])
def test_forward_converter_agrees_with_the_exact_truth_as_dt_halves(monkeypatch, kw):
    case = _case(monkeypatch, "forward_converter")
    errs = []
    for div in (1, 2):
        r = _run(case, div, kw)
        errs.append(abs(r.v("out")[-1] - FORWARD_VOUT_100US))
        # measured +2.1e-5 / +1.4e-4 A; before +0.160 / +0.149 A
        assert r.i("L_o")[-1] == pytest.approx(FORWARD_ILO_100US, abs=2e-3)
    # measured 7.1e-6 / 9.0e-5 V; before 0.157 / 0.151 V (not converging)
    assert max(errs) < 1e-3, errs
