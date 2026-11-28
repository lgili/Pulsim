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
        # measured +1.5602e-6 A at dt and dt/2 (F-R-4); with the one-mode
        # projection +1.05e-5 / +9.0e-7 A; before the BE re-take -0.040 / -0.033 A
        assert abs(r.i("T1.s")[k] - FLYBACK_IS_8_1US) < 1e-8
        # measured -3.4e-5 / -1.8e-5 V; before +0.146 / +0.134 V (not converging)
        assert r.v("out")[-1] == pytest.approx(FLYBACK_VOUT_100US, abs=5e-4)
        # F-R-4: the edge's projection now takes the magnetizing mode too, so
        # stage 2 has no residual to flip (5 BE re-takes with the one-mode fit);
        # the re-take itself stays covered by test_restart_review_fixes
        assert r.counters["stage2_be_kept"] == 0


@pytest.mark.parametrize("kw", [RULE, SPLIT], ids=["restart", "split"])
def test_forward_converter_agrees_with_the_exact_truth_as_dt_halves(monkeypatch, kw):
    case = _case(monkeypatch, "forward_converter")
    errs = []
    for div in (1, 2):
        r = _run(case, div, kw)
        errs.append(abs(r.v("out")[-1] - FORWARD_VOUT_100US))
        # measured +3e-6 / +1e-6 A (F-R-4; +2.1e-5 / +1.4e-4 A with the one-mode
        # projection); before +0.160 / +0.149 A
        assert r.i("L_o")[-1] == pytest.approx(FORWARD_ILO_100US, abs=2e-3)
    # measured 2.2e-7 / 7.6e-7 V rule, 2.3e-6 / 8.5e-7 V split (F-R-4; 7.1e-6 /
    # 9.0e-5 V with the one-mode projection); before the
    # BE re-take 0.157 / 0.151 V (not converging)
    assert max(errs) < 1e-3, errs


@pytest.mark.parametrize("kw", [RULE, SPLIT], ids=["restart", "split"])
def test_forward_switch_off_projects_both_winding_modes_and_rectifier_conducts(monkeypatch, kw):
    """F-R-4: at the switch-off both windings see only g_off, so the fast subspace
    has two modes (leakage, τ ≈ 10 ps; magnetizing, τ = L(1 + k)·g_off ≈ 0.39 ns).
    The edge's projection fitted one (its growth test stopped at the leakage floor
    g_leak·(1 + v_max), 19.5 mA at stage 1's 6.5 kV), left 5 % of the magnetizing
    mode in x0+ and D_rect was held off for a whole step (first order in dt). On
    the OFF mode's slow manifold the primary carries g_off·V_in, the secondary ~0
    and D_rect is forward-biased (the truth re-conducts 6.1 ns after the edge)."""
    case = _case(monkeypatch, "forward_converter")
    case.t_end = 8.25e-6
    r = _run(case, 1, kw)
    t = np.asarray(r.times)
    k0 = int(np.searchsorted(t, 8e-6 - 1e-15))   # the edge's left sample
    i1, i2 = r.i("T1.p")[k0], r.i("T1.s")[k0]
    recs = [x for x in r.reinit_records if 7.99e-6 < x.t < 8.25e-6]
    assert len(recs) == 1 and recs[0].t == pytest.approx(8e-6, abs=1e-15)
    assert recs[0].kind == "cut_set" and sorted(recs[0].states) == ["T1.p", "T1.s"]
    # dW = ½·JᵀLJ, J = x0+ − x0 with x0+ = (g_off·V_in, 0): measured rel 1.1e-8;
    # with the one-mode fit about 1 % off
    lm, mm = 200e-6, 0.95 * 200e-6
    j1, j2 = 1e-6 * 24.0 - i1, -i2
    assert recs[0].dW == pytest.approx(0.5 * lm * (j1 * j1 + j2 * j2) + mm * j1 * j2, rel=1e-6)
    k = int(np.argmin(np.abs(t - 8.1e-6)))
    # (SW, D_rect, D_fw), the netlist's order: D_rect conducts from the edge's
    # step on (with the one-mode fit [0, 0, 1] until the next step)
    assert [bool(b) for b in r.switch_states[k]] == [False, True, True]
    # the secondary carries no trapped flux: dt/100 gives -6.71e-6 A (measured
    # -6.87e-6 A here; with the one-mode fit D_rect was off; trapped it was ~16 mA)
    assert abs(r.i("T1.s")[k] - -6.71e-6) < 1e-6
    assert r.counters["stage2_be_kept"] == 0


# half_bridge_transformer_dc (rule alone, dt 100 ns): I(T1.s) at the end of the steps
# in which a rectifier diode turns off; reference: pwl + restart at dt/100
HALF_BRIDGE_IS_EDGES = ((492.4e-6, 3.4911e-5), (631.3e-6, 3.6752e-5))


def test_half_bridge_turn_off_cut_keeps_its_located_crossing(monkeypatch):
    """A rectifier turn-off inside a restart step is located and the step cut there;
    the first piece ends AT the crossing (v_D = 0), so its stage 2 may iterate on
    that round-off: a flip, or a flip and back. The cycle was left as solved, read
    as an unsettled piece, and the uncut step stood: the secondary kept 15 % / 35 %
    of its off-state settling at these two edges (+3.3e-6 V on V(out) each, first
    order in dt), which edge depended on a 1e-9 V difference in the previous
    switch edge's projection. Kept, the step's error is its first-order part:
    +1.2e-7 / +1.9e-7 A (0.3 % / 0.5 %); the bound 2 % lies between."""
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks" / "tools"))
    pytest.importorskip("yaml")
    import v1_reader

    case = v1_reader.load(ROOT / "benchmarks" / "circuits" / "magnetics" / "half_bridge_transformer_dc.yaml")
    case.t_end = 0.64e-3
    r = _run(case, 1, RULE)
    t = np.asarray(r.times)
    i_s = np.asarray(r.i("T1.s"))
    for te, ref in HALF_BRIDGE_IS_EDGES:
        k = int(np.searchsorted(t, te + 1e-12)) - 1   # the step end's last sample
        assert t[k] == pytest.approx(te, abs=1e-12)
        assert abs(i_s[k] - ref) < 0.02 * ref, (te, i_s[k])
