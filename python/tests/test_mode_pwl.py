"""add-complementarity-mode-selection on the fixed engine (tasks 4.2, 4b.3, 6.3, 6.4).

'shadow' poses the LCP of each decision site's diode k-port beside the settle (numbers unchanged);
'lcp' lets it decide (the settle never runs: its rounds and mask cycles stay 0); the default guard
re-decides a settle that ends in a mode violating a diode's law. References: trbdf2 at rtol 1e-10
(the dsed lcp test list, test_mode_lcp.py) and the default run at the same dt where the settle is
right (DECISIONS F1-F3). The bridges compare p - n: their DC side floats.
"""
import warnings

import numpy as np
import pytest

import pulsim as p
from tests.oracles import lcp_pulsim_bridge as B

T = 1e-5
GON, GOFF = 1e3, 1e-9
SPLIT = {"discontinuity_restart": True, "enable_substep_state_correction": True}
# "pwl" = the restart-less fixed step, explicit since the release made the restart the default
CONFIGS = [{"discontinuity_restart": False}, {"discontinuity_restart": True}, SPLIT]
# Legacy subject: "the default run" these tests compare with is the settle (and its guard);
# the release default mode_selection='auto' picks 'lcp' for these circuits.
SETTLE = {"mode_selection": "legacy"}


# Measured under the event split (probes/pwl_shadow_survey.out, pwl+split, err_lcp against
# trbdf2 rtol 1e-10): DCM buck +9.06e-7, L family +8.2e-7 ... +1.02e-6, boost 50 / 500 Ohm
# -2.29e-6 / -7.32e-6, bridges V_f = 0 / 0.7 / three-phase +4.60e-7 / +4.33e-7 / +4.34e-10 V.
# Bounds: the measured value with headroom, not a rounding.
DCM_TOL, LD_TOL, B50_TOL, B500_TOL = 2e-6, 2e-6, 5e-6, 1.5e-5
BR0_TOL, BR07_TOL, BR3_TOL = 1e-6, 1e-6, 1e-9


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture
def conductance(monkeypatch):
    """Legacy subject: the default guard exists on the conductance form only (run_transient
    turns it off for a branch-form diode); add_switch / add_diode without form= (the oracle's
    builder too) build that form, as before the release."""
    for name in ("add_switch", "add_diode"):
        f = getattr(p.CircuitBuilder, name)
        monkeypatch.setattr(p.CircuitBuilder, name,
                            lambda self, *a, _f=f, **k: _f(self, *a, **{"form": "conductance", **k}))


def _buck(L=100e-6, vth=0.0, phase=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_switch("S", "vin", "sw", GON, GOFF)
    b.add_diode("D1", "gnd", "sw", GON, GOFF, vth)
    b.add_inductor("L", "sw", "vout", L)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("R", "vout", "gnd", 50.0)
    return b, p.make_pwm_switch_fn(1 / T, 0.3, b.switch_index_of("S"), b.graph.num_switches, phase)


def _boost(R, vth=0.0):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", GON, GOFF)
    b.add_diode("D1", "sw", "out", GON, GOFF, vth)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("R", "out", "gnd", R)
    return b, p.make_pwm_switch_fn(1 / T, 0.5, b.switch_index_of("S"), b.graph.num_switches, 0.0)


def _sim(b, t_end, dt, fn=None, guard=True, **kw):
    if fn is not None:
        kw["switch_fn"] = fn
    p._PWL_MODE_GUARD = guard
    try:
        return p.simulate(b, t_end=t_end, dt=dt, engine="pwl", **kw)
    finally:
        p._PWL_MODE_GUARD = True


def _end(r, node):
    if "-" in node:
        a, c = node.split("-")
        return float(np.asarray(r.v(a))[-1] - np.asarray(r.v(c))[-1])
    return float(np.asarray(r.v(node))[-1])


def _same(a, b):
    return (np.array_equal(np.asarray(a.times), np.asarray(b.times))
            and np.array_equal(np.asarray(a.states), np.asarray(b.states)))


@pytest.mark.parametrize("kw", CONFIGS, ids=["pwl", "restart", "split"])
def test_shadow_is_bit_identical_and_poses_an_lcp_at_every_site(kw):
    b, fn = _buck(phase=0.37 * T / 200)          # gate edges off the grid: the split runs
    r0 = _sim(b, 40 * T, T / 200, fn, guard=False, **SETTLE, **kw)
    r1 = _sim(b, 40 * T, T / 200, fn, guard=False, mode_selection="shadow", **kw)
    assert _same(r0, r1)
    ms = r1.diagnostics["mode_shadow"]
    sites = ms["sites"]
    assert ms["events"] == sum(sites.values()) > 0
    if kw["discontinuity_restart"]:
        assert sites["restart"] > 0                 # the restart stages are sites too
    if kw is SPLIT:
        assert sites["substep"] > 0
    assert ms["errors"] == 0
    assert ms["disagreements"] == []                 # DCM buck: the settle and the LCP agree


def _perturbed(fam, k):
    """The family's circuit with C and R moved by k units in the last place."""
    f = B.FAMILIES[fam]
    parts = []
    for kind, *a in f.parts:
        if kind in ("C", "R") and k:
            a = list(a)
            a[3] = float(np.nextafter(a[3], np.inf) if k > 0 else np.nextafter(a[3], -np.inf))
            for _ in range(abs(k) - 1):
                a[3] = float(np.nextafter(a[3], np.inf if k > 0 else -np.inf))
        parts.append((kind, *a))
    # the release default form (the bridge builds the conductance form unless told)
    return B.Family(fam, parts, f.run).builder(0.0, form="branch")


@pytest.mark.parametrize("mode", [None, "lcp"], ids=["default", "lcp"])
@pytest.mark.parametrize("fam", ["bridge1_vf0", "bridge1_vf07", "bridge3_cfilter"])
def test_rounding_fuzz_gives_one_mode_sequence(fam, mode):
    """20 last-bit perturbations of the circuit (C and R +-1..10 ulp) give one waveform and one
    diode mode sequence, up to ties: on the single-phase bridges a few samples (<= 6 of 4001, from
    t = 5.11 ms) label a diode differently while every diode carries the leak current only
    (< 1e-7 A measured: g_off * 100 V) -- the LCP and the settle both see a tie there (the survey's
    'both' disagreements). The waveform stays within rounding (2.7e-11 V measured)."""
    kw = {"mode_selection": mode} if mode else {}
    names = [d[1] for d in B.FAMILIES[fam].parts if d[0] == "D"]

    def run(k):
        r = _sim(_perturbed(fam, k), 0.04, 1e-5, **kw)
        return (np.asarray(r.switch_states).astype(np.int8),
                np.asarray(r.v("p")) - np.asarray(r.v("n")),
                np.max(np.abs([np.asarray(r.i(n)) for n in names]), axis=0))
    s0, w0, i0 = run(0)
    for k in [s * m for m in range(1, 11) for s in (1, -1)]:
        s, w, i = run(k)
        assert np.max(np.abs(w - w0)) <= 1e-9
        rows = np.where((s != s0).any(axis=1))[0]
        if fam == "bridge3_cfilter":
            assert rows.size == 0
        assert np.all(np.maximum(i[rows], i0[rows]) < 1e-6)


# ---- the dsed 'lcp' test list (task 6.2) on pwl; 6.4 hysteresis -----------------------------
def test_lcp_buck_at_rest_stays_at_zero():
    b, _ = _buck()
    r = _sim(b, 1e-6, 1e-8, lambda t: p.SwitchStateMask(b.graph.num_switches), mode_selection="lcp")
    assert abs(_end(r, "sw")) < 1e-9
    ms = r.diagnostics["mode_selection"]
    assert ms["lcp_count"] >= 1 and ms["settle_rounds_total"] == 0 and ms["mask_cycles"] == 0


def test_lcp_banded_dcm_buck_equals_the_default():
    """6.4: V_th = 1e-2 (a hysteresis band) — the LCP decides exactly as the settle."""
    b, fn = _buck(vth=1e-2)
    for kw in CONFIGS:
        r0 = _sim(b, 400 * T, T / 200, fn, **SETTLE, **kw)
        r1 = _sim(b, 400 * T, T / 200, fn, mode_selection="lcp", **kw)
        assert np.max(np.abs(np.asarray(r0.states) - np.asarray(r1.states))) <= 1e-9
        assert r1.diagnostics["mode_selection"]["settle_rounds_total"] == 0


# The dsed 'lcp' list (task 6.2) on pwl with the event split, at the converters' T/200 and the
# bridges' 10 us, against trbdf2 rtol 1e-10. On the converters 'lcp' is also bit-identical to the
# default run (the settle is right there: the survey found no disagreement).
def _l_buck(delta, sign):
    return lambda: _buck(L=100e-6 * (1 + sign * delta))


LCP_LIST = [
    ("dcm_buck", lambda: _buck(), 400 * T, T / 200, "vout", 9.100213085463578, DCM_TOL),
    ("boost_50", lambda: _boost(50.0), 400 * T, T / 200, "out", 23.80641899306707, B50_TOL),
    ("boost_500", lambda: _boost(500.0), 400 * T, T / 200, "out", 44.72166182694367, B500_TOL),
] + [(f"buck_L{'+' if s > 0 else '-'}{d:g}", _l_buck(d, s), 400 * T, T / 200, "vout", ref, LD_TOL)
     for (d, s), ref in {(1e-5, 1): 9.1001802163689, (1e-5, -1): 9.100245922918166,
                         (1e-4, 1): 9.099884756731521, (1e-4, -1): 9.10054142420285,
                         (1e-2, 1): 9.067580363866686, (1e-2, -1): 9.133251054852796}.items()]


@pytest.mark.parametrize("name,make,t_end,dt,node,ref,tol", LCP_LIST, ids=[c[0] for c in LCP_LIST])
def test_lcp_list_on_pwl(name, make, t_end, dt, node, ref, tol):
    b, fn = make()
    r0 = _sim(b, t_end, dt, fn, **SETTLE, **SPLIT)
    r1 = _sim(b, t_end, dt, fn, mode_selection="lcp", **SPLIT)
    ms = r1.diagnostics["mode_selection"]
    assert ms["settle_rounds_total"] == 0 and ms["mask_cycles"] == 0 and ms["errors"] == 0
    assert ms["inconsistent"] == 0
    assert _same(r0, r1)
    assert abs(_end(r1, node) - ref) <= tol


BRIDGES = [("bridge1_vf0", "p-n", 95.17111663119155, BR0_TOL),
           ("bridge1_vf07", "p-n", 93.83805537707278, BR07_TOL),
           ("bridge3_cfilter", "p-n", 171.7878336075392, BR3_TOL)]


@pytest.mark.parametrize("fam,node,ref,tol", BRIDGES, ids=[c[0] for c in BRIDGES])
def test_lcp_bridges_on_pwl(fam, node, ref, tol):
    b = B.FAMILIES[fam].builder(0.0)
    r1 = _sim(b, 0.2, 1e-5, mode_selection="lcp", **SPLIT)
    ms = r1.diagnostics["mode_selection"]
    assert ms["settle_rounds_total"] == 0 and ms["mask_cycles"] == 0 and ms["errors"] == 0
    assert abs(_end(r1, node) - ref) <= tol


# ---- 4b.3: the default guard changes nothing where the settle is right -----------------------
@pytest.mark.parametrize("kw", CONFIGS, ids=["pwl", "restart", "split"])
def test_guard_leaves_consistent_runs_bit_identical(kw, conductance):
    kw = {**SETTLE, **kw}
    b, fn = _buck()
    assert _same(_sim(b, 400 * T, T / 200, fn, guard=False, **kw),
                 _sim(b, 400 * T, T / 200, fn, **kw))
    b3 = B.FAMILIES["bridge3_cfilter"].builder(0.0)
    on = _sim(b3, 0.06, 1e-5, **kw)
    assert _same(_sim(B.FAMILIES["bridge3_cfilter"].builder(0.0), 0.06, 1e-5, guard=False, **kw), on)
    assert on.diagnostics["mode_guard"]["repairs"] == 0
    assert on.diagnostics["mode_guard"]["checks"] > 0          # the guard was evaluated


def test_guard_repairs_a_settle_left_in_a_wrong_mode(conductance):
    """The floating single-phase bridge (V_f = 0) ends two settles without a fixed point in a mode
    that violates D1's law (9.9e-7 V); the guard re-decides them. The waveform moves by 4.5e-13 V
    (the repaired steps carry ~0 current) and the end value is unchanged."""
    b = B.FAMILIES["bridge1_vf0"].builder(0.0)
    off = _sim(b, 0.2, 1e-5, guard=False, **SETTLE)
    on = _sim(B.FAMILIES["bridge1_vf0"].builder(0.0), 0.2, 1e-5, **SETTLE)
    g = on.diagnostics["mode_guard"]
    assert g["repairs"] >= 1 and len(list(off.event_iteration_breaches)) >= g["repairs"]
    assert all(v > 0 for v in g["repair_violations"])
    pn = lambda r: np.asarray(r.v("p")) - np.asarray(r.v("n"))  # noqa: E731
    assert np.max(np.abs(pn(on) - pn(off))) <= 1e-9


@pytest.mark.parametrize("kw", [{"discontinuity_restart": True}, SPLIT], ids=["restart", "split"])
def test_shadow_on_the_floating_bridge_compares_only_solved_settles(kw):
    """A restart stage entered with the previous stage's mask cycle never solves; the shadow skips
    it instead of judging an empty mode (review finding: it read past the mode vector)."""
    b = B.FAMILIES["bridge1_vf0"].builder(0.0)
    r0 = _sim(b, 0.2, 1e-5, guard=False, **SETTLE, **kw)
    r1 = _sim(B.FAMILIES["bridge1_vf0"].builder(0.0), 0.2, 1e-5, guard=False,
              mode_selection="shadow", **kw)
    assert _same(r0, r1)
    dis = r1.diagnostics["mode_shadow"]["disagreements"]
    assert dis and all(len(d["settle_mode"]) == 4 for d in dis)


def test_lcp_with_frozen_modes_is_refused():
    b, fn = _buck()
    with pytest.raises(p.UnsupportedError) as e:
        _sim(b, 4 * T, T / 200, fn, mode_selection="lcp", max_event_iterations=0)
    assert e.value.engine == "pwl" and e.value.feature == "mode_selection"
