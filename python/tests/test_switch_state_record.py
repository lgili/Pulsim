"""The kernel's per-sample record of the STAMPED switch / diode mask.

A PWL ``SwitchedDiode`` is ``g_on`` or ``g_off`` with no voltage offset;
``V_th`` only gates OFF→ON. A conducting diode therefore sits at
``i / g_on`` — inside ``0 < v_D < V_th``, where both states are
self-consistent — and the stored voltage cannot say which stamp produced
a sample. The reporting used to infer it (KCL at the diode's nodes plus
commutation records plus hysteresis fill), and every circuit the
inference could not see brought the zero-loss defect back.

The kernel now records, with every stored sample, the mask of the solve
that produced it (``SimulationResult.switch_states``), and the reporting
reads ``i_D[k] = g(mask_k) · v_D[k]``. These tests take each reproducer
from the adversarial review of the inference and check the reported
diode current against charge the circuit can prove WITHOUT the record —
capacitor ``C·Δv``, resistor ``∫v/R``, a series inductor's or source's
state current — on ``engine='pwl'``, ``engine='auto'`` and, where the
DSED engine runs the circuit, ``engine='dsed'``.

Measured before the record (this branch at 2e4e1b17, ``engine='pwl'``,
``V_th = 0.7``): the snubbed boost reported 0.00297 C of 0.01063 C
(duty 0.078 of 0.327), the transformer-secondary and VCVS-probe
rectifiers reported −1.4e-9 C and −7.1e-9 C of 0.080 C and 0.399 C
(duty 0.0000, two warnings each), the RL half-wave counted 48
recoveries (E_sw 6.93e-4 J; 4 and 1.72e-4 J at V_th = 0) and its
``engine='auto'`` run 8 (f_sw 120 Hz for a 60 Hz rectifier), and the
line-inductance bridge on ``engine='auto'`` reported D3 at 700.5 A
while the line inductor carried 294.1 A.
"""

from __future__ import annotations

import functools
import pickle
import warnings

import numpy as np
import pytest

import pulsim as p

ENGINES = ("pwl", "auto")
_RULE = "commutation rule forbids"
_CHATTER = "may be numerical chatter"


def _run(b, t_end, engine, dt=None, **kw):
    if engine == "pwl":
        kw.update(engine="pwl", dt=dt)
    elif engine == "dsed":
        kw.update(engine="dsed")
    with warnings.catch_warnings():
        # Notices about the SIMULATION (sliding-mode, chatter) are not
        # what these tests are about.
        warnings.simplefilter("ignore")
        return p.simulate(b, t_end, **kw)


def _report(b, r, names, **kw):
    """(summary, {name: i_D}, [RuntimeWarning messages])."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = p.device_loss_summary(b, r, **kw)
        cur = {n: np.asarray(r.i(n)) for n in names}
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, RuntimeWarning)]
    return s, cur, msgs


def _entry(summary, name):
    return next(e for e in summary if e["name"] == name)


def _q_load(r, node, C, R):
    """Charge delivered into ``C || R`` on ``node`` (to ground), from
    node voltages alone: ``C·Δv + ∫v/R dt``."""
    t = np.asarray(r.times)
    v = np.asarray(r.v(node))
    return C * (v[-1] - v[0]) + np.trapezoid(v / R, t)


# The summary's i_avg against C·Δv + ∫v/R: exact bookkeeping on the
# fixed engine's trapezoidal companions (measured ~3e-12); the adaptive
# grid integrates the same currents to ~1e-4. DSED stores the LEFT
# limit at an event sample and, separately, the RIGHT limit the next
# step starts from; the summary integrates each step from the latter
# (measured +1.4e-4 at the default 1 µs step on the transformer
# rectifier, second order in the step; see
# test_dsed_summary_starts_each_step_from_the_right_limit).
TOL = {"pwl": 1e-9, "auto": 1e-3, "dsed": 5e-4}


# ---------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------

def _buck():
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("M1", "vin", "sw")          # body diode: its own bit
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "out", 100e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 2.0)
    T = 1 / 50e3
    m1 = b.switch_index_of("M1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(m1, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@pytest.mark.parametrize("engine", ENGINES)
def test_record_is_parallel_to_times_and_follows_mask_indices(engine):
    """One row per stored sample, one column per switch-kind branch,
    column j = switch_index_of(name) — with a MOSFET's body diode
    interleaved between the switch and the freewheel diode."""
    b, pwm, T = _buck()
    r = _run(b, 1e-3, engine, dt=T / 200, switch_fn=pwm)
    S = r.switch_states
    assert S is not None and S.dtype == bool
    assert S.shape == (len(r.times), b.graph.num_switches)
    assert [b.switch_index_of(n) for n in ("M1", "M1_body", "D1")] \
        == [0, 1, 2]
    assert r.switch_states_unsolved.shape == (len(r.times),)
    assert not r.switch_states_unsolved.any()
    # The gate bit is what the engine stamped for each step: the
    # step's end on the fixed engine, its midpoint on TR-BDF2.
    t = np.asarray(r.times)
    q = t if engine == "pwl" else np.concatenate(
        ([t[0]], 0.5 * (t[:-1] + t[1:])))
    expect = np.array([pwm(float(x)).get(0) for x in q])
    assert np.array_equal(S[1:, 0], expect[1:])
    # The freewheel diode conducts during the off-time.
    assert S[:, 2].any() and (~S[:, 2]).any()
    # Compact: ceil(3/8) = 1 byte per sample, < 3 % of the states.
    assert r.switch_states_bytes == len(r.times)
    assert r.switch_states_bytes < 0.03 * r.states_bytes


def test_seeded_initial_state_is_flagged_unsolved():
    b, pwm, T = _buck()
    r0 = _run(b, 2e-4, "pwl", dt=T / 200, switch_fn=pwm)
    x0 = np.asarray(r0.states)[-1].copy()
    r = _run(b, 2e-4, "pwl", dt=T / 200, switch_fn=pwm, initial_state=x0)
    u = r.switch_states_unsolved
    assert u[0] and not u[1:].any()
    # resume_from carries the bits of the solve that produced its
    # state, so its first row IS a stamped mask: not flagged, and equal
    # to the previous run's last row.
    for engine in ENGINES:
        ra = _run(b, 4e-4, engine, dt=T / 200, switch_fn=pwm,
                  resume_from=r0.final_snapshot)
        assert not ra.switch_states_unsolved.any(), engine
        assert np.array_equal(ra.switch_states[0], r0.switch_states[-1])


def _rl_seeded(engine: str):
    """RL half-wave seeded with ``initial_state`` mid-conduction."""
    b = _rl_half_wave(0.7)
    r1 = _run(b, 0.0105, engine, dt=2e-6)
    b2 = _rl_half_wave(0.7)
    r2 = _run(b2, 0.02, engine, dt=2e-6, t_start=0.0105,
              initial_state=np.asarray(r1.states[-1]))
    return b2, r2


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_row_has_no_switched_current(engine):
    """initial_state carries no diode bits: row 0 records the diode
    OFF while the seeded state carries 1.259 A through it (its own
    solve had it ON). Read with the carried row, result.i() missed the
    inductor current by 1.26 A at k = 0. Reading it with the first
    solve's mask instead closed KCL here, but invents a g_on·V current
    where the seed sits on a gate edge (48 kA through a buck's MOSFET,
    test_seeded_state_on_a_gate_edge_has_no_phantom_current). An
    unsolved row has no defined switched-branch current: result.i()
    reads the first solved sample's value there (a real current one
    step late) and the summaries start at that sample."""
    b, r = _rl_seeded(engine)
    bit = b.switch_index_of("D")
    assert r.switch_states_unsolved[0]
    assert not r.switch_states_unsolved[1:].any()
    assert not r.switch_states[0, bit] and r.switch_states[1, bit]
    i_d = np.asarray(r.i("D"))
    i_l = np.asarray(r.i("L"))
    assert i_d[0] == i_d[1] and np.isfinite(i_d).all()
    assert i_l[0] > 1.0
    assert np.abs(i_d[1:] - i_l[1:]).max() <= 1e-9 * np.abs(i_l).max()
    s, cur, msgs = _report(b, r, ["D"])
    assert not msgs, msgs
    assert _entry(s, "D")["i_peak"] == pytest.approx(
        float(np.abs(i_l[1:]).max()), rel=1e-9)
    # The shortened window is in the data, not only in the docs.
    t = np.asarray(r.times)
    for e in s:
        assert e["t_window_start"] == t[1] and e["t_window_end"] == t[-1]


@pytest.mark.parametrize("where", ("gate_edge", "mid_period"))
@pytest.mark.parametrize("engine", ENGINES)
def test_resume_row_zero_is_the_previous_runs_last_row(engine, where):
    """A resume's row 0 is the previous run's last sample: same x, and
    now the same stamped mask (SolverSnapshot.stamped_mask), solved.
    Before, it was labelled with diode_on + switch_fn(t_start): on a
    TR-BDF2 run that ended exactly on a gate edge those are the
    post-edge bits, while x is the pre-edge solve (M1 read ON at the
    pre-edge freewheel state, a g_on·V phantom, with no flag)."""
    b, pwm, T = _buck()
    t_end = 12 * T if where == "gate_edge" else 12.3 * T
    r0 = _run(b, t_end, engine, dt=T / 200, switch_fn=pwm)
    snap = r0.final_snapshot
    last = np.asarray(r0.switch_states[-1], dtype=bool)
    assert list(snap.stamped_mask) == last.tolist()
    b1, pwm1, _ = _buck()
    r1 = _run(b1, t_end + 3 * T, engine, dt=T / 200, switch_fn=pwm1,
              t_start=float(snap.t), resume_from=snap)
    assert not r1.switch_states_unsolved.any()
    assert np.array_equal(r1.switch_states[0], last)
    assert np.array_equal(np.asarray(r1.states[0]),
                          np.asarray(r0.states[-1]))
    i_l = np.abs(np.asarray(r1.i("L1"))).max()
    for n in ("M1", "D1"):
        assert r1.i(n)[0] == pytest.approx(r0.i(n)[-1], rel=1e-12,
                                           abs=1e-12), n
        assert np.abs(np.asarray(r1.i(n))).max() <= 1.01 * i_l, n
    if engine == "auto" and where == "gate_edge":
        # The case the field exists for: the stored sample is the
        # pre-edge solve (M1 OFF, D1 freewheeling) while the bits the
        # next step starts from are already post-edge.
        m1, d1 = b.switch_index_of("M1"), b.switch_index_of("D1")
        assert not last[m1] and last[d1]
        assert pwm(float(snap.t)).get(m1)
        assert not list(snap.diode_on)[-1]


def test_snapshot_without_stamped_mask_flags_row_zero():
    """A snapshot pickled before the field (10 fields), or one whose
    stamped_mask was cleared, resumes with row 0 unsolved."""
    b, pwm, T = _buck()
    r0 = _run(b, 12 * T, "pwl", dt=T / 200, switch_fn=pwm)
    snap = r0.final_snapshot
    st = snap.__getstate__()
    assert len(st) == 12   # 11 + restart_pending (add-discontinuity-restart-rule)
    again = pickle.loads(pickle.dumps(snap))
    assert list(again.stamped_mask) == list(snap.stamped_mask)
    pre = type(snap).__new__(type(snap))   # an 11-field pickle still loads
    pre.__setstate__(tuple(st[:11]))
    assert list(pre.stamped_mask) == list(snap.stamped_mask)
    assert not pre.restart_pending
    old = type(snap).__new__(type(snap))
    old.__setstate__(tuple(st[:10]))
    assert list(old.stamped_mask) == [] and old.valid
    r1 = _run(b, 14 * T, "pwl", dt=T / 200, switch_fn=pwm,
              t_start=float(old.t), resume_from=old)
    assert r1.switch_states_unsolved[0]
    assert not r1.switch_states_unsolved[1:].any()
    assert r1.i("M1")[0] == r1.i("M1")[1]


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_state_on_a_gate_edge_has_no_phantom_current(engine):
    """initial_state taken from a TR-BDF2 run that ended on a gate edge
    (pre-edge freewheel state, M1 turning ON at t_start). Reading the
    unsolved row 0 with the first solve's mask (M1 ON) gave M1
    g_on · 48 V: i_peak 48023.6 A, P_avg 36x high on 'auto' and 3500x
    on 'pwl', no warning. Row 0 is now left out of the summaries, and
    result.i() reads the first solved sample there."""
    b, pwm, T = _buck()
    r0 = _run(b, 12 * T, "auto", switch_fn=pwm)
    x0 = np.asarray(r0.states)[-1].copy()
    b1, pwm1, _ = _buck()
    r1 = _run(b1, 17 * T, engine, dt=T / 200, switch_fn=pwm1,
              t_start=float(r0.times[-1]), initial_state=x0)
    assert r1.switch_states_unsolved[0]
    for n in ("M1", "D1"):
        i = np.asarray(r1.i(n))
        assert i[0] == i[1] and np.isfinite(i).all(), n
        assert np.abs(i).max() <= 1.01 * np.abs(r1.i("L1")).max(), n
    s, _, msgs = _report(b1, r1, [])
    assert not msgs, msgs
    i_l = float(np.abs(np.asarray(r1.i("L1"))).max())
    for n in ("M1", "D1"):
        assert _entry(s, n)["i_peak"] <= 1.01 * i_l, (n, _entry(s, n))
    # The same window of an unbroken run: the averages agree.
    rc = _run(b, 17 * T, engine, dt=T / 200, switch_fn=pwm)
    t = np.asarray(rc.times)
    i_m = np.abs(np.asarray(rc.i("M1")))
    w = t >= float(r1.times[1])
    ref = np.trapezoid(i_m[w], t[w]) / (t[w][-1] - t[w][0])
    assert _entry(s, "M1")["i_avg"] == pytest.approx(ref, rel=0.02)
    if engine == "pwl":
        th = p.device_thermal_summary(
            b1, r1, thermal_specs={"M1": {"stages": [
                p.FosterStage(R_th_K_per_W=1.0, tau_s=1e-4)]}})
        assert th[0]["T_j_trace"].shape == np.asarray(r1.times).shape
        assert th[0]["T_j_trace"][0] == 25.0
        assert th[0]["t_window_start"] == float(r1.times[1])
        assert th[0]["T_j_peak"] < 30.0


def _resume_buck(src, t_end, dt, seed):
    b, pwm, T = _buck()
    snap = _run(b, t_end, src, dt=dt, switch_fn=pwm).final_snapshot
    b1, pwm1, _ = _buck()
    kw = (dict(resume_from=snap) if seed == "resume"
          else dict(initial_state=np.asarray(snap.x)))
    r1 = _run(b1, 15 * T, "auto", switch_fn=pwm1, t_start=float(snap.t),
              **kw)
    return b1, r1, float(snap.t)


@pytest.mark.parametrize("seed", ("resume", "initial_state"))
@pytest.mark.parametrize("src,k_ulp", [("pwl", 0), ("auto", 1),
                                       ("auto", 50), ("auto", 1000)])
def test_resume_just_before_a_gate_edge_matches_the_edge(src, k_ulp, seed):
    """A TR-BDF2 run resumed 1..1000 ulp BEFORE M1's turn-on edge (a
    pwl snapshot at 12 T with dt = 1e-7 lands 2 ulp short) settled its
    diodes under the pre-edge gates, but its first step reads them at
    t + h_min / 2, past the edge, without landing it: t[1] was stored
    with M1 ON and D1 still ON (M1 i_peak 24011.8 A, P_avg 2.187 W;
    resumed exactly on the edge: 25.3 A, 0.259 W), and, as a
    commutation landing, without a warning. The settle now reads the
    gates where the first step does."""
    T = 1 / 50e3
    ulp = np.spacing(12 * T)
    # dt = 1e-7 (not T / 200 = 1.0000000000000001e-7): 2400 steps of it
    # end 2 ulp short of 12 T.
    b1, r1, t0 = _resume_buck(src, 12 * T - k_ulp * ulp, 1e-7, seed)
    assert 0 < 12 * T - t0 < 1e-15
    s, _, msgs = _report(b1, r1, [])
    assert not msgs, msgs
    # Reference: 1e6 ulp (27 fs) short, far enough for the first step
    # to land the edge. Measured within 2e-3 (pwl P_avg) of the above.
    b2, r2, t2 = _resume_buck(src, 12 * T - 1e6 * ulp, 1e-7, seed)
    s2, _, _ = _report(b2, r2, [])
    for n in ("M1", "D1"):
        for k in ("i_peak", "P_avg"):
            assert _entry(s, n)[k] == pytest.approx(_entry(s2, n)[k],
                                                    rel=1e-2), (n, k)


class _Edited:
    """A recorded result with its states replaced (the rule check reads
    nothing else)."""

    def __init__(self, r, states):
        self.times = r.times
        self.states = states
        self.switch_states = np.asarray(r.switch_states)
        self.switch_states_unsolved = np.asarray(r.switch_states_unsolved)
        self.commutation_events = list(r.commutation_events)
        self.event_iteration_breaches = []
        self.engine_used = r.engine_used


def _rl_auto_landing():
    """(builder, result, sample index of a D turn-off landing stored
    under the ON stamp, anode row, cathode row) on the RL half-wave.

    This used the 50 kHz buck of ``_buck`` until the TR-BDF2 crossing
    detector stopped reading the step start from the PRE-gate-edge
    solve: the buck's D1 landings were that detector "finding" each
    gate-forced commutation a second time, and a CCM buck now has none
    (its commutations are decided at the gate edges). The half-wave's
    turn-off at the current zero is a real landing."""
    b = _rl_half_wave(0.7)
    r = _run(b, 4 / 60, "auto")
    d = b.branch_index_of("D")
    bit = b.switch_index_of("D")
    t = np.asarray(r.times)
    rec = np.asarray(r.switch_states)
    k = next(int(np.searchsorted(t, ev.t_estimated))
             for ev in r.commutation_events
             if ev.branch_id == d
             and t[np.searchsorted(t, ev.t_estimated)] == ev.t_estimated
             and rec[np.searchsorted(t, ev.t_estimated), bit])
    a, kn = next(c for c in b.components() if c["name"] == "D")["nodes"]
    return b, r, k, a, kn


def test_landing_outside_the_envelope_is_checked():
    """The rule check's landing exemption, on a recorded auto run whose
    landings all sit inside the envelope: a turn-off landing stored
    under the ON stamp, edited to a shoot-through-sized reverse bias
    (v_D = -24 V, 24 kA against a 6.4 A peak on every other sample) is
    named; one edited to a reverse current within the run's peak (5 A)
    stays exempt."""
    from pulsim._diode_state import diode_state_traces

    b, r, k, a, kn = _rl_auto_landing()
    d = b.branch_index_of("D")
    S = np.array(r.states, dtype=float)

    def check(v_d):
        S2 = S.copy()
        S2[k, kn] = S[k, a] - v_d
        return diode_state_traces(b, _Edited(r, S2))[d]

    assert check(S[k, a] - S[k, kn]).n_rule_violations == 0
    assert check(-0.005).n_rule_violations == 0         # 5 A reverse
    tr = check(-24.0)
    assert tr.n_rule_violations == 1
    assert tr.max_violation_V == pytest.approx(24.0)
    assert any("commutation landing" in m for m in tr._messages)


def test_source_step_landing_stays_inside_the_envelope():
    """The other side of the landing envelope: the pulsed charger's
    source steps 2.9 V at each edge, so its auto landings hold D OFF
    at 2.4 V forward (1.7 V past V_th) for one sample — the largest
    legitimate landing violation measured, inside the 14.4 V the
    circuit reaches. No rule warning."""
    b = _pulsed_charger()
    r = _run(b, 1e-3, "auto")
    _, _, msgs = _report(b, r, [])
    assert not any(_RULE in m for m in msgs), msgs


def test_switch_duty_needs_no_switch_fn():
    """The record makes a switch's state solver-owned data: the summary
    reports it with no switch_fn at all, on both engines."""
    for engine in ENGINES:
        b, pwm, T = _buck()
        r = _run(b, 2e-3, engine, dt=T / 200, switch_fn=pwm)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            s = p.device_loss_summary(b, r)
        assert _entry(s, "M1")["duty_closed"] == pytest.approx(0.5,
                                                               abs=3e-3)


# ---------------------------------------------------------------------
# Reproducer 1 — snubber capacitor directly across the diode
# ---------------------------------------------------------------------

def _snubbed_boost(V_th: float, Cs: float):
    """24 V, 220 µH, Q1 g_on = 50 S at 50 kHz / D = 0.5, D1 with Cs
    directly across it, 100 µF || 30 Ω."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_inductor("L1", "vin", "sw", 220e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=50.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("Cs", "sw", "out", Cs)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 30.0)
    T = 2e-5
    q1 = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q1, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


def _snubber_charge(r, Cs):
    """Charge through D1 by KCL at the output node, from node voltages
    alone: C1·Δv_out + ∫v_out/R − Cs·Δ(v_sw − v_out)."""
    v_sn = np.asarray(r.v("sw")) - np.asarray(r.v("out"))
    return _q_load(r, "out", 100e-6, 30.0) - Cs * (v_sn[-1] - v_sn[0])


@pytest.mark.parametrize("Cs", (1e-9, 100e-9))
@pytest.mark.parametrize("V_th", (0.7, 0.0))
def test_snubber_across_the_diode_closes_the_charge_on_pwl(V_th, Cs):
    """The charge closes on what the solver stamped. At V_th = 0.7 that
    stamp chatters inside the conduction (the trapezoidal rule reflects
    the Cs-g_on step from +V_th to about -V_th), and the summary says
    so by name; at V_th = 0 it does not chatter and nothing warns."""
    b, pwm, T = _snubbed_boost(V_th, Cs)
    r = _run(b, 3e-3, "pwl", dt=T / 400, switch_fn=pwm)
    s, cur, msgs = _report(b, r, ["D1"])
    if V_th > 0:
        assert msgs and all(_CHATTER in m for m in msgs), msgs
        assert _entry(s, "D1")["n_possible_chatter_gaps"] > 100
    else:
        assert not msgs, msgs
    q_d = np.trapezoid(cur["D1"], np.asarray(r.times))
    q_phys = _snubber_charge(r, Cs)
    assert q_phys > 5e-3
    assert abs(q_d - q_phys) <= TOL["pwl"] * q_phys, (q_d, q_phys)
    assert _entry(s, "D1")["duty_conducting"] > 0.2


def test_snubber_across_the_diode_on_auto_reports_what_was_carried():
    """engine='auto' closes the same balance. Its own trajectory holds
    D1 across its commutation threshold on a cluster of samples near
    the first gate edge (an engine defect, reported by name), so the
    only warning allowed is that one."""
    b, pwm, T = _snubbed_boost(0.7, 1e-9)
    r = _run(b, 3e-3, "auto", switch_fn=pwm)
    _, cur, msgs = _report(b, r, ["D1"])
    assert all(_RULE in m for m in msgs), msgs
    q_d = np.trapezoid(cur["D1"], np.asarray(r.times))
    q_phys = _snubber_charge(r, 1e-9)
    assert abs(q_d - q_phys) <= TOL["auto"] * q_phys, (q_d, q_phys)


# ---------------------------------------------------------------------
# Reproducers 2 and 3 — ideal-transformer secondary, VCVS probe
# ---------------------------------------------------------------------

def _xfmr_rectifier(V_th: float):
    """100 V / 50 Hz, 0.1 Ω, ideal transformer n = 0.2, D, 1000 µF ||
    20 Ω on the secondary."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "gnd", v_dc=0.0,
                              v_amplitude=100.0, frequency=50.0, phase=0.0)
    b.add_resistor("Rp", "s", "pri", 0.1)
    b.add_ideal_transformer("T1", "pri", "gnd", "sec", "gnd", 0.2)
    b.add_diode("D", "sec", "k", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("C", "k", "gnd", 1000e-6)
    b.add_resistor("RL", "k", "gnd", 20.0)
    return b


def _vcvs_probe_rectifier(V_th: float):
    """Series rectifier (0.1 Ω, D, 1000 µF || 20 Ω) plus a VCVS used as
    a measurement probe on an unrelated node."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", v_dc=0.0,
                              v_amplitude=100.0, frequency=50.0, phase=0.0)
    b.add_resistor("Rp", "s", "ss", 0.1)
    b.add_diode("D", "ss", "k", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("C", "k", "gnd", 1000e-6)
    b.add_resistor("RL", "k", "gnd", 20.0)
    b.add_vcvs("E1", "s", "gnd", "probe", "gnd", 0.01)
    b.add_resistor("Rprobe", "probe", "gnd", 1e3)
    return b


@functools.lru_cache(maxsize=None)
def _rectifier_run(kind: str, V_th: float, engine: str):
    b = (_xfmr_rectifier if kind == "xfmr" else _vcvs_probe_rectifier)(V_th)
    r = _run(b, 0.1, engine, dt=2e-6)
    s, cur, msgs = _report(b, r, ["D"])
    return b, r, s, cur, msgs


# DSED runs these circuits only at V_th = 0.7; at V_th = 0 its diode
# predicate does not settle at t = 0 from a zero state (a DSED
# start-up limitation, not a reporting one).
_RECT_CASES = [(k, v, e) for k in ("xfmr", "vcvs") for v in (0.7, 0.0)
               for e in ENGINES] + [("xfmr", 0.7, "dsed"),
                                     ("vcvs", 0.7, "dsed")]


@pytest.mark.parametrize("kind,V_th,engine", _RECT_CASES)
def test_rectifier_charge_closes_through_any_device(kind, V_th, engine):
    _, r, s, cur, msgs = _rectifier_run(kind, V_th, engine)
    assert not msgs, msgs
    t = np.asarray(r.times)
    e = _entry(s, "D")
    q_d = e["i_avg"] * (t[-1] - t[0])
    q_phys = _q_load(r, "k", 1000e-6, 20.0)
    assert q_phys > 0.05
    assert abs(q_d - q_phys) <= TOL[engine] * q_phys, (q_d, q_phys)
    if engine != "dsed":
        # No stored right limit: the plain trapezoid of r.i() is the
        # same integral.
        assert np.trapezoid(cur["D"], t) == pytest.approx(q_d, rel=1e-12)
    assert 0.2 < e["duty_conducting"] < 0.23, e


def test_dsed_summary_starts_each_step_from_the_right_limit():
    """A DSED event sample is the LEFT limit (the stamped-mask
    convention), so a current that jumps at a V_th turn-on is drawn as
    a ramp over the next step by a plain trapezoid of r.i(): first order
    in the step, with the sign of a missing half-step of current
    (measured -4.22e-3, -2.15e-3 at the default step and dt = 5e-6).
    The result also stores the RIGHT limit at each event, and the
    summary integrates every step from it: +1.39e-4 and +3.5e-5, second
    order. Only the 10 samples where an event changed the state carry
    a right limit."""
    raw, summ = [], []
    for kw in ({}, {"dt": 5e-6}):
        b = _xfmr_rectifier(0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b, 0.1, engine="dsed", **kw)
            s = p.device_loss_summary(b, r)
        t = np.asarray(r.times)
        q_phys = _q_load(r, "k", 1000e-6, 20.0)
        q_raw = np.trapezoid(np.asarray(r.i("D")), t)
        q_sum = _entry(s, "D")["i_avg"] * (t[-1] - t[0])
        raw.append((q_raw - q_phys) / q_phys)
        summ.append((q_sum - q_phys) / q_phys)
        assert r.right_limit_index.size == r.n_events
        assert r.right_limit_states.shape == (
            r.n_events, np.asarray(r.states).shape[1])
    assert raw[0] < 0 and raw[1] < 0, raw
    assert 0.35 < raw[1] / raw[0] < 0.65, raw
    assert abs(summ[0]) < 3e-4 and abs(summ[1]) < 0.4 * abs(summ[0]), summ


@pytest.mark.parametrize("engine", ENGINES + ("dsed",))
def test_vcvs_probe_diode_is_the_series_resistor_current(engine):
    """Sample by sample, against an independent branch."""
    _, r, s, cur, _ = _rectifier_run("vcvs", 0.7, engine)
    i_rp = (np.asarray(r.v("s")) - np.asarray(r.v("ss"))) / 0.1
    assert float(np.abs(i_rp).max()) > 30.0
    assert np.allclose(cur["D"], i_rp, rtol=0, atol=1e-8)
    # Same diode at V_th = 0 (no band at all): the same loss.
    _, _, s0, _, _ = _rectifier_run("vcvs", 0.0, "pwl")
    assert _entry(s, "D")["P_avg"] == pytest.approx(
        _entry(s0, "D")["P_avg"], rel=5e-3)


@pytest.mark.parametrize("engine", ENGINES)
def test_transformer_diode_is_the_secondary_state_current(engine):
    _, r, _, cur, _ = _rectifier_run("xfmr", 0.7, engine)
    i_sec = np.asarray(r.i("T1"))
    assert float(np.abs(i_sec).max()) > 5.0
    # The secondary branch current is a state variable; the diode
    # carries it (with the branch's sign convention: sec -> gnd).
    assert np.allclose(np.abs(cur["D"]), np.abs(i_sec), rtol=0,
                       atol=1e-6 * float(np.abs(i_sec).max()))


# ---------------------------------------------------------------------
# Reproducer 4 — RL half-wave: immaterial conduction is not a recovery
# ---------------------------------------------------------------------

def _rl_half_wave(V_th: float):
    """50 V / 60 Hz, D, 5 Ω + 20 mH, 4 cycles."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "a", "gnd", v_dc=0.0, v_amplitude=50.0,
                              frequency=60.0, phase=0.0)
    b.add_diode("D", "a", "k", 1e3, 1e-9, V_th=V_th)
    b.add_resistor("R", "k", "m", 5.0)
    b.add_inductor("L", "m", "gnd", 20e-3)
    return b


@functools.lru_cache(maxsize=None)
def _rl_run(V_th: float, engine: str):
    b = _rl_half_wave(V_th)
    r = _run(b, 4 / 60, engine, dt=2e-6)
    # t_rr = 5 µs (a line-frequency rectifier diode): every engine's
    # grid resolves it (pwl dt = 2 µs, dsed dt_max = 1 µs), so V_R is
    # read at the same physical instant everywhere and nothing warns.
    s, cur, msgs = _report(b, r, ["D"],
                           diode_specs={"D": {"Q_rr": 1e-6, "t_rr": 5e-6}})
    return r, _entry(s, "D"), cur["D"], msgs


_RL_CASES = [(v, e) for v in (0.7, 2.0, 0.0) for e in ENGINES] \
    + [(0.7, "dsed"), (2.0, "dsed")]


@pytest.mark.parametrize("V_th,engine", _RL_CASES)
def test_rl_half_wave_recovery_counts_real_turn_offs_only(V_th, engine):
    """The fixed engine's trapezoidal g_off/L stiff mode turns the
    reverse-biased diode ON for single ~36 µA steps (5.6e-6·i_peak,
    ~7e-11 C). They were each charged Q_rr·V_R: 48 events and 4× the
    energy on pwl at V_th = 0.7, and auto counted 8 (f_sw 120 Hz).
    Four cycles, four recoveries, on every engine and V_th; the
    dropped stretches are reported, and E_sw matches V_th = 0 on pwl.
    V_R is read at t_off + t_rr on every engine (measured at
    t_rr = 100 ns: 1.7218e-4 J pwl, 1.7221e-4 auto and dsed; the
    previous two-mean-step window gave 1.7236e-4 / 1.755e-4 /
    1.7226e-4)."""
    r, e, i_d, msgs = _rl_run(V_th, engine)
    assert not msgs, msgs
    assert e["n_turn_off_events"] == 4, e
    assert e["f_sw_estimate"] == pytest.approx(60.0, rel=1e-3), e
    _, e_ref, _, _ = _rl_run(0.0, "pwl")
    assert e["E_sw_total"] == pytest.approx(e_ref["E_sw_total"], rel=0.025)
    if engine == "pwl" and V_th > 0:
        assert e["n_turn_off_events_dropped"] > 0, e
    # Charge: the diode carries the inductor's state current.
    i_l = np.asarray(r.i("L"))
    t = np.asarray(r.times)
    assert np.allclose(i_d, i_l, rtol=0, atol=1e-9 * 1e3)
    q = np.trapezoid(i_l, t)
    assert abs(np.trapezoid(i_d, t) - q) <= 1e-9 * abs(q)


def test_recovered_energy_is_capped_at_the_forward_charge():
    """A stretch that conducted less charge than Q_rr cannot recover
    Q_rr: the event is capped at Q_F·V_R and counted."""
    from pulsim.losses import _diode_switching_loss
    t = np.linspace(0.0, 1e-3, 11)
    ev = [{"t": 5e-4, "V_R": 10.0, "I_F": 1.0, "I_F_peak": 1.0,
           "Q_F": 2e-8, "material": True},
          {"t": 6e-4, "V_R": 10.0, "I_F": 1.0, "I_F_peak": 1.0,
           "Q_F": 1e-3, "material": True},
          {"t": 7e-4, "V_R": 10.0, "I_F": 1e-6, "I_F_peak": 1e-6,
           "Q_F": 1e-12, "material": False}]
    out = _diode_switching_loss(ev, t, np.zeros_like(t), {"Q_rr": 1e-7})
    assert out["n_turn_off_events"] == 2
    assert out["n_turn_off_events_dropped"] == 1
    assert out["n_turn_off_events_capped"] == 1
    assert out["E_sw_total"] == pytest.approx(2e-8 * 10 + 1e-7 * 10)


# ---------------------------------------------------------------------
# Reproducer 5 — bridge with line inductance on engine='auto'
# ---------------------------------------------------------------------

def _bridge_with_line_l(V_th: float):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                              frequency=50.0, phase=0.3)
    b.add_resistor("Rs", "s", "m", 0.2)
    b.add_inductor("Ls", "m", "a", 1e-3)
    p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b",
                                    dc_pos="p", dc_neg="gnd", V_th=V_th)
    b.add_capacitor("C", "p", "gnd", 220e-6)
    b.add_resistor("RL", "p", "gnd", 100.0)
    return b


BRIDGE = ("BR__D1", "BR__D2", "BR__D3", "BR__D4")


@pytest.mark.parametrize("engine", ENGINES)
def test_line_inductance_bridge_has_no_phantom_landing_peak(engine):
    """An event landing is a real trap solve under the PRE-commutation
    mask and is recorded as such, so no sample reports g_on·V_th from
    the post-flip bits: every diode current closes KCL against the line
    inductor's STATE current at every sample (before: 268 A off at one
    landing, D3 reported at 700.5 A with the inductor at 294.1 A).

    The 294 A was in the auto trajectory (the pwl run peaks at 65 A):
    engine='auto' held BR__D2/BR__D4 ON against the commutation rule for
    25 samples and discharged the DC capacitor through them. The engine
    now re-decides a diode held on the forbidden side and reads each
    step's start from the solve its bits were decided on: the auto run
    peaks at 65.32 A (pwl 65.35 A) and nothing is held."""
    b = _bridge_with_line_l(0.7)
    r = _run(b, 0.08, engine, dt=1e-6)
    s, cur, msgs = _report(b, r, BRIDGE)
    assert not msgs, msgs
    i_l = np.asarray(r.i("Ls"))
    i_pk = float(np.abs(i_l).max())
    assert np.abs(i_l - (cur["BR__D1"] - cur["BR__D4"])).max() \
        <= 1e-9 * i_pk
    assert np.abs(i_l - (cur["BR__D3"] - cur["BR__D2"])).max() \
        <= 1e-9 * i_pk
    for name in BRIDGE:
        assert _entry(s, name)["i_peak"] <= i_pk * (1 + 1e-9), name
    t = np.asarray(r.times)
    q_in = np.trapezoid(cur["BR__D1"] + cur["BR__D2"], t)
    q_dc = _q_load(r, "p", 220e-6, 100.0)
    assert abs(q_in - q_dc) <= TOL[engine] * abs(q_dc)


# ---------------------------------------------------------------------
# Reference bridge: 325 V / 50 Hz, 1 mΩ, defaults, 470 µF || 1 kΩ
# ---------------------------------------------------------------------

def _reference_bridge(V_th: float):
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                              frequency=50.0, phase=0.0)
    b.add_resistor("Rs", "s", "a", 1e-3)
    p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b",
                                    dc_pos="p", dc_neg="gnd", V_th=V_th)
    b.add_capacitor("C", "p", "gnd", 470e-6)
    b.add_resistor("RL", "p", "gnd", 1000.0)
    return b


@functools.lru_cache(maxsize=None)
def _ref_run(V_th: float, engine: str, dt: float = 1e-6):
    b = _reference_bridge(V_th)
    r = _run(b, 0.06, engine, dt=dt)
    return _report(b, r, BRIDGE) + (r,)


def test_reference_bridge_agrees_across_engines_at_v_th_0_7():
    """auto vs a converged pwl run (dt = 0.1 µs) at the helper defaults
    (V_th = 0.7). Measured: D1 rms 10.253 / 10.244 A, P_avg 0.10520 /
    0.10515 W; D2 rms 2.957 / 2.922 A, P_avg 0.00885 / 0.00875 W; duty
    0.1030 and 0.0293 on both.

    D2's loss is dominated by its V_th turn-on spike (the diode voltage
    collapses from 0.7 V into a ~3 mΩ loop with no inductance; τ ≈
    1.4 µs), so it is a grid property: pwl at dt = 1 µs reads a 345 A
    peak and D2 rms 2.702 A (dt = 0.25 µs: 429 A, 2.878 A; 0.1 µs:
    451 A, 2.922 A), auto 458 A and 2.957 A. Each step is integrated
    as the line between its ends, so the dt = 1 µs run no longer
    overstates the spike's decaying step by the trapezoid of its
    squares (it read 2.937 A with that, close to converged only by
    accident). Peaks are therefore not compared."""
    s_p, _, m_p, _ = _ref_run(0.7, "pwl", dt=1e-7)
    s_a, _, m_a, _ = _ref_run(0.7, "auto")
    assert not m_p and not m_a, (m_p, m_a)
    for name in BRIDGE:
        a, b_ = _entry(s_p, name), _entry(s_a, name)
        assert b_["i_rms"] == pytest.approx(a["i_rms"], rel=0.02), name
        assert abs(b_["duty_conducting"] - a["duty_conducting"]) < 1e-3
        assert b_["P_avg"] == pytest.approx(a["P_avg"], rel=0.02), name
        assert a["duty_conducting"] > 0.02 and a["P_avg"] > 1e-3


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("V_th", (0.7, 0.0))
def test_reference_bridge_charge_and_source_kcl(V_th, engine):
    _, cur, msgs, r = _ref_run(V_th, engine)
    if engine == "pwl":
        assert not msgs, msgs
    else:
        # engine='auto' at V_th = 0 holds D1/D3 OFF while forward
        # biased (a missed commutation) and says so.
        assert all(_RULE in m for m in msgs), msgs
    t = np.asarray(r.times)
    q_dc = _q_load(r, "p", 470e-6, 1000.0)
    q_in = np.trapezoid(cur["BR__D1"] + cur["BR__D2"], t)
    assert abs(q_in - q_dc) <= TOL[engine] * q_dc
    i_src = -np.asarray(r.i("Vac"))
    assert np.abs(i_src - (cur["BR__D1"] - cur["BR__D4"])).max() \
        <= 1e-9 * float(np.abs(i_src).max())


# ---------------------------------------------------------------------
# Results without a record fall back, loudly
# ---------------------------------------------------------------------

class _NoRecord:
    """A result-like object built without the kernel record."""

    def __init__(self, r):
        self.times = r.times
        self.states = np.asarray(r.states)
        self._builder = r._builder
        self.engine_used = r.engine_used


def test_result_without_record_falls_back_with_a_named_reason():
    b = _vcvs_probe_rectifier(0.7)
    r = _run(b, 0.02, "pwl", dt=2e-6)
    bare = _NoRecord(r)
    with pytest.warns(RuntimeWarning,
                      match=r"diode 'D'.*no record.*switch_states"):
        s = p.device_loss_summary(b, bare)
    # The replay cannot see a turn-on the fixed engine's event
    # iteration resolved inside one step (the stored sample before it
    # is below V_th, the one after it already in the band), so here it
    # reports the original defect — near-zero current — which is
    # exactly why the warning is mandatory and the record exists.
    e = _entry(s, "D")
    e_rec = _entry(_report(b, r, [])[0], "D")
    assert abs(e["i_avg"]) < 1e-3 * e_rec["i_avg"]
    # Once per result and diode: the summary above already warned
    # about 'D' on `bare`; another result without the record warns.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        p.SimulationResult.i(bare, "D")
    assert not [w for w in rec if "diode 'D'" in str(w.message)]
    with pytest.warns(RuntimeWarning, match=r"diode 'D'.*no record"):
        p.SimulationResult.i(_NoRecord(r), "D")


def test_adaptive_result_stitches_the_record():
    b = _vcvs_probe_rectifier(0.7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ar = p.adaptive.run_transient_adaptive(
            b, t_end=0.01, dt_init=5e-6, segment_steps=200)
    assert ar.switch_states is not None
    assert ar.switch_states.shape == (ar.times.size, b.graph.num_switches)
    assert not ar.switch_states_unsolved.any()


# ---------------------------------------------------------------------
# Review round 2 — DSED records the mask of the STEP, not the post-event
# mask
# ---------------------------------------------------------------------

def _boost(V_th: float, R_load: float):
    """12 V → boost, 100 µH, Q 100 kHz at D = 0.5, D1 sw → out,
    47 µF || R_load (500 Ω: DCM, 20 Ω: CCM)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("Q", "sw", "gnd", g_on=100.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("C", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", R_load)
    T = 1e-5
    q = b.switch_index_of("Q")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@functools.lru_cache(maxsize=None)
def _boost_run(V_th: float, R_load: float, engine: str, div: int = 1000):
    b, pwm, T = _boost(V_th, R_load)
    r = _run(b, 3e-3, engine, dt=T / div, switch_fn=pwm)
    return b, r


@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_dsed_duty_matches_the_fixed_engine(R_load):
    """DSED fires an event and then stored the POST-event mask, so each
    event sample booked the step that ended there to the new mode: the
    DCM diode read duty 0.2430 against 0.2209 on pwl (steady state
    0.2000 against a true 0.1780), CCM 0.4767 against 0.4665, with no
    warning. Recording the mask of the step (and reconstructing the
    event sample through it, from the pre-event state) gives 0.2210 and
    0.4667. The switch's duty needs no switch_fn on either engine."""
    out = {}
    for engine in ("pwl", "dsed"):
        b, r = _boost_run(0.7, R_load, engine)
        s, _, msgs = _report(b, r, [])
        assert not msgs, (engine, msgs)
        out[engine] = s
    for name, key in (("D1", "duty_conducting"), ("Q", "duty_closed")):
        assert _entry(out["dsed"], name)[key] == pytest.approx(
            _entry(out["pwl"], name)[key], abs=0.005), name
    assert _entry(out["dsed"], "Q")["duty_closed"] == pytest.approx(
        0.5, abs=1e-3)


def test_dsed_record_books_each_step_to_the_mode_integrated_over_it():
    """Over every step, the recorded mode must be the one whose
    dynamics the inductor followed: Q ON → di_L/dt = V_in/L > 0; D1 ON
    with Q OFF → di_L/dt = (V_in − V_out)/L < 0; both OFF (DCM idle) →
    i_L ≈ 0. Steady-state D1 duty from the record matches the i_L
    down-slope time (0.1780) to 1e-3; the post-event convention read
    0.2000. Q's recorded duty is 0.5."""
    b, r = _boost_run(0.7, 500.0, "dsed")
    t = np.asarray(r.times)
    i_l = np.asarray(r.i("L"))
    S = r.switch_states
    q_on = S[1:, b.switch_index_of("Q")]
    d_on = S[1:, b.switch_index_of("D1")]
    di = np.diff(i_l)
    tol = 1e-9 * np.abs(i_l).max()
    # After start-up (V_out > V_in from ~0.3 ms on).
    late = t[1:] > 1e-3
    assert (di[q_on & late] > -tol).all()
    assert (di[d_on & ~q_on & late] < tol).all()
    idle = ~q_on & ~d_on & late
    assert idle.any()
    assert np.abs(i_l[1:][idle]).max() < 1e-6 * np.abs(i_l).max()
    dt = np.diff(t)
    late = t[1:] > 2e-3
    falling = (di < 0) & (i_l[:-1] > 1e-6 * np.abs(i_l).max())
    T_late = dt[late].sum()
    duty_truth = np.dot(falling[late], dt[late]) / T_late
    duty_rec = np.dot((d_on & ~q_on)[late], dt[late]) / T_late
    assert duty_truth == pytest.approx(0.178, abs=2e-3)
    assert duty_rec == pytest.approx(duty_truth, abs=1e-3)
    q_duty = np.dot(q_on[late], dt[late]) / T_late
    assert q_duty == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------
# Review round 2 — V_R at a physical instant, t_off + t_rr
# ---------------------------------------------------------------------

def _line_bridge():
    """325 V / 50 Hz, 1 mΩ, add_bridge_rectifier (V_th = 0.7),
    470 µF || 1 kΩ."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "s", "b", v_dc=0.0, v_amplitude=325.0,
                              frequency=50.0, phase=0.0)
    b.add_resistor("Rs", "s", "a", 1e-3)
    p.topology.add_bridge_rectifier(b, "BR", ac_a="a", ac_b="b",
                                    dc_pos="p", dc_neg="gnd", V_th=0.7)
    b.add_capacitor("C", "p", "gnd", 470e-6)
    b.add_resistor("RL", "p", "gnd", 1000.0)
    return b


def test_line_rectifier_recovery_is_negligible_on_every_grid():
    """A capacitor-input line rectifier commutates naturally: the diode
    current decays to zero and v_D then rises as (t − t_0)², so there
    is essentially no voltage to recover against. The two-mean-step
    window read V_R 1.66 mV on pwl (dt = 1 µs) and 1.81 V on auto
    (200 µs steps): E_sw 1.66e-8 J vs 1.81e-5 J, 1087× apart, silently.
    Read at t_off + t_rr both are below 1e-12 J here, and auto, whose
    step cannot resolve t_rr, says so."""
    spec = {"BR__D1": {"Q_rr": 1e-6}}
    e = {}
    for engine in ("pwl", "auto"):
        b = _line_bridge()
        r = _run(b, 0.1, engine, dt=1e-6)
        s, _, msgs = _report(b, r, [], diode_specs=spec)
        e[engine] = (_entry(s, "BR__D1"), msgs)
    for engine, (d, _) in e.items():
        assert d["n_turn_off_events"] == 5, (engine, d)
        assert d["E_sw_total"] < 1e-12, (engine, d)
        assert d["E_sw_total"] < 1e-9 * d["P_avg"] * 0.1, (engine, d)
    d_auto, m_auto = e["auto"]
    assert d_auto["n_turn_off_events_unresolved"] > 0
    assert not m_auto or any("longer than t_rr" in m for m in m_auto)


def _snubbed_dcm_boost(V_th: float = 0.7):
    """The DCM boost (500 Ω) with an RC snubber across the switch node
    and 10 nH in series with D1: its L1–Cs ring after each natural
    turn-off is what a window counted in steps read differently on
    every grid."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L1", "vin", "sw", 100e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=100.0, g_off=1e-9)
    b.add_diode("D1", "sw", "x", 1e3, 1e-9, V_th=V_th)
    b.add_inductor("Ld", "x", "out", 10e-9)
    b.add_resistor("Rs", "sw", "sm", 10.0)
    b.add_capacitor("Cs", "sm", "x", 1e-9)
    b.add_capacitor("C1", "out", "gnd", 47e-6)
    b.add_resistor("RL", "out", "gnd", 500.0)
    T = 1e-5
    q = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@functools.lru_cache(maxsize=None)
def _snubbed_report(div):
    b, pwm, T = _snubbed_dcm_boost()
    if div == "auto":
        r = _run(b, 2e-3, "auto", switch_fn=pwm)
    else:
        r = _run(b, 2e-3, "pwl", dt=T / div, switch_fn=pwm)
    s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 100e-9}})
    return _entry(s, "D1"), msgs


def test_recovery_energy_does_not_follow_the_step_size():
    """Snubbed DCM boost, Q_rr = 100 nC. Two-mean-step window: E_sw
    1.059e-4 (pwl T/1000), 1.367e-4 (T/200), 2.406e-4 (T/50),
    8.734e-4 J (T/20), no warning. At t_off + t_rr: 1.302e-4, 1.253e-4
    and 1.332e-4 J (auto) on the grids that resolve t_rr = 100 ns."""
    ref, m_ref = _snubbed_report(1000)
    assert not m_ref, m_ref
    assert ref["n_turn_off_events"] == 200
    for div in (200, "auto"):
        d, msgs = _snubbed_report(div)
        assert not msgs, (div, msgs)
        assert d["n_turn_off_events"] == 200, div
        assert d["n_turn_off_events_unresolved"] == 0, div
        assert d["E_sw_total"] == pytest.approx(ref["E_sw_total"],
                                                rel=0.08), div


@pytest.mark.parametrize("div", (50, 20))
def test_unresolved_recoveries_and_capped_blips_warn(div):
    """dt = T/50 and T/20 are longer than t_rr = 100 ns and than the
    snubber's 10 ns time constant: the turn-offs are counted as
    unresolved, and the trapezoidal ring's one-step re-conductions
    (239 recorded turn-offs at T/50 on a 200-cycle run, 212 at T/20)
    are counted as recorded; each extra one is capped at its own
    forward charge (31 / 12, warned about) or is a 1-step gap next to a
    1-step re-conduction that the chatter detector reports (33 / 4).
    Both used to pass silently.
    (Round 3 merged the gaps and pinned <= 210; merging is gone: the
    record is what the solver stamped.)"""
    d, msgs = _snubbed_report(div)
    assert d["n_turn_off_events_unresolved"] > 0
    assert any("'D1'" in m and "longer than t_rr" in m for m in msgs)
    assert d["n_turn_off_events_capped"] > 0, d
    assert any("'D1'" in m and "capped" in m for m in msgs)
    extra = d["n_turn_off_events"] - 200
    assert 0 < extra <= (d["n_turn_off_events_capped"]
                         + d["n_possible_chatter_gaps"]), d


@pytest.mark.parametrize("engine,div", [("pwl", 1000), ("pwl", 200),
                                        ("auto", 1000), ("dsed", 1000)])
@pytest.mark.parametrize("R_load", (20.0, 500.0))
def test_hard_and_natural_recoveries_agree_across_engines(R_load, engine,
                                                          div):
    """Unsnubbed boost, Q_rr = 100 nC (CCM: forced turn-offs at Q's
    edge; DCM: natural ones at the current zero). Measured: CCM
    6.935e-4 / 6.922e-4 / 6.934e-4 / 6.764e-4 J (pwl T/1000, T/200,
    auto, dsed), DCM 9.877e-4 / 9.863e-4 / 9.871e-4 / 9.868e-4 J."""
    b, r = _boost_run(0.7, R_load, engine, div)
    b_ref, r_ref = _boost_run(0.7, R_load, "pwl", 1000)
    spec = {"D1": {"Q_rr": 100e-9}}
    d = _entry(_report(b, r, [], diode_specs=spec)[0], "D1")
    d_ref = _entry(_report(b_ref, r_ref, [], diode_specs=spec)[0], "D1")
    assert d["n_turn_off_events"] == d_ref["n_turn_off_events"]
    assert d["E_sw_total"] == pytest.approx(d_ref["E_sw_total"], rel=0.03)


def test_recovery_voltage_reading_rules():
    """The V_R reading on synthetic traces: the fixed engine's
    alternation, a forced turn-off on a coarse grid, a natural one on a
    coarse grid, and a diode that stops inside its V_th band."""
    from pulsim.losses import _diode_turn_offs

    class Tr:
        def __init__(self, v, closed, g_on=1e3):
            self.v = np.asarray(v, dtype=float)
            self.closed = np.asarray(closed, dtype=bool)
            self.conducting = self.closed & (self.v > 0)
            self.current = np.where(self.closed, g_on, 1e-9) * self.v

    # (a) Alternation V_s ± A after the turn-off, fine grid: exact V_s.
    t = np.arange(40) * 10e-9
    v = np.full(40, 1e-3)
    v[10:] = -34.66 + 28.9 * (-1.0) ** np.arange(30)
    closed = np.arange(40) < 10
    (ev,) = _diode_turn_offs(Tr(v, closed), t, t_rr=100e-9)
    assert ev["V_R"] == pytest.approx(34.66, rel=1e-12)
    assert not ev["unresolved"]
    # (b) Forced (another bit flips on that step) and flat, 1 µs grid.
    t = np.arange(10) * 1e-6
    v = np.where(np.arange(10) < 4, 1e-3, -24.0)
    flip = np.zeros(10, dtype=bool)
    flip[4] = True
    (ev,) = _diode_turn_offs(Tr(v, np.arange(10) < 4), t, t_rr=100e-9,
                             other_flip=flip)
    assert ev["forced"] and not ev["unresolved"]
    assert ev["V_R"] == pytest.approx(24.0)
    # (c) The same samples with no other bit flipping (a natural
    #     turn-off): the grid hides how v_D rose — unresolved.
    (ev,) = _diode_turn_offs(Tr(v, np.arange(10) < 4), t, t_rr=100e-9,
                             other_flip=np.zeros(10, dtype=bool))
    assert ev["unresolved"]
    # (d) A slow (t − t_0)² rise after a natural zero on a 200 µs grid
    #     is not extrapolated into a forward voltage, and a diode that
    #     stops inside 0 < v_D < V_th blocks nothing.
    t = np.arange(10) * 200e-6
    v = np.where(np.arange(10) < 4, 1e-4,
                 -0.32 * (np.arange(10) - 3.0) ** 2)
    (ev,) = _diode_turn_offs(Tr(v, np.arange(10) < 4), t, t_rr=100e-9)
    assert 0.0 <= ev["V_R"] < 0.01
    v = np.where(np.arange(10) < 4, 1e-3, 2e-3)
    (ev,) = _diode_turn_offs(Tr(v, np.arange(10) < 4), t)
    assert ev["V_R"] == 0.0


def test_t_rr_must_be_a_positive_time():
    b, r = _boost_run(0.7, 20.0, "pwl", 200)
    for bad in (0.0, -1e-7, float("nan")):
        with pytest.raises(ValueError, match="t_rr"):
            p.device_loss_summary(b, r, diode_specs={
                "D1": {"Q_rr": 1e-7, "t_rr": bad}})


# ---------------------------------------------------------------------
# Review round 3 — integrals start each step from its right limit;
# recoveries are judged by their own charge
# ---------------------------------------------------------------------

class _Window:
    """The samples of ``r`` from ``t0`` on, as a result (with the
    record and, unless ``right_limits=False``, the DSED right limits)."""

    def __init__(self, r, t0, right_limits=True):
        t = np.asarray(r.times)
        k0 = int(np.searchsorted(t, t0))
        self.times = t[k0:]
        self.states = np.asarray(r.states)[k0:]
        self.switch_states = np.asarray(r.switch_states)[k0:]
        self.switch_states_unsolved = np.zeros(self.times.size, bool)
        self.engine_used = getattr(r, "engine_used", None)
        idx = getattr(r, "right_limit_index", None)
        if idx is not None and right_limits:
            m = idx >= k0
            self.right_limit_index = idx[m] - k0
            self.right_limit_states = r.right_limit_states[m]
            self.right_limit_switch_states = \
                r.right_limit_switch_states[m]


@pytest.mark.parametrize("R_load", (500.0, 20.0))
@pytest.mark.parametrize("engine", ("pwl", "auto", "dsed"))
def test_switched_branch_charge_closes_in_steady_state(engine, R_load):
    """Boost (DCM at 500 Ω, CCM at 20 Ω), steady-state window
    t >= 2 ms. D1's i_avg must equal the load charge C·Δv + ∫v/R
    (an independent KCL balance) and Q's i_avg the charge of i_L over
    the steps the record has Q ON. On DSED the turn-on sample is the
    left limit (i_D = 0), and a trapezoid from it spread each 0 -> I_L
    jump over the next 1 µs step: D1 i_avg read 0.02343 A against
    0.05345 A in DCM (-56 %) and -2.5 % in CCM, with no warning. All
    engines now agree to 1e-4."""
    b, r = _boost_run(0.7, R_load, engine)
    w = _Window(r, 2e-3)
    s, _, msgs = _report(b, w, [])
    assert not msgs, msgs
    t = w.times
    T = t[-1] - t[0]
    v_out = np.asarray(r.v("out"))[-t.size:]
    q_load = 47e-6 * (v_out[-1] - v_out[0]) + np.trapezoid(v_out / R_load, t)
    d = _entry(s, "D1")
    assert d["i_avg"] * T == pytest.approx(q_load, rel=1e-4)
    i_l = np.asarray(r.i("L"))[-t.size:]
    q_on = w.switch_states[1:, b.switch_index_of("Q")]
    q_q = np.dot(q_on * np.diff(t), 0.5 * (i_l[:-1] + i_l[1:]))
    # (pwl: its commutation step starts from the other stamp's sample,
    # measured +2.0e-3 in DCM.)
    assert _entry(s, "Q")["i_avg"] * T == pytest.approx(
        q_q, rel=3e-3 if engine == "pwl" else 1e-4)
    if engine == "dsed":
        bad = _entry(_report(b, _Window(r, 2e-3, right_limits=False),
                             [])[0], "D1")
        assert bad["i_avg"] * T < (0.6 if R_load == 500.0 else 0.99) \
            * q_load


@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_conduction_loss_agrees_across_engines_in_steady_state(R_load):
    """Each step is integrated as the line between its ends, and its
    square exactly, h·(a² + a·c + c²)/3. The trapezoid of the squared
    samples read the DCM diode's P_avg +16 % high on auto (one long step
    across its down-ramp) and DSED's -69 % (its jump drawn as a ramp).
    Measured now, t >= 2 ms, D1 P_avg: DCM 2.2797e-5 / 2.2766e-5 /
    2.2785e-5 W (pwl T/1000, auto, dsed), CCM 3.0867e-3 / 3.0860e-3 /
    3.0859e-3 W."""
    ref = {}
    for engine in ("pwl", "auto", "dsed"):
        b, r = _boost_run(0.7, R_load, engine)
        s, _, msgs = _report(b, _Window(r, 2e-3), [])
        assert not msgs, (engine, msgs)
        for name in ("D1", "Q"):
            e = _entry(s, name)
            if engine == "pwl":
                ref[name] = e
                continue
            assert e["P_avg"] == pytest.approx(ref[name]["P_avg"],
                                               rel=5e-3), (engine, name)
            assert e["i_rms"] == pytest.approx(ref[name]["i_rms"],
                                               rel=5e-3), (engine, name)


def _sync_buck(R_load: float):
    """48 V synchronous buck, add_mosfet HS / LS with body diodes,
    22 µH, 100 µF || R_load, 100 kHz, D = 0.4, 200 ns dead time."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 48.0)
    b.add_mosfet("HS", "vin", "sw")
    b.add_mosfet("LS", "sw", "gnd")
    b.add_inductor("L1", "sw", "out", 22e-6)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", R_load)
    T, D, TD = 1e-5, 0.4, 200e-9
    hs, ls = b.switch_index_of("HS"), b.switch_index_of("LS")

    def pwm(t):
        x = t % T
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(hs, x < D * T)
        m.set(ls, (D * T + TD) <= x < (T - TD))
        return m
    return b, pwm, T


@pytest.mark.parametrize("engine", ("pwl", "auto", "dsed"))
def test_sync_buck_pair_charge_is_the_inductor_charge(engine):
    """The high-side pair (HS minus its body diode) carries i_L over
    exactly the steps the record has it ON. DSED read that pair's charge
    +72 % high at RL = 50 Ω (1.250e-3 C against 7.27e-4 C): the charge
    of every turn-on jump went to the wrong device. Now pwl T/1000
    +0.66 % (its own commutation steps), auto -0.02 %, dsed 0.000 %."""
    b, pwm, T = _sync_buck(50.0)
    r = _run(b, 2e-3, engine, dt=T / 1000, switch_fn=pwm)
    s, _, msgs = _report(b, r, [])
    assert not msgs, msgs
    t = np.asarray(r.times)
    i_l = np.asarray(r.i("L1"))
    S = np.asarray(r.switch_states, dtype=bool)
    hs_on = S[1:, b.switch_index_of("HS")] | S[1:, b.switch_index_of(
        "HS_body")]
    ls_on = S[1:, b.switch_index_of("LS")] | S[1:, b.switch_index_of(
        "LS_body")]
    seg = np.diff(t) * 0.5 * (i_l[:-1] + i_l[1:])
    q_exact = seg[hs_on & ~ls_on].sum()
    e = {x["name"]: x for x in s}
    q_pair = (e["HS"]["i_avg"] - e["HS_body"]["i_avg"]) * (t[-1] - t[0])
    tol = 1e-2 if engine == "pwl" else 1e-3
    assert q_pair == pytest.approx(q_exact, rel=tol)


@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_dsed_switch_edge_energy_reads_the_edge_itself(R_load):
    """E_on scales with the load current just after the edge. On DSED
    the next sample is a whole 1 µs step into the on-time, so E_on read
    +4.7 % (DCM) / +3.7 % (CCM) against pwl; the current at the edge is
    the event's right limit. Measured now: DCM 2.1771e-4 / 2.1810e-4 J,
    CCM 4.4355e-4 / 4.4388e-4 J (dsed / pwl T/1000)."""
    spec = {"Q": {"E_on_ref": 1e-6, "E_off_ref": 1e-6, "V_ref": 24.0,
                  "I_ref": 2.0}}
    out = {}
    for engine in ("pwl", "dsed"):
        b, r = _boost_run(0.7, R_load, engine)
        out[engine] = _entry(_report(b, r, [], switch_specs=spec)[0], "Q")
    for key in ("E_sw_on_total", "E_sw_off_total"):
        assert out["dsed"][key] == pytest.approx(out["pwl"][key],
                                                 rel=5e-3), key


def test_recoveries_after_a_start_up_inrush_are_not_dropped():
    """400 V boost, 10 µH, 1 mF || 2 kΩ, D = 0.01 at 100 kHz, Q_rr =
    100 nC. The start-up inrush peaks at 4 kA; afterwards each cycle is
    a light-load DCM recovery of ~4 A (forward charge 0.14-0.21 µC,
    above Q_rr) against ~400 V. The old rule (I_F_peak >= 1e-3 ·
    i_peak of the whole run) dropped 237 of them: n_turn_off_events 61,
    f_sw_estimate 20.3 kHz, E_sw_total 2.46e-3 J, no warning. Judged
    by their own charge they count: 298 turn-offs, f_sw 99.3 kHz,
    1.195e-2 J."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 400.0)
    b.add_inductor("L1", "vin", "sw", 10e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=100.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("C1", "out", "gnd", 1e-3)
    b.add_resistor("RL", "out", "gnd", 2000.0)
    T = 1e-5
    q = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % T) < 0.01 * T)
        return m
    r = _run(b, 3e-3, "pwl", dt=T / 1000, switch_fn=pwm)
    s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 100e-9}})
    d = _entry(s, "D1")
    assert not msgs, msgs
    assert np.abs(np.asarray(r.i("L1"))).max() > 1e3
    assert d["n_turn_off_events_dropped"] == 0
    assert d["f_sw_estimate"] == pytest.approx(1e5, rel=0.02)
    # Every light-load recovery is ~Q_rr · 400 V = 4e-5 J.
    assert d["E_sw_total"] > 0.9 * 270 * 100e-9 * 400.0


def test_immaterial_reconductions_are_bounded_by_charge():
    """The RL half-wave's fixed-engine blips (single steps at ~36 µA,
    ~7e-11 C, against Q_rr = 1 µC) are still not turn-offs, and the
    energy they would add is reported and negligible."""
    _, d, _, msgs = _rl_run(0.7, "pwl")
    assert d["n_turn_off_events"] == 4
    assert d["n_turn_off_events_dropped"] > 0
    assert 0.0 <= d["E_sw_dropped"] < 1e-3 * d["E_sw_total"]
    assert not any("would add" in m for m in msgs), msgs


def test_rule_violation_names_the_actual_violation():
    """The warning used to say 'e.g. OFF while forward-biased' when the
    auto-engine bridge with line inductance held D2 / D4 ON while
    reverse-biased; it names what happened and that the whole circuit
    is suspect from there on. That bridge no longer violates the rule
    (the crossing detector reads the step start from the solve the
    bits were decided on, and re-decides a diode held on the forbidden
    side), so the wording is pinned on edited samples of a recorded
    run: one ON stamp reverse-biased, one OFF stamp forward-biased."""
    from pulsim._diode_state import diode_state_traces

    b = _bridge_with_line_l(0.7)
    r = _run(b, 0.08, "auto")
    _, _, msgs = _report(b, r, list(BRIDGE))
    assert not [m for m in msgs if _RULE in m], msgs

    b, r, k, a, kn = _rl_auto_landing()
    d = b.branch_index_of("D")
    bit = b.switch_index_of("D")
    rec = np.asarray(r.switch_states)
    t = np.asarray(r.times)
    ev_t = {ev.t_estimated for ev in r.commutation_events}
    plain = [j for j in range(1, t.size) if t[j] not in ev_t]
    j_on = next(j for j in plain if rec[j, bit])
    j_off = next(j for j in plain if not rec[j, bit])
    S = np.array(r.states, dtype=float)
    S[j_on, kn] = S[j_on, a] + 2.0             # ON, 2 V reverse
    S[j_off, kn] = S[j_off, a] - 5.0           # OFF, 5 V forward
    tr = diode_state_traces(b, _Edited(r, S))[d]
    assert tr.n_rule_violations == 2
    rule = [m for m in tr._messages if _RULE in m]
    assert len(rule) == 1, tr._messages
    assert "ON while reverse-biased on 1 samples" in rule[0], rule
    assert "OFF while forward-biased above V_th = 0.7 V" in rule[0], rule
    assert "every switched branch" in rule[0], rule

    # simulate() says the same at the end of the run, naming the diode.
    with pytest.warns(RuntimeWarning, match=r"held diode\(s\) 'D' "):
        p._trbdf2_warn_stats(_Edited(r, S), b, {}, "auto")


# ---------------------------------------------------------------------
# Repair round 3: curve materiality, snubber chatter, counts-only
# materiality, DSED junction temperature
# ---------------------------------------------------------------------

# A recovery-energy curve through the origin (a straight line, as the
# shipped IKW40N120T2 Erec table and PLECS tables starting at (0, 0)).
_CURVE_ORIGIN = {"E_rr_curve": [(1.0, 1e-6), (5.0, 5e-6)], "V_R_ref": 50.0,
                 "t_rr": 5e-6}


@pytest.mark.parametrize("V_th,engine", [(0.7, "pwl"), (2.0, "pwl"),
                                         (0.7, "auto"), (2.0, "auto"),
                                         (0.7, "dsed")])
def test_rl_half_wave_curve_through_the_origin_counts_real_turn_offs(
        V_th, engine):
    """With an E_rr_curve the charge a stretch had to reach was
    E_rr_curve(I_F) / V_R_ref, which goes to zero with the stretch's own
    current: every ~36 µA chatter blip counted. Measured before: 48
    turn-offs on pwl at V_th = 0.7 (f_sw 720 Hz), 21 at V_th = 2.0
    (315 Hz), 8 on auto (120 Hz), no warning. Materiality is now judged
    on the curve's own current axis: I_F_peak against 1e-3 of the
    smallest tabulated current."""
    b = _rl_half_wave(V_th)
    r = _run(b, 4 / 60, engine, dt=2e-6)
    s, _, msgs = _report(b, r, ["D"], diode_specs={"D": _CURVE_ORIGIN})
    e = _entry(s, "D")
    assert e["n_turn_off_events"] == 4, e
    assert e["f_sw_estimate"] == pytest.approx(60.0, rel=1e-3), e
    # The dropped blips are what the curve would charge them, and the
    # only warning allowed says so.
    assert all("would add" in m for m in msgs), msgs
    if engine == "pwl":
        assert e["n_turn_off_events_dropped"] > 0, e


def test_curve_materiality_uses_the_tabulated_current_axis():
    """Unit check of the rule: a curve with no positive current falls
    back to its (constant) charge."""
    from pulsim.losses import _diode_switching_loss
    t = np.linspace(0.0, 1e-3, 11)
    ev = [{"t": 5e-4, "V_R": 10.0, "I_F": 0.0, "I_F_peak": 2.0,
           "Q_F": 1e-6},
          {"t": 6e-4, "V_R": 10.0, "I_F": 0.0, "I_F_peak": 5e-4,
           "Q_F": 1e-11}]
    curve = {"E_rr_curve": [(0.0, 0.0), (1.0, 1e-6)], "V_R_ref": 10.0}
    out = _diode_switching_loss(ev, t, np.zeros_like(t), curve)
    assert out["n_turn_off_events"] == 1
    assert out["n_turn_off_events_dropped"] == 1
    flat = {"E_rr_curve": [(0.0, 1e-6)], "V_R_ref": 10.0}
    out = _diode_switching_loss(ev, t, np.zeros_like(t), flat)
    # Constant 1e-7 C asked: 1e-6 C passes, 1e-11 C does not.
    assert out["n_turn_off_events"] == 1


def _snubbed_boost_100k(V_th: float):
    """The physics review's circuit: 24 V, 220 µH, Q1 g_on = 50 S at
    100 kHz / D = 0.5, D1 with 100 nF directly across it, 100 µF ||
    30 Ω."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    b.add_inductor("L1", "vin", "sw", 220e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=50.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=V_th)
    b.add_capacitor("Cs", "sw", "out", 100e-9)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 30.0)
    T = 1e-5
    q1 = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q1, (t % T) < 0.5 * T)
        return m
    return b, pwm, T


@pytest.mark.parametrize("div", (400, 2000))
def test_snubber_chatter_is_counted_as_recorded_and_warned(div):
    """Cs = 100 nF across a V_th = 0.7 diode on the fixed engine: the
    trapezoidal rule reflects each ON step from +V_th to about -V_th,
    so the record reads ON OFF OFF ... through every conduction: 4580
    recorded turn-offs at dt = T/400 and 15887 at T/2000 (f_sw 15x to
    53x the PWM). Round 3 merged those gaps; that also merged real
    source-commutated turn-offs into a small reverse voltage
    (test_source_commutated_turn_off_into_a_small_voltage_is_counted),
    so the record is now reported as stamped and a warning says the
    gaps may be chatter and which numbers it inflates. At V_th = 0
    nothing chatters and nothing warns."""
    for V_th in (0.7, 0.0):
        b, pwm, T = _snubbed_boost_100k(V_th)
        r = _run(b, 3e-3, "pwl", dt=T / div, switch_fn=pwm)
        s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 50e-9}})
        e = _entry(s, "D1")
        chatter = [m for m in msgs if _CHATTER in m]
        n_rec = _recorded_turn_offs(b, r, "D1")
        assert (e["n_turn_off_events"] + e["n_turn_off_events_dropped"]
                == n_rec), (e, n_rec)
        if V_th > 0:
            assert e["n_possible_chatter_gaps"] > 1000, e
            assert e["f_sw_estimate"] > 10.0 / T, e
            assert len(chatter) == 1, msgs
            for word in ("n_turn_off_events", "f_sw_estimate",
                         "E_sw_total", "engine='trbdf2'", "WITHOUT dt",
                         "treat a disagreement as unresolved"):
                assert word in chatter[0], (word, chatter[0])
        else:
            assert e["f_sw_estimate"] == pytest.approx(1.0 / T, rel=0.02)
            assert not chatter, msgs


def _recorded_turn_offs(b, r, name):
    """Conducting-to-blocking transitions of the recorded stamp."""
    from pulsim._diode_state import diode_state_trace
    tr = diode_state_trace(b, r, b.branch_index_of(name), warn=False)
    c = np.asarray(tr.conducting, dtype=bool)
    return int(np.count_nonzero(c[:-1] & ~c[1:]))


def test_forced_commutation_into_a_small_voltage_is_still_a_turn_off():
    """The first cycles of a boost whose output is still below V_th: Q
    turning on leaves D1 blocking only millivolts. On engine='auto'
    every commutation is a turn-off and the chatter warning stays
    quiet (its TR-BDF2 steps do not reflect the snubber mode)."""
    b, pwm, T = _snubbed_boost_100k(0.7)
    r = _run(b, 3e-3, "auto", switch_fn=pwm)
    s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 50e-9}})
    e = _entry(s, "D1")
    assert e["f_sw_estimate"] == pytest.approx(1.0 / T, rel=0.02), e
    assert not any(_CHATTER in m for m in msgs), msgs


def _pulsed_charger():
    """100 kHz pulse 11.5 / 14.4 V through 0.5 Ω and a V_th = 0.7
    diode into a 12 V battery: the diode conducts on the high level and
    blocks 0.5 V on the low one, with no other switch in the circuit."""
    b = p.CircuitBuilder()
    b.add_pulse_voltage_source("V", "s", "gnd", 11.5, 14.4, 0.0, 5e-6, 1e-5)
    b.add_resistor("R", "s", "a", 0.5)
    b.add_diode("D", "a", "k", 1e3, 1e-9, V_th=0.7)
    b.add_voltage_source("B", "k", "gnd", 12.0)
    return b


@pytest.mark.parametrize("engine,dt", [("pwl", 1e-6), ("pwl", 1e-7),
                                       ("auto", None)])
def test_source_commutated_turn_off_into_a_small_voltage_is_counted(
        engine, dt):
    """Round 3 merged every blocking gap that stayed above -2·V_th and
    that no other switch forced: in a circuit with no other switch,
    real turn-offs into a small reverse voltage too. The pulsed
    charger's 100 turn-offs read 1 (f_sw 1 kHz on 100 kHz, E_sw 100x
    low) on every engine, with a chatter warning. Every recorded
    stretch now counts. The chatter detector does not take them for
    chatter either: the conduction on each side lasts 5 steps at
    dt = 1 µs (50 at 0.1 µs), where a reflected ON step lasts one. A
    detector keyed on the gap length (at most 32 steps) warned on the
    1 µs run."""
    b = _pulsed_charger()
    r = _run(b, 1e-3, engine, dt=dt)
    s, _, msgs = _report(b, r, [], diode_specs={"D": {"Q_rr": 50e-9}})
    e = _entry(s, "D")
    assert e["n_turn_off_events"] == 100, e
    assert e["f_sw_estimate"] == pytest.approx(1e5, rel=1e-6), e
    assert e["E_sw_total"] == pytest.approx(100 * 50e-9 * 0.5, rel=1e-6)
    assert e["n_possible_chatter_gaps"] == 0, e
    assert not any(_CHATTER in m for m in msgs), msgs


def test_paired_diode_chatter_warns_and_reports_the_record():
    """Two in-phase boost phases, each D with 100 nF across it, on the
    fixed engine: both diodes chatter in the same steps. Round 3's
    'no other mask bit flips' guard counted the other diode's bit, so
    no gap was ever merged and nothing warned (601 turn-offs each on
    this 1 ms run, f_sw 601 kHz on 100 kHz). Now each diode warns (502
    short gaps each), and every count is the recorded one."""
    T = 1e-5
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 24.0)
    for ph in "ab":
        b.add_inductor("L" + ph, "vin", "sw" + ph, 220e-6)
        b.add_switch("Q" + ph, "sw" + ph, "gnd", g_on=50.0, g_off=1e-9)
        b.add_diode("D" + ph, "sw" + ph, "out", 1e3, 1e-9, V_th=0.7)
        b.add_capacitor("Cs" + ph, "sw" + ph, "out", 100e-9)
    b.add_capacitor("C1", "out", "gnd", 100e-6)
    b.add_resistor("RL", "out", "gnd", 15.0)
    qa, qb = b.switch_index_of("Qa"), b.switch_index_of("Qb")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        on = (t % T) < 0.5 * T
        m.set(qa, on)
        m.set(qb, on)
        return m
    r = _run(b, 1e-3, "pwl", dt=T / 400, switch_fn=pwm)
    spec = {"Q_rr": 50e-9}
    s, _, msgs = _report(b, r, [], diode_specs={"Da": spec, "Db": spec})
    for n in ("Da", "Db"):
        e = _entry(s, n)
        assert (e["n_turn_off_events"] + e["n_turn_off_events_dropped"]
                == _recorded_turn_offs(b, r, n)), e
        assert e["n_possible_chatter_gaps"] > 0.5 * e["n_turn_off_events"]
        assert sum(1 for m in msgs if _CHATTER in m and f"{n!r}" in m) == 1


def _transformer_rectifier(C):
    """50 Hz, 100 V through 0.1 Ω and a 5:1 ideal transformer into a
    V_th = 0.7 diode, C and 20 Ω: tau = C·(1/g_on + R_loop) = 0.5 µs
    at C = 100 µF."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("V", "s", "gnd", 0.0, 100.0, 50.0)
    b.add_resistor("Rp", "s", "p", 0.1)
    b.add_ideal_transformer("T1", "p", "gnd", "sec", "gnd", 0.2)
    b.add_diode("D", "sec", "k", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("C", "k", "gnd", C)
    b.add_resistor("RL", "k", "gnd", 20.0)
    return b


@pytest.mark.parametrize("dt,n_gaps", [(1e-6, 0), (2e-6, 355)])
def test_rectifier_chatter_with_long_gaps_is_reported(dt, n_gaps):
    """A capacitor-input rectifier, 100 µF, 0.1 s (5 real turn-offs).
    At dt = 2 µs = 4·tau the trapezoidal rule reflects each ON step and
    the capacitor takes 40 to 187 steps to recharge to V_th: 360
    recorded turn-offs, E_sw 70x the real one. A detector capped at 32
    gap steps reported 0 gaps and stayed silent. Each gap sits between
    1-step conducting stretches, which is what the detector now keys
    on; at dt = 1 µs = 2·tau there is no chatter and nothing warns."""
    b = _transformer_rectifier(100e-6)
    r = _run(b, 0.1, "pwl", dt=dt)
    s, _, msgs = _report(b, r, [], diode_specs={"D": {"Q_rr": 1e-6}})
    e = _entry(s, "D")
    chatter = [m for m in msgs if _CHATTER in m]
    assert e["n_turn_off_events"] + e["n_turn_off_events_dropped"] == (
        _recorded_turn_offs(b, r, "D"))
    assert e["n_possible_chatter_gaps"] == n_gaps, e
    if n_gaps:
        assert e["n_turn_off_events"] > 300, e
        assert len(chatter) == 1, msgs
    else:
        assert e["n_turn_off_events"] == 5, e
        assert not chatter, msgs


@pytest.mark.parametrize("spec", ({}, {"t_rr": 1e-7}))
def test_counts_only_recoveries_after_an_inrush_are_not_dropped(spec):
    """With no recovery charge in the spec the count was judged against
    1e-3 of the run's peak current, 4 A after a 4 kA inrush: 237 of
    the boost's 3-4 A light-load recoveries dropped, f_sw 20.3 kHz on
    100 kHz, no warning. The reference is now the median peak of the
    stretches that pass the run-peak cut (never above the run's peak,
    so nothing the old rule kept is dropped)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 400.0)
    b.add_inductor("L1", "vin", "sw", 10e-6)
    b.add_switch("Q1", "sw", "gnd", g_on=100.0, g_off=1e-9)
    b.add_diode("D1", "sw", "out", 1e3, 1e-9, V_th=0.7)
    b.add_capacitor("C1", "out", "gnd", 1e-3)
    b.add_resistor("RL", "out", "gnd", 2000.0)
    T = 1e-5
    q = b.switch_index_of("Q1")

    def pwm(t):
        m = p.SwitchStateMask(b.graph.num_switches)
        m.set(q, (t % T) < 0.01 * T)
        return m
    r = _run(b, 3e-3, "pwl", dt=T / 1000, switch_fn=pwm)
    s, _, msgs = _report(b, r, [], diode_specs={"D1": spec})
    d = _entry(s, "D1")
    assert not msgs, msgs
    assert d["n_turn_off_events_dropped"] == 0, d
    assert d["f_sw_estimate"] == pytest.approx(1e5, rel=0.02), d
    # The RL half-wave blips are still dropped without a spec.
    b2 = _rl_half_wave(0.7)
    r2 = _run(b2, 4 / 60, "pwl", dt=2e-6)
    s2, _, _ = _report(b2, r2, [], diode_specs={"D": dict(spec)})
    assert _entry(s2, "D")["n_turn_off_events"] == 4


def test_dsed_trace_without_a_step_start_jump_is_on_result_times():
    """A DSED device whose step-start power equals its stored sample
    (a resistor on the output capacitor) had no samples inserted, and
    took the pwl/trbdf2 branch: its T_j_trace was compute_temperature's
    uniform resample, len(times) samples only because the resample is
    clipped to at least that many, NOT at result.times (index 500 at
    485.9 µs against t[500] = 497.8 µs; 36 K off an IIR sampled at
    result.times on a 907 K ramp). Every DSED device is sampled back."""
    b, pwm, T = _boost(0.7, 20.0)
    r = _run(b, 1e-3, "dsed", switch_fn=pwm)
    t = np.asarray(r.times)
    R_th, tau = 10.0, 2e-5
    st = [p.FosterStage(R_th_K_per_W=R_th, tau_s=tau)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        th = p.device_thermal_summary(b, r, switch_fn=pwm,
                                      thermal_specs={"RL": {"stages": st}})
    tj = np.asarray(th[0]["T_j_trace"])
    assert tj.shape == t.shape
    # Reference: the same IIR on a fine uniform grid of v_out²/R.
    tf = np.linspace(t[0], t[-1], 100_001)
    P = np.interp(tf, t, np.asarray(r.v("out"))) ** 2 / 20.0
    a = np.exp(-(tf[1] - tf[0]) / tau)
    out = np.empty_like(P)
    acc = 0.0
    for k, x in enumerate(P):
        acc = a * acc + R_th * (1.0 - a) * x
        out[k] = acc
    ref = 25.0 + np.interp(t, tf, out)
    rise = float(ref.max() - ref.min())
    assert rise > 500.0
    assert np.abs(tj - ref).max() <= 0.01 * rise


@pytest.mark.parametrize("R_load", (500.0, 20.0))
def test_dsed_junction_temperature_follows_the_step_start_power(R_load):
    """T_j was built from the stored DSED samples, the left limits,
    while P_cond_avg integrates each step from its right limit: on the
    boost (R_th 1000 K/W, tau 0.1 ms) (T_j_avg - 25)/R_th / P_cond_avg
    read 0.989 for D1 and 1.010 for Q1 in DCM, 0.980 for D1 in CCM.
    The thermal trace now carries the step-start power too; DSED then
    agrees with the fixed engine (whose own residual, 0.6 % in CCM, is
    the T_j_avg mean over the start-up rise)."""
    st = [p.FosterStage(R_th_K_per_W=1000.0, tau_s=1e-4)]
    ratios = {}
    for engine in ("pwl", "dsed"):
        b, pwm, T = _boost(0.7, R_load)
        r = _run(b, 3e-3, engine, dt=T / 1000, switch_fn=pwm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            th = p.device_thermal_summary(
                b, r, switch_fn=pwm,
                thermal_specs={"D1": {"stages": st}, "Q": {"stages": st}})
        t = np.asarray(r.times)
        for e in th:
            ratios[(engine, e["name"])] = (
                (e["T_j_avg"] - 25.0) / 1000.0 / e["P_cond_avg"])
            # One T_j per stored sample (DSED computes on the grid
            # with the step-start power inserted and samples back), and
            # T_j_avg is a time average: a plain mean over DSED's
            # event-clustered samples is not.
            tj = np.asarray(e["T_j_trace"])
            assert tj.shape == t.shape, (engine, tj.shape, t.shape)
            tw = np.trapezoid(tj, t) / (t[-1] - t[0])
            assert e["T_j_avg"] - 25.0 == pytest.approx(
                tw - 25.0, rel=2e-3), (engine, e["name"])
    for name in ("D1", "Q"):
        assert ratios[("dsed", name)] == pytest.approx(
            ratios[("pwl", name)], abs=5e-3), ratios
