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
import warnings

import numpy as np
import pytest

import pulsim as p

ENGINES = ("pwl", "auto")
_RULE = "commutation rule forbids"
_CHATTER = "That is numerical chatter"


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
def test_seeded_row_is_read_with_the_first_solve_mask(engine):
    """initial_state carries no diode bits: row 0 records the diode
    OFF while the seeded state carries 1.259 A through it (its own
    solve had it ON). Read with the carried row, result.i() missed the
    inductor current by 1.26 A at k = 0 (KCL residual 0.197 relative,
    2e-12 everywhere else). The consumers read an unsolved row with the
    mask of the first solve after it, so KCL closes at every sample."""
    b, r = _rl_seeded(engine)
    bit = b.switch_index_of("D")
    assert r.switch_states_unsolved[0]
    assert not r.switch_states[0, bit] and r.switch_states[1, bit]
    i_d = np.asarray(r.i("D"))
    i_l = np.asarray(r.i("L"))
    assert i_l[0] > 1.0
    assert np.abs(i_d - i_l).max() <= 1e-9 * np.abs(i_l).max()
    s, cur, msgs = _report(b, r, ["D"])
    assert not msgs, msgs
    assert _entry(s, "D")["i_peak"] == pytest.approx(
        float(np.abs(i_l).max()), rel=1e-9)


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
        assert _entry(s, "D1")["n_chatter_gaps"] > 100
    else:
        # (Gaps of immaterial current at v_D ≈ 0 may still be merged;
        # they cover too little time to warn about.)
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

    The 294 A itself is in the auto trajectory (the pwl run peaks at
    65 A): engine='auto' holds BR__D2/BR__D4 ON against the commutation
    rule for 25 samples and discharges the DC capacitor through them —
    an engine defect, which the report names instead of hiding."""
    b = _bridge_with_line_l(0.7)
    r = _run(b, 0.08, engine, dt=1e-6)
    s, cur, msgs = _report(b, r, BRIDGE)
    assert all(_RULE in m for m in msgs), msgs
    if engine == "pwl":
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
    with pytest.warns(RuntimeWarning, match=r"diode 'D'"):
        p.SimulationResult.i(bare, "D")


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
    (235 "turn-offs" at T/50 on a 200-cycle run, f_sw 117.5 kHz) are
    either capped at their own forward charge (and warned about) or,
    where the diode never blocked more than 2·V_th between them, merged
    into the conduction around them as chatter (206 at T/50, 33 merged;
    208 at T/20 with 8 capped). Both used to pass silently."""
    d, msgs = _snubbed_report(div)
    assert d["n_turn_off_events_unresolved"] > 0
    assert any("'D1'" in m and "longer than t_rr" in m for m in msgs)
    assert d["n_turn_off_events_capped"] > 0 or d["n_chatter_gaps"] > 0, d
    if d["n_turn_off_events_capped"] > 0:
        assert any("'D1'" in m and "capped" in m for m in msgs)
    assert d["n_turn_off_events"] <= 210, d


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
    """The auto-engine bridge with line inductance holds D2 / D4 ON
    while reverse-biased. The warning said 'e.g. OFF while
    forward-biased', the opposite case; it now names what happened and
    that the whole circuit is suspect from there on."""
    b = _bridge_with_line_l(0.7)
    r = _run(b, 0.08, "auto", dt=1e-6)
    _, _, msgs = _report(b, r, list(BRIDGE))
    rule = [m for m in msgs if _RULE in m]
    assert rule, msgs
    for m in rule:
        assert "ON while reverse-biased" in m or "OFF while" in m, m
        assert "every switched branch" in m, m
    assert any("ON while reverse-biased" in m for m in rule), rule


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
def test_snubber_chatter_is_not_counted_as_turn_offs(div):
    """Cs = 100 nF across a V_th = 0.7 diode on the fixed engine: the
    trapezoidal rule reflects each ON step from +V_th to about -V_th,
    so the record reads ON OFF OFF ... through every conduction. Each
    gap followed more than Q_rr, so each was charged: 4580 turn-offs
    at dt = T/400 and 15887 at T/2000 on a 100 kHz version (f_sw 15x
    to 53x the PWM), with no warning, and 2857 of them never reverse-
    biased the diode. A gap whose step averages never fall below -2·V_th
    and that no other switch forced is now chatter inside one
    conduction: the count matches V_th = 0 (one per cycle; 299 and
    300 at V_th = 0.7), and a warning names the chatter."""
    counts = {}
    for V_th in (0.7, 0.0):
        b, pwm, T = _snubbed_boost_100k(V_th)
        r = _run(b, 3e-3, "pwl", dt=T / div, switch_fn=pwm)
        s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 50e-9}})
        e = _entry(s, "D1")
        counts[V_th] = e["n_turn_off_events"]
        assert e["f_sw_estimate"] == pytest.approx(1.0 / T, rel=0.02), e
        if V_th > 0:
            assert e["n_chatter_gaps"] > 1000, e
            assert any(_CHATTER in m for m in msgs), msgs
        else:
            assert not any(_CHATTER in m for m in msgs), msgs
    assert abs(counts[0.7] - counts[0.0]) <= 2, counts


def test_forced_commutation_into_a_small_voltage_is_still_a_turn_off():
    """The first cycles of a boost whose output is still below V_th: Q
    turning on leaves D1 blocking only millivolts, inside the band the
    chatter rule forgives. A commutation another switch forced is a
    turn-off all the same (engine='auto' merged 6 of them as chatter,
    and warned, before the other-switch condition)."""
    b, pwm, T = _snubbed_boost_100k(0.7)
    r = _run(b, 3e-3, "auto", switch_fn=pwm)
    s, _, msgs = _report(b, r, [], diode_specs={"D1": {"Q_rr": 50e-9}})
    e = _entry(s, "D1")
    assert e["f_sw_estimate"] == pytest.approx(1.0 / T, rel=0.02), e
    assert not any(_CHATTER in m for m in msgs), msgs


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
        for e in th:
            ratios[(engine, e["name"])] = (
                (e["T_j_avg"] - 25.0) / 1000.0 / e["P_cond_avg"])
    for name in ("D1", "Q"):
        assert ratios[("dsed", name)] == pytest.approx(
            ratios[("pwl", name)], abs=5e-3), ratios
