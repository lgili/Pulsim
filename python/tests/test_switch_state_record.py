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


# Trapezoid over the stored grid: exact bookkeeping on the fixed
# engine's trapezoidal companions (measured ~3e-12); the adaptive grid
# integrates the same currents to ~1e-4. DSED stores the LEFT limit at
# an event sample (the stamped-mask convention), so a current that
# jumps at a V_th turn-on is drawn as a ramp over the next step: an
# O(h) trapezoid error, -4.2e-3 at the default 1 µs step on the
# transformer rectifier (it halves with the step; see
# test_dsed_charge_error_is_the_one_step_jump_term).
TOL = {"pwl": 1e-9, "auto": 1e-3, "dsed": 6e-3}


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
    b, pwm, T = _snubbed_boost(V_th, Cs)
    r = _run(b, 3e-3, "pwl", dt=T / 400, switch_fn=pwm)
    s, cur, msgs = _report(b, r, ["D1"])
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
    q_d = np.trapezoid(cur["D"], np.asarray(r.times))
    q_phys = _q_load(r, "k", 1000e-6, 20.0)
    assert q_phys > 0.05
    assert abs(q_d - q_phys) <= TOL[engine] * q_phys, (q_d, q_phys)
    e = _entry(s, "D")
    assert 0.2 < e["duty_conducting"] < 0.23, e


def test_dsed_charge_error_is_the_one_step_jump_term():
    """The DSED charge residual on the transformer rectifier is the
    trapezoid drawing each V_th turn-on's current jump as a ramp over
    one step, so it is first order in the step and has the sign of a
    missing half-step of current (measured -4.22e-3, -2.15e-3,
    -0.87e-3 at a median step of 1 µs, 0.5 µs, 0.2 µs)."""
    errs = []
    for kw in ({}, {"dt": 5e-6}):
        b = _xfmr_rectifier(0.7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = p.simulate(b, 0.1, engine="dsed", **kw)
        q_d = np.trapezoid(np.asarray(r.i("D")), np.asarray(r.times))
        q_phys = _q_load(r, "k", 1000e-6, 20.0)
        errs.append((q_d - q_phys) / q_phys)
    assert errs[0] < 0 and errs[1] < 0, errs
    assert 0.35 < errs[1] / errs[0] < 0.65, errs


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
def _ref_run(V_th: float, engine: str):
    b = _reference_bridge(V_th)
    r = _run(b, 0.06, engine, dt=1e-6)
    return _report(b, r, BRIDGE) + (r,)


def test_reference_bridge_agrees_across_engines_at_v_th_0_7():
    """pwl vs auto at the helper defaults (V_th = 0.7). Measured:
    D1 rms 10.246 / 10.263 A, duty 0.1030 / 0.1030, P_avg 0.1050 /
    0.1054 W; D2 rms 2.937 / 3.006 A, duty 0.0292 / 0.0293, P_avg
    0.00866 / 0.00907 W. D2's residual is the V_th turn-on spike (the
    diode voltage collapses from 0.7 V into a ~3 mΩ loop with no
    inductance: 345 A on pwl at dt = 1 µs, 458 A on auto) — about a
    fifth of D2's small loss, resolved differently by the two grids.
    Peaks are therefore not compared."""
    s_p, _, m_p, _ = _ref_run(0.7, "pwl")
    s_a, _, m_a, _ = _ref_run(0.7, "auto")
    assert not m_p and not m_a, (m_p, m_a)
    for name in BRIDGE:
        a, b_ = _entry(s_p, name), _entry(s_a, name)
        assert b_["i_rms"] == pytest.approx(a["i_rms"], rel=0.03), name
        assert abs(b_["duty_conducting"] - a["duty_conducting"]) < 1e-3
        assert b_["P_avg"] == pytest.approx(a["P_avg"], rel=0.06), name
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
    capped at their own forward charge. Both used to pass silently;
    each now warns and names the diode."""
    d, msgs = _snubbed_report(div)
    assert d["n_turn_off_events_unresolved"] > 0
    assert any("'D1'" in m and "longer than t_rr" in m for m in msgs)
    assert d["n_turn_off_events_capped"] > 0
    assert any("'D1'" in m and "capped" in m for m in msgs)


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
