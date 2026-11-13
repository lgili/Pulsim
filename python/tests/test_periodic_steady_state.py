"""Periodic steady state by Newton shooting on the true state.

`add-periodic-steady-state-shooting`, slice 1 (the dsed exact path).
Reference numbers are the change's prototypes (`probes/`), ported to
`_pss_reference.py`; the DCM orbit is a 50-digit value. The buck is
the audit's `numerics/p2_buck.py`.

Three groups:

* characterisation of today's fixed-engine routines on the same two
  bucks. The DCM cases are the ones the change flips: `steady_state`
  without `dt` (this slice) and `run_periodic_shooting` (task 5.3,
  after Phase 2) — until then they pin the defect;
* the numpy reference, so the constants below are reproducible;
* the solver: orbit, convergence, monodromy, small-signal model,
  refusals, and the delegation of `steady_state` /
  `frequency_response`.

Efficiency is asserted by counters (period runs), never wall clock.
"""

import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.periodic import run_periodic_shooting

from . import _pss_reference as ref

T = ref.T


def _buck(R):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", ref.VIN)
    b.add_switch("S", "vin", "sw", ref.GON, ref.GOFF)
    b.add_diode("D1", "gnd", "sw", ref.GON, ref.GOFF)
    b.add_inductor("L", "sw", "vout", 100e-6)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    return b


def _pwm(b, D):
    return p.make_pwm_switch_fn(1 / T, D, b.switch_index_of("S"),
                                b.graph.num_switches)


def _grid_gate(b, D):
    """The audit's Python gate (fixed engine, dt = T/100)."""
    n, k = b.graph.num_switches, b.switch_index_of("S")

    def fn(t):
        m = p.SwitchStateMask(n)
        ph = t / T
        m.set(k, bool(ph - np.floor(ph + 1e-9) < D - 1e-9))
        return m

    return fn


def _iv(res):
    """(i_L, v_C) of a result's x0."""
    lab = res.state_labels
    return res.x0[lab.index("i(L)")], res.x0[lab.index("v(vout)")]


@pytest.fixture(scope="module")
def dcm():
    b = _buck(50.0)
    return b, p.periodic_steady_state(b, period=T,
                                      switch_fn=_pwm(b, 0.3))


# ---------------------------------------------------------------------
# 1. today's routines (fixed engine, dt = T/100)
# ---------------------------------------------------------------------

def test_legacy_steady_state_ccm_passes():
    b = _buck(1.0)
    ss = p.steady_state(b, period=T, dt=T / 100,
                        switch_fn=_grid_gate(b, 0.5))
    res = p.simulate(b, t_end=T + 1e-12, dt=T / 100,
                     switch_fn=_grid_gate(b, 0.5),
                     resume_from=ss.snapshot)
    # The fixed engine's own orbit: 2.6e-5 from the exact one at T/100.
    v0 = float(np.asarray(res.v("vout"))[0])
    assert v0 == pytest.approx(ref.CCM_ORBIT[1], rel=1e-4)


def test_legacy_steady_state_dcm_raises():
    """Flipped by `periodic_steady_state` (and `steady_state` without
    `dt`); the fixed-engine path keeps raising until Phase 2."""
    b = _buck(50.0)
    with pytest.raises(ValueError, match="no periodic orbit found"):
        p.steady_state(b, period=T, dt=T / 100,
                       switch_fn=_grid_gate(b, 0.3))


def test_legacy_shooting_dcm_is_biased_by_the_history_reset():
    """`run_periodic_shooting` restarts each period from the MNA
    vector and converges to 8.8998 V, -1.07 %. Task 5.3 flips it."""
    b = _buck(50.0)
    r = run_periodic_shooting(b, t_period=T, dt=T / 100,
                              switch_fn=_grid_gate(b, 0.3))
    assert r.converged
    v_c = float(r.x_steady_state[2])
    assert v_c == pytest.approx(8.8998, abs=2e-4)
    assert v_c / ref.DCM_ORBIT[1] - 1 == pytest.approx(-0.0107,
                                                       abs=2e-4)


# ---------------------------------------------------------------------
# 2. the numpy reference reproduces the change's table
# ---------------------------------------------------------------------

def test_reference_dcm_orbit_multipliers_and_turn_off():
    x, M, hist = ref.shoot(ref.buck_period, ref.DCM, [0.0, 0.0])
    assert x[0] == pytest.approx(ref.DCM_ORBIT[0], abs=1e-15)
    assert x[1] == pytest.approx(ref.DCM_ORBIT[1], rel=1e-10)
    assert len(hist) == 5                      # map evaluations
    mu = np.sort(np.abs(np.linalg.eigvals(M)))
    assert mu[0] < 1e-9
    assert mu[1] == pytest.approx(ref.DCM_MULTIPLIERS[1], abs=1e-6)
    tau = ref.buck_period(ref.DCM, x)[2]
    assert tau == pytest.approx(ref.DCM_TURN_OFF, abs=1e-9)


def test_reference_ccm_orbit():
    x, M, _ = ref.shoot(ref.buck_period, ref.CCM, [0.0, 0.0])
    assert np.allclose(x, ref.CCM_ORBIT, rtol=0, atol=1e-9)
    mu = np.linalg.eigvals(M)
    assert np.abs(mu).max() == pytest.approx(0.899035, abs=1e-6)


def test_reference_pcm_needs_saltation():
    x, M, _ = ref.shoot(ref.pcm_period, ref.PCM, [6.0, 12.0])
    assert np.allclose(x, ref.PCM_ORBIT, rtol=0, atol=1e-9)
    mu = np.sort(np.linalg.eigvals(M).real)
    assert np.allclose(mu, ref.PCM_MULTIPLIERS, atol=1e-5)
    no = np.linalg.eigvals(ref.pcm_period(ref.PCM, x, salt=False)[1])
    assert np.allclose(sorted(no, key=lambda z: z.imag),
                       sorted(ref.PCM_NO_SALTATION, key=lambda z: z.imag),
                       atol=1e-4)


# ---------------------------------------------------------------------
# 3. the solver
# ---------------------------------------------------------------------

def test_dcm_orbit_from_zero(dcm):
    _b, r = dcm
    i_l, v_c = _iv(r)
    assert v_c == pytest.approx(ref.DCM_ORBIT[1], rel=1e-8)
    assert i_l == pytest.approx(ref.DCM_ORBIT[0], abs=1e-12)
    # prototype: 5 map evaluations, plus the verification period
    assert r.n_period_runs <= 8
    assert r.residual < 1e-9
    off = [tau for name, tau in r.pattern if "turns OFF" in name]
    assert len(off) == 1
    assert off[0] == pytest.approx(ref.DCM_TURN_OFF, abs=1e-9)


def test_dcm_iteration_passes_through_a_ccm_pattern(dcm):
    _b, r = dcm
    has_off = ["D1 (branch #2) turns OFF" in pat
               for pat in r.pattern_history]
    assert not all(has_off)       # some iterate had no diode turn-off
    assert has_off[-1]            # the orbit is DCM


def test_dcm_convergence_is_quadratic(dcm):
    _b, r = dcm
    h = r.residual_history
    # after the pattern has settled (prototype 2.2e-4, 3.0e-7, 5.7e-13)
    tail = h[-3:]
    assert tail[-1] < 1e-11
    for a, b in zip(tail[:-1], tail[1:]):
        assert b <= 1e3 * a * a


def test_dcm_from_a_ccm_pattern_start():
    b = _buck(50.0)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3),
                                start_from=[12.0, 11.7])
    assert "turns OFF" not in " ".join(r.pattern_history[0])
    assert _iv(r)[1] == pytest.approx(ref.DCM_ORBIT[1], rel=1e-8)
    assert r.n_period_runs <= 12


@pytest.mark.parametrize("R, D", [(50.0, 0.3), (1.0, 0.5)])
def test_orbit_equals_a_4000_period_transient(R, D):
    """Brute force: 4000 periods of the same exact stepper from zero,
    one period per run. (One continuous 4000-period `simulate(engine=
    'dsed')` of the DCM buck stops at t = 0.996 ms on the engine's
    diode-chatter guard; a period run starts each idle phase fresh.)"""
    b = _buck(R)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, D))
    s = r._ctx
    x = np.zeros(len(r.x0))
    for _ in range(4000):
        x = s.run(x).x_end
    assert np.abs(x - r.x0).max() < 1e-9


def test_ccm_orbit_and_cost():
    b = _buck(1.0)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.5))
    i_l, v_c = _iv(r)
    assert i_l == pytest.approx(ref.CCM_ORBIT[0], abs=1e-9)
    assert v_c == pytest.approx(ref.CCM_ORBIT[1], abs=1e-9)
    assert r.n_period_runs <= 4
    assert r.stable
    assert r.floquet_radius == pytest.approx(0.899035, abs=1e-6)


def test_resume_from_the_orbit(dcm):
    b, r = dcm
    one = p.simulate(b, t_end=T, dt=T, engine="dsed",
                     switch_fn=_pwm(b, 0.3), initial_state=r.snapshot)
    assert np.abs(np.asarray(one.states_reduced[-1]) - r.x0).max() < 1e-9


def test_dcm_multipliers_and_monodromy_equal_the_map_derivative(dcm):
    _b, r = dcm
    mu = np.sort(np.abs(r.multipliers))
    assert mu[0] < 1e-9
    assert mu[1] == pytest.approx(ref.DCM_MULTIPLIERS[1], abs=1e-6)
    assert r.stable and r.unstable_multipliers == []
    # M against a central difference of the map as computed
    s = r._ctx
    eps = 1e-7
    fd = np.empty_like(r.monodromy)
    for j in range(len(r.x0)):
        e = np.zeros(len(r.x0))
        e[j] = eps
        fd[:, j] = (s.run(r.x0 + e).x_end - s.run(r.x0 - e).x_end) / (2 * eps)
    assert np.abs(fd - r.monodromy).max() < 1e-7


def test_dcm_dc_gain_duty_to_vout(dcm):
    _b, r = dcm
    sm = r.sampled_model(inputs=[("duty", "S")], outputs=["v(vout)"])
    assert np.allclose(sm.A, r.monodromy)
    h0 = sm.response([0.0])[0, 0, 0]
    assert h0.real == pytest.approx(ref.DCM_DVC_DD, rel=1e-6)
    assert abs(h0.imag) < 1e-9


def test_source_input_matches_a_finite_difference(dcm):
    """B for a DC source value, from the same propagators and
    saltation matrices, against the map run at Vin +- dV."""
    b, r = dcm
    sm = r.sampled_model(inputs=[("source", "Vin")], outputs=["vout"])
    s = r._ctx
    dv = 1e-3
    cols = []
    for sign in (1, -1):
        b.set_value("Vin.V", ref.VIN + sign * dv)
        try:
            other = p.periodic_steady_state(b, period=T,
                                            switch_fn=_pwm(b, 0.3))
            cols.append(other._ctx.run(r.x0).x_end)
        finally:
            b.set_value("Vin.V", ref.VIN)
    fd = (cols[0] - cols[1]) / (2 * dv)
    assert np.allclose(sm.B[:, 0], fd, rtol=1e-6, atol=1e-9)
    # vout is the capacitor state: C selects it, D is zero
    assert np.allclose(sm.C[0], [1.0, 0.0]) or np.allclose(
        sm.C[0], [0.0, 1.0])
    assert abs(sm.D[0, 0]) < 1e-12
    _ = s


def test_ccm_duty_to_output_matches_the_averaged_model():
    """The test_sampled_ac buck (Q = 8): within 1 % of
    Vin/(1 + sL/R + s^2 LC) below 5 kHz (fixed engine: 0.77 %)."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_switch("Q1", "vin", "sw", 1e3, 1e-9)
    b.add_diode("D1", "gnd", "sw", 1e3, 1e-9, V_th=0.7)
    b.add_inductor("L1", "sw", "vout", 220e-6)
    b.add_capacitor("Cout", "vout", "gnd", 220e-6)
    b.add_resistor("R_L", "vout", "gnd", 8.0)
    sf = p.make_pwm_switch_fn(1 / T, 0.42, b.switch_index_of("Q1"),
                              b.graph.num_switches)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf)
    sm = r.sampled_model(inputs=[("duty", "Q1")], outputs=["vout"])
    f = np.logspace(1.5, np.log10(5e3), 80)
    H = sm.response(f)[:, 0, 0]
    s = 2j * np.pi * f
    ana = 12.0 / (1 + s * 220e-6 / 8.0 + s ** 2 * 220e-6 * 220e-6)
    assert (np.abs(np.abs(H) - np.abs(ana)) / np.abs(ana)).max() < 0.01


# ---------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------

def test_incommensurate_sine_source_is_named():
    b = _buck(50.0)
    b.add_sine_voltage_source("Vgrid", "g", "gnd", 0.0, 1.0, 50.3)
    b.add_resistor("Rg", "g", "gnd", 10.0)
    with pytest.raises(ValueError, match=r"Vgrid.*50\.3 Hz"):
        p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3))


def test_commensurate_sine_source_is_solved():
    """A diode rectifier on a sine whose period divides T: the map
    needs the sine's time derivative in the saltation (g_t), and the
    1 mV hysteresis band (V_th > V_f) makes the field jump at turn-on,
    so the saltation matrix is not the identity there."""
    b = p.CircuitBuilder()
    b.add_sine_voltage_source("Vac", "ac", "gnd", 0.0, 10.0, 1 / T)
    b.add_resistor("Rs", "ac", "a", 1.0)
    b.add_diode("D", "a", "out", 1e3, 1e-9, V_th=1e-3)
    b.add_capacitor("Co", "out", "gnd", 10e-6)
    b.add_resistor("Rl", "out", "gnd", 100.0)
    r = p.periodic_steady_state(b, period=T)
    long = p.simulate(b, t_end=3000 * T, dt=T, engine="dsed")
    x_far = np.asarray(long.states_reduced[-1])
    assert np.abs(x_far - r.x0).max() < 1e-6 * np.abs(r.x0).max()
    s = r._ctx
    eps = 1e-6
    fd = (s.run(r.x0 + eps).x_end - s.run(r.x0 - eps).x_end) / (2 * eps)
    assert abs(fd[0] - r.monodromy[0, 0]) < 1e-6


@pytest.mark.parametrize("kind", ["sine", "pulse", "pwl"])
def test_waveform_current_sources_and_vccs_are_solved(kind):
    """The add-controlled-source-set kinds reach the map through the
    dsed adapter: the orbit equals a tight TR-BDF2 transient (measured
    agreement 5e-8 relative, ideal pulse edges included)."""
    Tp = 1e-3

    def build():
        b = p.CircuitBuilder()
        if kind == "sine":
            b.add_sine_current_source("I", "gnd", "o", 0.0, 1e-3, 1 / Tp)
        elif kind == "pulse":
            b.add_pulse_current_source("I", "gnd", "o", 0.0, 1e-3, 1e-4,
                                       3e-4, Tp, 0.0, 0.0)
        else:
            b.add_pwl_current_source("I", "gnd", "o", [0, 2e-4, 6e-4, Tp],
                                     [0, 1e-3, -1e-3, 0], periodic=True)
        b.add_resistor("R", "o", "gnd", 1e3)
        b.add_capacitor("C", "o", "gnd", 1e-6)
        b.add_vccs("G", "o", "gnd", "gnd", "q", 1e-3)
        b.add_resistor("Rq", "q", "gnd", 1e3)
        b.add_capacitor("Cq", "q", "gnd", 1e-6)
        return b

    r = p.periodic_steady_state(build(), period=Tp)
    tr = p.simulate(build(), 30 * Tp, engine="trbdf2", rtol=1e-9,
                    atol=1e-12)
    x_tr = np.array([tr.v(lbl[2:-1])[-1] for lbl in r.state_labels])
    assert sorted(r.state_labels) == ["v(o)", "v(q)"]
    assert np.abs(r.x0 - x_tr).max() < 1e-6


def test_lossless_lc_loop_is_not_isolated():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "a", "gnd", 1.0)
    b.add_inductor("Lx", "a", "m", 1e-3)
    b.add_capacitor("Cx", "m", "gnd", 1e-6)
    with pytest.raises(ValueError, match=r"not isolated.*Cx.*Lx|"
                                         r"not isolated.*Lx.*Cx"):
        p.periodic_steady_state(b, period=T)


def test_opaque_python_gate_is_refused():
    b = _buck(50.0)
    with pytest.raises(p.UnsupportedError, match="opaque Python callable") as ei:
        p.periodic_steady_state(b, period=T,
                                switch_fn=_grid_gate(b, 0.3))
    assert (ei.value.engine, ei.value.feature) == (
        "periodic_steady_state", "switch_fn")


def test_schedule_not_periodic_in_the_period_is_refused():
    b = _buck(50.0)
    with pytest.raises(ValueError, match="not periodic"):
        p.periodic_steady_state(b, period=T,
                                switch_fn=p.make_pwm_switch_fn(
                                    1 / (2 * T), 0.3, 0, 2))


def test_nonlinear_device_is_refused():
    b = _buck(50.0)
    b.add_shockley_diode("Dn", "vout", "gnd")
    with pytest.raises(p.UnsupportedError, match="nonlinear or stateful.*Dn") as ei:
        p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3))
    assert (ei.value.device, ei.value.kind) == ("Dn", "ShockleyDiode")


def test_power_mosfet_composite_is_refused_by_name():
    # components() reports the channel as kind "power_mosfet"; with linear
    # gate capacitances and no body diode it is the only Newton entry.
    b = _buck(50.0)
    b.add_voltage_source("Vg", "g", "gnd", 0.0)
    b.add_power_mosfet("Mq", "vout", "gnd", "g", K=1e-2, V_T=2.0,
                       C_gs=1e-9, C_gd=100e-12, C_ds=0.0)
    with pytest.raises(p.UnsupportedError, match="nonlinear or stateful.*Mq") as ei:
        p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3))
    assert (ei.value.device, ei.value.kind) == ("Mq", "MosfetLevel1")


def test_budget_exhaustion_reports_the_history():
    b = _buck(50.0)
    with pytest.raises(ValueError, match="max_period_runs=2.*Residual"):
        p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.3),
                                max_period_runs=2)


# ---------------------------------------------------------------------
# 4. delegation
# ---------------------------------------------------------------------

def test_steady_state_without_dt_uses_the_solver():
    b = _buck(50.0)
    r = p.steady_state(b, period=T, switch_fn=_pwm(b, 0.3))
    assert isinstance(r, p.PeriodicSteadyStateResult)
    assert _iv(r)[1] == pytest.approx(ref.DCM_ORBIT[1], rel=1e-8)


def test_frequency_response_without_dt_on_the_dcm_buck():
    b = _buck(50.0)
    k, n = b.switch_index_of("S"), b.graph.num_switches

    def of(D):
        return p.make_pwm_switch_fn(1 / T, D, k, n)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fr = p.frequency_response(b, period=T, switch_fn_of=of,
                                  output="vout", u0=0.3,
                                  frequencies=[1e-3, 100.0, 1e3, 1e4])
    assert np.isfinite(fr.H).all()
    assert fr.H[0].real == pytest.approx(ref.DCM_DVC_DD, rel=1e-6)


# A waveform current source that is not T-periodic has no T-periodic
# orbit; before the check each of these returned stable=True with an
# x0 the transient never reaches (review probe, T = 1 ms into
# 1 kOhm || 1 uF): sine at 1.5/T -> -0.227 V (transient alternates
# +-0.105 V), one-shot pulse -> -0.225 V (transient decays to 0),
# periodic=False table -> +0.061 V (settles at -0.500 V), table of
# period 2T -> -0.490 V (alternates -0.132 / -0.358 V).
TP = 1e-3
_NOT_T_PERIODIC = {
    "sine_1.5_per_T": (lambda b: b.add_sine_current_source(
        "I", "gnd", "o", 0.0, 1e-3, 1.5 / TP), "SineCurrentSource",
        "f.T = 1.5"),
    "pulse_period_0.7T": (lambda b: b.add_pulse_current_source(
        "I", "gnd", "o", 0.0, 1e-3, 1e-4, 3e-4, 0.7 * TP, 0.0, 0.0),
        "PulseCurrentSource", "does not divide"),
    "pulse_one_shot": (lambda b: b.add_pulse_current_source(
        "I", "gnd", "o", 0.0, 1e-3, 1e-4, 3e-4, 0.0, 0.0, 0.0),
        "PulseCurrentSource", "one-shot"),
    "pulse_late_start": (lambda b: b.add_pulse_current_source(
        "I", "gnd", "o", 0.0, 1e-3, 0.9 * TP, 0.5 * TP, TP, 0.0, 0.0),
        "PulseCurrentSource", "first period"),
    "table_not_periodic": (lambda b: b.add_pwl_current_source(
        "I", "gnd", "o", [0, 2e-4, 6e-4, TP], [0, 1e-3, -1e-3, 0.5e-3],
        periodic=False), "PwlCurrentSource", "periodic=False"),
    "table_period_2T": (lambda b: b.add_pwl_current_source(
        "I", "gnd", "o", [0, 5e-4, TP, 2 * TP], [0, 1e-3, 0, 0],
        periodic=True), "PwlCurrentSource", "does not divide"),
}


@pytest.mark.parametrize("case", list(_NOT_T_PERIODIC))
def test_current_source_not_periodic_in_the_period_is_refused(case):
    add, kind, frag = _NOT_T_PERIODIC[case]
    b = p.CircuitBuilder()
    add(b)
    b.add_resistor("R", "o", "gnd", 1e3)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    with pytest.raises(p.UnsupportedError, match=frag) as ei:
        p.periodic_steady_state(b, period=TP)
    e = ei.value
    assert (e.feature, e.device, e.kind) == ("period", "I", kind)


def test_slow_mode_is_not_accepted_on_a_small_residual():
    """Zero-mean sine current into 1 uF || 1 TOhm (|mu| = 1 - 1e-9): from
    zero the residual is already within rtol, and the solver used to
    return x0 = 0 with stable=True. The state error is residual/(1 - |mu|);
    one Newton step finds the orbit, whose v(0) is i_pk/(omega C) =
    0.159155 V. The map's rounding (~1e-12 V) is amplified 1e9 times, so
    the answer is good to about 1 % (measured 0.15726 V)."""
    b = p.CircuitBuilder()
    b.add_sine_current_source("I", "gnd", "o", 0.0, 1e-3, 1 / TP)
    b.add_capacitor("C", "o", "gnd", 1e-6)
    b.add_resistor("Rbig", "o", "gnd", 1e12)
    r = p.periodic_steady_state(b, period=TP)
    assert r.floquet_radius == pytest.approx(1 - 1e-9, abs=1e-12)
    assert r.x0[0] == pytest.approx(1e-3 / (2 * np.pi / TP * 1e-6), rel=2e-2)


@pytest.mark.parametrize("v_th", (0.0, 1e-3))
@pytest.mark.parametrize("R,v_out,i_l", [(20.0, 24.053630, 2.098779),
                                         (500.0, 36.594401, 0.0)])
def test_boost_with_a_zero_band_diode_settles(v_th, R, v_out, i_l):
    """Before add-exact-propagator-strategy this boost failed ("did not
    settle after 64 rounds") at V_th = 0 and 1e-3. CCM (20 Ohm) and DCM
    (500 Ohm): trbdf2 (rtol 1e-10) started from the orbit stays on it
    for 50 periods to 1e-6 V. (A plain dsed transient of the same boost
    is refused by the diode-law guard: the PSS map runs without event
    projection at rtol 1e-11.)"""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "vin", "gnd", 12.0)
    b.add_inductor("L", "vin", "sw", 100e-6)
    b.add_switch("S", "sw", "gnd", ref.GON, ref.GOFF)
    b.add_diode("D1", "sw", "vout", ref.GON, ref.GOFF, v_th)
    b.add_capacitor("C", "vout", "gnd", 47e-6)
    b.add_resistor("RL", "vout", "gnd", R)
    r = p.periodic_steady_state(b, period=T, switch_fn=_pwm(b, 0.5))
    lab = r.state_labels
    assert r.stable and r.residual < 1e-10 * v_out   # 1.14e-9 V at 36.6 V
    assert r.x0[lab.index("v(vout)")] == pytest.approx(v_out, abs=1e-5)
    assert r.x0[lab.index("i(L)")] == pytest.approx(i_l, abs=1e-5)
    assert r.exact["masks"]
