"""engine='auto' — variable-step TR-BDF2 on the sparse MNA kernel.

v2.0 Phase 3. The C++ tests pin the stepper against closed forms;
these pin the PYTHON surface and the two audit gates' physics:
the buck answer without anyone choosing dt, and the flyback's
commutation treatment (L-stable crossing of the snubber ring).
Wall-clock is NOT asserted here (CI machines vary) — the gate
numbers live in the PR/CHANGELOG measurements.
"""

import warnings

import numpy as np
import pytest

import pulsim as p


def _mean_tail(res, node, frac=0.5):
    t = np.asarray(res.times)
    v = np.asarray(res.v(node))
    cut = t[0] + (t[-1] - t[0]) * frac
    m = t > cut
    return float(np.trapezoid(v[m], t[m]) / (t[m][-1] - t[m][0]))


def test_rc_analytic_and_tolerance_proportionality():
    def run(rtol):
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 5.0)
        b.add_resistor("R", "in", "n1", 1e3)
        b.add_capacitor("C", "n1", "gnd", 1e-6)
        # engine='trbdf2' names the variable-step engine
        # explicitly: under engine='auto' an explicit dt requests a
        # FIXED step, and here dt is the step CEILING.
        res = p.simulate(b, t_end=5e-3, engine="trbdf2", rtol=rtol,
                          atol=rtol * 1e-3, dt=2e-4)
        t = np.asarray(res.times)
        v = np.asarray(res.v("n1"))
        ref = 5.0 * (1.0 - np.exp(-t / 1e-3))
        return float(np.abs(v - ref).max()), len(t)

    worst5, n5 = run(1e-5)
    worst7, n7 = run(1e-7)
    assert worst5 < 1e-3
    assert worst7 < worst5 / 5.0     # tolerance buys accuracy
    assert n5 < 800                   # and the controller is pacing
    # (a fixed grid resolving the same needs thousands of steps)


def test_no_dt_needed_and_result_surface():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 10.0)
    b.add_resistor("R", "in", "n1", 100.0)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    res = p.simulate(b, t_end=1e-3, engine="auto")   # NO dt anywhere
    t = np.asarray(res.times)
    assert t[0] == 0.0 and t[-1] == pytest.approx(1e-3, rel=1e-6)
    # irregular grid is the point:
    assert len(np.unique(np.round(np.diff(t), 15))) > 3
    # name-based accessors work off the full MNA layout
    assert np.isfinite(res.v("n1")).all()
    assert np.isfinite(res.i("R")).all()
    assert res._trbdf2_stats["n_accept"] > 0


def test_buck_matches_fixed_dt_reference():
    """The audit gate's physics: 100 kHz buck, 5 ms, nobody chose
    dt. Must land on the dt=1e-8 fixed-trap answer."""
    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 48.0)
        b.add_switch("HS", "in", "sw", 1e3, 1e-9)
        b.add_diode("DFW", "gnd", "sw", 1e3, 1e-9)
        b.add_inductor("L", "sw", "out", 22e-6)
        b.add_capacitor("C", "out", "gnd", 100e-6)
        b.add_resistor("R", "out", "gnd", 1.0)
        return b

    b = build()
    pwm = p.NativePwm2Switch(10e-6, 0.5, b.graph.num_switches, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # V_th=0 freewheel confesses
        res = p.simulate(b, t_end=5e-3, switch_fn=pwm,
                          engine="auto")
    vm = _mean_tail(res, "out")
    # Reference measured at dt=1e-8 (500k steps): 23.976496 V.
    assert vm == pytest.approx(23.9765, abs=5e-3)
    st = res._trbdf2_stats
    # The engine must be doing EVENT-DRIVEN work, not grid work:
    # ~1000 gate edges landed, ~1000 freewheel commutations, and
    # fifty-times fewer accepted steps than the fixed reference.
    assert st["n_gate_events"] >= 998
    assert st["n_diode_events"] >= 998
    assert st["n_accept"] < 20000


def test_flyback_snubber_parity_and_spike():
    """Commutation treatment: leakage + RC snubber ring (1.2 MHz,
    kV-class spike). The fixed-trap ladder converges to
    <vout> = 16.7902 / peak 1225.6 at dt = 5e-9; the variable
    engine must land there at default tolerance."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_transformer("T1", "in", "sw", "sec", "gnd",
                       200e-6, 50e-6, 0.98)
    b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9)
    b.add_resistor("Rsn", "sw", "sn1", 47.0)
    b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9)
    b.add_diode("D1", "sec", "vout", 1e3, 1e-9, 0.7)
    b.add_capacitor("Co", "vout", "gnd", 47e-6)
    b.add_resistor("Ro", "vout", "gnd", 5.0)
    T = 10e-6
    q = b.switch_index_of("Q1")
    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(q, (t % T) / T < 0.4)
        return m

    res = p.simulate(b, t_end=3e-3, switch_fn=sf, engine="auto")
    vm = _mean_tail(res, "vout")
    vsw = np.asarray(res.v("sw"))
    assert vm == pytest.approx(16.79, abs=0.02)
    assert vsw.max() == pytest.approx(1225.6, rel=5e-3)


def test_vth_hysteresis_gives_tolerance_convergence():
    """With the diode's physical V_th (the hysteresis band), the
    answer converges monotonically toward the fine-trap reference
    as rtol tightens — measured 16.789986 at dt=2e-9."""
    def run(rtol):
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 48.0)
        b.add_transformer("T1", "in", "sw", "sec", "gnd",
                           200e-6, 50e-6, 0.98)
        b.add_switch("Q1", "sw", "gnd", 1e3, 1e-9)
        b.add_resistor("Rsn", "sw", "sn1", 47.0)
        b.add_capacitor("Csn", "sn1", "gnd", 2.2e-9)
        b.add_diode("D1", "sec", "vout", 1e3, 1e-9, 0.7)
        b.add_capacitor("Co", "vout", "gnd", 47e-6)
        b.add_resistor("Ro", "vout", "gnd", 5.0)
        T = 10e-6
        q = b.switch_index_of("Q1")
        n = b.graph.num_switches

        def sf(t):
            m = p.SwitchStateMask(n)
            m.set(q, (t % T) / T < 0.4)
            return m

        res = p.simulate(b, t_end=3e-3, switch_fn=sf,
                          engine="auto", rtol=rtol,
                          atol=rtol * 1e-3)
        return _mean_tail(res, "vout"), res._trbdf2_stats

    ref = 16.789986
    v5, st5 = run(1e-5)
    v6, st6 = run(1e-6)
    assert abs(v6 - ref) <= abs(v5 - ref) + 5e-5
    assert abs(v6 - ref) < 2e-4
    # No boundary riding with a real V_th: the event count is the
    # physical 2-per-period, identical at both tolerances.
    assert st5["n_diode_events"] == st6["n_diode_events"]
    # A startup graze is fine; sliding produces them continually.
    assert st5["n_chatter_breaks"] <= 3


def test_sliding_mode_diode_confesses():
    """V_th = 0 in DCM rides the conduction boundary (the fixed
    reference chatters ~630 flips/period on this circuit). The
    engine must WARN with the mechanism, not stay silent."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_switch("HS", "in", "sw", 1e3, 1e-9)
    b.add_diode("DFW", "gnd", "sw", 1e3, 1e-9)   # V_th = 0
    b.add_inductor("L", "sw", "out", 22e-6)
    b.add_capacitor("C", "out", "gnd", 100e-6)
    b.add_resistor("R", "out", "gnd", 1.0)
    pwm = p.NativePwm2Switch(10e-6, 0.5, b.graph.num_switches,
                              True)
    with pytest.warns(UserWarning, match="conduction boundary"):
        p.simulate(b, t_end=5e-3, switch_fn=pwm, engine="auto")


def test_gate_edge_lands_on_awkward_instant():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_switch("S", "in", "n1", 1e3, 1e-9)
    b.add_resistor("R", "n1", "n2", 1e3)
    b.add_capacitor("C", "n2", "gnd", 1e-6)
    t_close = 1.2345e-3
    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(0, t >= t_close)
        return m

    res = p.simulate(b, t_end=6e-3, switch_fn=sf, engine="auto")
    t = np.asarray(res.times)
    assert np.abs(t - t_close).min() < 1e-9
    tau = (1e3 + 1e-3) * 1e-6
    v2 = np.asarray(res.v("n2"))
    post = t > t_close + 1e-9
    ref = 5.0 * (1.0 - np.exp(-(t[post] - t_close) / tau))
    assert np.abs(v2[post] - ref).max() < 1e-3


def test_refusals_name_the_mechanism():
    def fresh():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 5.0)
        b.add_resistor("R", "in", "n1", 1e3)
        b.add_capacitor("C", "n1", "gnd", 1e-6)
        return b

    from pulsim import _pulsim as _k
    b = fresh()
    b.add_nonlinear_diode("D", "n1", "gnd", _k.IdealDiodeParams())
    # 'auto' would ROUTE this to the fixed engine; asking for the
    # variable one by name gets the reason instead.
    with pytest.raises(ValueError, match="nonlinear device"):
        p.simulate(b, t_end=1e-3, engine="trbdf2")
    # And 'auto' with no dt says what to do about it.
    with pytest.raises(ValueError, match="pass dt="):
        p.simulate(b, t_end=1e-3, engine="auto")

    for kwargs, match in (
        (dict(step_observer=lambda t, x: None), "step_observer"),
        (dict(store_every=4), "store_every"),
        (dict(start_from_dc_op=True), "start_from_dc_op"),
        (dict(max_dt_halvings=3), "max_dt_halvings"),
    ):
        with pytest.raises(ValueError, match=match):
            p.simulate(fresh(), t_end=1e-3, engine="trbdf2",
                        **kwargs)

    # A controlled switch with no driver is an ERROR here, not a
    # silent all-closed short (the v2.0 semantics).
    b = fresh()
    b.add_switch("S", "in", "n1", 1e3, 1e-9)
    with pytest.raises(ValueError, match="switch"):
        p.simulate(b, t_end=1e-3, engine="auto")


def test_user_b_extra_current_injection():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 2.0)
    state_size = b.pool.state_size(b.graph)
    row = b.node_id_of("a")

    def ub(t):
        v = np.zeros(state_size)
        v[row] = -1.0          # inject 1 A into a -> v(a) = 2 V
        return v

    res = p.simulate(b, t_end=1e-4, engine="auto", b_extra_fn=ub)
    assert res.v("a")[-1] == pytest.approx(2.0, rel=1e-9)


# ---------------------------------------------------------------
# Gaps the adversarial review named (each one demonstrated a real
# silent-wrong-answer or a silently-ignored kwarg before the fix).
# ---------------------------------------------------------------

def test_narrow_pulse_train_is_not_stepped_over():
    """The step ceiling must respect the circuit's fastest source.

    A 500 ns pulse every 10 µs feeding a peak detector: with the
    naive span/1000 ceiling the engine stepped straight over every
    pulse for a 10 ms run and reported 0.0000 V — silently, with
    zero diode events. The kernel now caps h_max at a third of the
    narrowest pulse.
    """
    def build():
        b = p.CircuitBuilder()
        b.add_pulse_voltage_source("Vp", "in", "gnd",
                                    0.0, 10.0, 0.0, 500e-9, 10e-6)
        b.add_diode("D", "in", "out", 1e3, 1e-9, 0.6)
        b.add_capacitor("Co", "out", "gnd", 100e-9)
        b.add_resistor("Ro", "out", "gnd", 100e3)
        return b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(build(), t_end=10e-3, engine="auto")
        ref = p.simulate(build(), t_end=10e-3, dt=5e-9, engine="pwl")
    v_auto = float(np.asarray(res.v("out"))[-1])
    v_ref = float(np.asarray(ref.v("out"))[-1])
    assert v_auto == pytest.approx(v_ref, rel=0.02)
    assert res._trbdf2_stats["n_diode_events"] > 0


def test_should_continue_cancels_and_keeps_partial():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    calls = []

    def keep_going():
        calls.append(1)
        return len(calls) < 5

    res = p.simulate(b, t_end=5e-3, engine="auto",
                      should_continue=keep_going)
    assert len(calls) >= 5
    # Cancelled early: far fewer samples than a full run, and the
    # partial trace survives.
    assert 0 < len(res.times) < 50
    assert np.isfinite(res.v("n1")).all()


def test_stats_expose_forced_accepts():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    res = p.simulate(b, t_end=1e-3, engine="auto")
    # The key must EXIST even at zero — the warning that reads it
    # was dead code while the binding omitted it.
    assert "n_forced_accepts" in res._trbdf2_stats
    assert res._trbdf2_stats["n_forced_accepts"] == 0


def test_fft_and_rms_survive_the_irregular_grid():
    """Two analyses that infer a sample rate or average samples.

    On the accepted grid `times[1]-times[0]` was 6.6e6x off the
    mean spacing (the first step is an event landing), so the FFT
    axis was stretched into nonsense; and a sample-mean RMS
    over-weights the event clusters where ripple peaks.
    """
    from pulsim._result_views import (grid_is_uniform,
                                       time_weighted_rms)

    def build():
        b = p.CircuitBuilder()
        b.add_voltage_source("Vin", "in", "gnd", 48.0)
        b.add_switch("HS", "in", "sw", 1e3, 1e-9)
        b.add_diode("DFW", "gnd", "sw", 1e3, 1e-9)
        b.add_inductor("L", "sw", "out", 22e-6)
        b.add_capacitor("C", "out", "gnd", 100e-6)
        b.add_resistor("R", "out", "gnd", 1.0)
        return b

    b = build()
    pwm = p.NativePwm2Switch(10e-6, 0.5, b.graph.num_switches, True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = p.simulate(b, t_end=5e-3, switch_fn=pwm,
                          engine="auto")
    assert not grid_is_uniform(np.asarray(res.times))

    # The FFT half needs the plotting stack, which is an optional
    # extra; the resampling it relies on is exercised directly
    # below either way.
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    if matplotlib is not None:
        matplotlib.use("Agg")
        _fig, freqs, mag = p.scope_fft(b, res, signal="out",
                                        f_fundamental=100e3,
                                        show=False)
        freqs = np.asarray(freqs)
        mag = np.asarray(mag)
        k = int(np.argmax(mag[1:])) + 1
        assert freqs[k] == pytest.approx(100e3, rel=1e-3)

    # The resampling the FFT relies on, checked without the
    # plotting stack: a uniform regrid must preserve the ripple.
    from pulsim._result_views import resample_uniform
    t_raw = np.asarray(res.times)
    v_raw = np.asarray(res.v("out"))
    t_u, v_u, dt_u = resample_uniform(t_raw, v_raw)
    assert grid_is_uniform(t_u)
    sel_r = t_raw > 4e-3
    sel_u = t_u > 4e-3
    p2p_raw = v_raw[sel_r].max() - v_raw[sel_r].min()
    p2p_u = v_u[sel_u].max() - v_u[sel_u].min()
    assert p2p_u == pytest.approx(p2p_raw, rel=5e-3)

    # Time-weighted RMS matches the fixed-dt reference; the
    # sample mean does not have to.
    b2 = build()
    hs = b2.switch_index_of("HS")
    n = b2.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(hs, (t % 10e-6) / 10e-6 < 0.5)
        return m

    ref = p.simulate(b2, t_end=5e-3, dt=1e-8, switch_fn=sf,
                      engine="pwl")
    rms_ref = time_weighted_rms(np.asarray(ref.i("L")),
                                 np.asarray(ref.times))
    rms_auto = time_weighted_rms(np.asarray(res.i("L")),
                                  np.asarray(res.times))
    assert rms_auto == pytest.approx(rms_ref, rel=2e-3)


def test_t_start_offset_runs():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    res = p.simulate(b, t_end=6e-3, t_start=1e-3, engine="auto")
    t = np.asarray(res.times)
    assert t[0] == pytest.approx(1e-3)
    assert t[-1] == pytest.approx(6e-3, rel=1e-6)
    # The capacitor starts at 0 V at t_start, so the curve is the
    # same exponential shifted by t_start.
    v = np.asarray(res.v("n1"))
    ref = 5.0 * (1.0 - np.exp(-(t - 1e-3) / 1e-3))
    assert np.abs(v - ref).max() < 1e-3


def test_every_refusal_key_fires():
    """A dropped key means the kwarg is silently ignored — the
    failure mode this whole refusal dict exists to prevent."""
    def fresh():
        b = p.CircuitBuilder()
        b.add_voltage_source("V", "in", "gnd", 5.0)
        b.add_resistor("R", "in", "n1", 1e3)
        b.add_capacitor("C", "n1", "gnd", 1e-6)
        return b

    # step_observer and closed_loops are SUPPORTED here now (with
    # a scheduled cadence — see test_engine_auto_closed_loop.py);
    # an observer with no cadence still refuses, pinned there.
    cases = [
        (dict(live_stream=object()), "live_stream"),
        (dict(start_from_dc_op=True), "start_from_dc_op"),
        (dict(strict_event_iterations=True),
         "strict_event_iterations"),
        (dict(max_dt_halvings=3), "max_dt_halvings"),
        (dict(store_every=4), "store_every"),
        (dict(enable_substep_state_correction=True), "substep"),
        (dict(inductor_freeze_di_max=1.0),
         "inductor_freeze_di_max"),
        (dict(inductor_abs_clamp=100.0), "inductor_abs_clamp"),
        (dict(progress=True), "progress"),
    ]
    for kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            p.simulate(fresh(), t_end=1e-3, engine="trbdf2",
                        **kwargs)


def test_newton_kwargs_warn_instead_of_silently_dropping():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "in", "gnd", 5.0)
    b.add_resistor("R", "in", "n1", 1e3)
    b.add_capacitor("C", "n1", "gnd", 1e-6)
    with pytest.warns(UserWarning, match="Newton"):
        p.simulate(b, t_end=1e-3, engine="auto",
                    max_newton_iterations=5)


def test_voltage_sanity_detector_runs_under_auto():
    """The detector is engine-independent and this engine's
    all-OPEN default makes the inductor-open case MORE likely."""
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", 48.0)
    b.add_inductor("L", "in", "sw", 1e-3)
    b.add_switch("S", "sw", "gnd", 1e3, 1e-9)
    n = b.graph.num_switches

    def sf(t):
        m = p.SwitchStateMask(n)
        m.set(0, (t % 100e-6) < 50e-6)
        return m

    with pytest.warns(UserWarning, match="largest voltage"):
        p.simulate(b, t_end=3e-4, switch_fn=sf, engine="auto")
