"""Periodic steady state with compiled control graphs in x0 (Task 7.1 / 7.2).

Verifies:
1. A voltage-mode buck converter with a PI controller at the switching
   rate (f_sw = 1/T): ``periodic_steady_state`` includes the controller's
   discrete state vector ``x_ctrl`` in ``x0 = [x_circuit; x_ctrl]``,
   converges rapidly from zero, achieves zero sampled steady-state error
   ``|vout(0) - V_ref| < 1e-9``, and matches a long settled transient
   within ``1e-8`` across both circuit and controller states.
2. Composite compiled control graphs (``PIController`` + ``DelayBlock``,
   ``compute_delay='one_sample'``, ``control=`` / ``step_observer=`` /
   ``steady_state(dt=None)``), saturated starts, and state preservation
   on failure.
3. Refusals with ``UnsupportedError("periodic_steady_state",
   "hidden_controller_state", ...)`` for plain Python closures with
   mutable state, ``nonlocal`` mutations, stateful user objects, and
   uninspectable ``PythonBlock``s.
4. Full parity for stateless schedules (``make_pwm_switch_fn``,
   ``SwitchSchedule``, stateless Python ``switch_fn``).
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim._control_pss import ControlGraphStateMapper

T = 1e-5
V_IN = 24.0
V_REF = 12.0


def _buck(R: float = 5.0, C: float = 47e-6):
    b = p.CircuitBuilder()
    b.add_voltage_source("Vin", "in", "gnd", V_IN)
    b.add_switch("Q", "in", "sw", 1e3, 1e-9)
    b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
    b.add_inductor("L", "sw", "out", 100e-6)
    b.add_capacitor("C", "out", "gnd", C)
    b.add_resistor("R", "out", "gnd", R)
    return b


def _pi_loop(b, *, Kp: float = 0.03, Ki: float = 100.0, setpoint: float = V_REF):
    pi = p.PIController(Kp=Kp, Ki=Ki, output_min=0.05, output_max=0.95)
    return p.bind_pi_to_switch(
        b, pi=pi, measured="out", setpoint=setpoint, switch="Q", freq=1.0 / T
    )


# ---------------------------------------------------------------------------
# 1. Voltage-mode buck with PI at switching rate (exact, pwl, trbdf2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "engine,kw",
    [
        ("exact", {}),
        ("pwl", {"dt": T / 100}),
        ("trbdf2", {}),
    ],
)
def test_voltage_mode_buck_pi_orbit_and_long_transient(engine, kw):
    b = _buck()
    lp = _pi_loop(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = p.periodic_steady_state(b, period=T, closed_loops=[lp], engine=engine, **kw)

    assert isinstance(r, p.PeriodicSteadyStateResult)
    assert r.stable
    assert r.n_period_runs <= 20
    assert r.residual < 1e-8
    assert r.residual_history[0] > 1.0
    assert r.residual_history[-1] < 1e-8

    # Controller states are part of x0 and exposed separately
    assert r.controller_labels == ["ctrl(pi.integrator_state)", "ctrl(pi._prev_error)"]
    assert r.controller_state.shape == (2,)

    # Zero sampled steady-state error at the switching-rate PI
    v0 = float(r.x0[r.state_labels.index("v(out)")])
    assert abs(v0 - V_REF) < 1e-9
    assert abs(float(r.controller_state[1])) < 1e-9  # prev_error == 0

    # Orbit waveform closes and reproduces zero sampled error
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orb = r.orbit
    assert abs(float(np.asarray(orb.v("out"))[0]) - V_REF) < 1e-9
    assert abs(float(np.asarray(orb.v("out"))[-1]) - v0) < 1e-8

    # Compare against a long settled transient on the same engine
    b_tr = _buck()
    lp_tr = _pi_loop(b_tr)
    sim_kw = (
        dict(
            dt=T / 100,
            discontinuity_restart=True,
            enable_substep_state_correction=True,
            control_tick="state",
        )
        if engine == "pwl"
        else dict(rtol=1e-11, atol=1e-14)
        if engine == "exact"
        else dict(dt=T / 100, rtol=1e-10, atol=1e-13)
    )
    sim_eng = "dsed" if engine == "exact" else engine
    n_periods = 6000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_tr = p.simulate(
            b_tr, n_periods * T, engine=sim_eng, closed_loops=[lp_tr], **sim_kw
        )

    if engine == "exact":
        v_tr = float(res_tr.states_reduced[-1][0])
        i_tr = float(res_tr.states_reduced[-1][1])
        circ_tr = np.array([v_tr, i_tr], dtype=float)
    else:
        i_tr = float(np.asarray(res_tr.i("L"))[-1])
        v_tr = float(np.asarray(res_tr.v("out"))[-1])
        circ_tr = np.array([i_tr, v_tr], dtype=float)

    ctrl_tr = ControlGraphStateMapper([lp_tr.step_observer], T).read()
    x_tr = np.concatenate([circ_tr, ctrl_tr])
    assert np.abs(r.x0 - x_tr).max() < 1e-8
    # A PI sampling once per period zeroes the SAMPLED error (above), not the
    # mean: the orbit's time-averaged error is the ripple's offset from its
    # value at the sampling instant, and the long transient's last period
    # has the same offset (measured 5.458e-5 V pwl, 5.459e-5 V trbdf2). The
    # exact orbit stores 3 samples a period (its edges); a trapezoid over
    # them is not the mean (it reads 5.678e-5 V), so it is not checked here.
    if engine != "exact":
        def _mean_err(tt, vv):
            return float(np.trapezoid(vv, tt) / (tt[-1] - tt[0])) - V_REF
        t_o = np.asarray(orb.times)
        e_orbit = _mean_err(t_o, np.asarray(orb.v("out")))
        t_l = np.asarray(res_tr.times)
        last = t_l >= t_l[-1] - T * (1 + 1e-9)
        e_tr = _mean_err(t_l[last], np.asarray(res_tr.v("out"))[last])
        assert 1e-5 < e_orbit < 1e-4      # a ripple offset, not zero
        assert abs(e_orbit - e_tr) < 1e-7

    # each part against its own half of the long transient
    assert np.abs(r.circuit_state - circ_tr).max() < 1e-8
    assert np.abs(r.controller_state - ctrl_tr).max() < 1e-8


def test_warm_start_and_steady_state_delegation():
    b = _buck()
    lp = _pi_loop(b)
    r1 = p.steady_state(b, period=T, closed_loops=[lp])
    assert isinstance(r1, p.PeriodicSteadyStateResult)
    assert abs(float(r1.x0[r1.state_labels.index("v(out)")]) - V_REF) < 1e-9

    # Warm start from converged result takes 1 Newton check + FD Jacobian
    r2 = p.periodic_steady_state(b, period=T, closed_loops=[lp], start_from=r1)
    assert np.allclose(r2.x0, r1.x0, rtol=0, atol=1e-9)
    assert r2.n_period_runs <= 6


# ---------------------------------------------------------------------------
# 2. Control graph blocks (DelayBlock, compute_delay, control=), saturation,
#    failure state preservation, and rate periodicity
# ---------------------------------------------------------------------------

def test_pi_plus_delay_block_and_transient_agreement():
    def build_chain(b):
        ch = p.MixedDomainBlockChain(rate=p.Rate(T))
        ch.add(
            "pi",
            p.PIController(Kp=0.02, Ki=50.0, output_min=0.05, output_max=0.95),
            inputs=dict(setpoint=V_REF, measured="out", dt="dt"),
            output="d_raw",
        )
        ch.add(
            "del",
            p.DelayBlock(samples=1, initial_value=0.5),
            inputs=dict(x="channel:d_raw"),
            output="duty",
        )
        obs = ch.make_step_observer(b)
        gate = ch._cxx_chain.make_pwm_gate(
            b.graph.num_switches, [("duty", b.switch_index_of("Q"), T, 0.0)]
        )
        gate._graph = ch
        gate._closed_loop = "MixedDomainBlockChain"
        return obs, gate

    b = _buck()
    obs, gate = build_chain(b)
    r = p.periodic_steady_state(b, period=T, switch_fn=gate, step_observer=obs)
    assert r.stable
    assert r.controller_labels == [
        "ctrl(pi.integrator_state)",
        "ctrl(pi._prev_error)",
        "ctrl(del.buf[0])",
    ]
    assert abs(float(r.x0[r.state_labels.index("v(out)")]) - V_REF) < 1e-9
    assert r.controller_state[2] == pytest.approx(r.controller_state[0], abs=1e-10)

    b2 = _buck()
    obs2, gate2 = build_chain(b2)
    tr = p.simulate(
        b2, 4000 * T, engine="dsed", switch_fn=gate2, step_observer=obs2,
        rtol=1e-11, atol=1e-14,
    )
    circ_tr = np.asarray(tr.states_reduced[-1], dtype=float)
    ctrl_tr = ControlGraphStateMapper([obs2], T).read()
    assert np.abs(r.x0 - np.concatenate([circ_tr, ctrl_tr])).max() < 1e-8


def test_saturated_start_relaxes_and_converges():
    # High Kp saturates duty at 0.95 from v_C = 0; relaxation exits saturation.
    b = _buck()
    lp = _pi_loop(b, Kp=0.1, Ki=150.0)
    r = p.periodic_steady_state(b, period=T, closed_loops=[lp], n_relax=8,
                                max_period_runs=80)
    assert r.stable
    assert abs(float(r.x0[r.state_labels.index("v(out)")]) - V_REF) < 1e-9


@pytest.mark.parametrize("engine", ["exact", "trbdf2"])
def test_control_alias_with_chain_and_drive_source(engine):
    # Passive RC with PI regulating output via drive_source through control=
    b = p.CircuitBuilder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add(
        "pi",
        p.PIController(Kp=0.1, Ki=500.0, output_min=-50.0, output_max=50.0),
        inputs=dict(setpoint=5.0, measured="out", dt="dt"),
        output="vin_cmd",
    )
    ch.drive_source(b, "vin_cmd", ("v", "src", "gnd"))
    b.add_resistor("R1", "src", "out", 10.0)
    b.add_capacitor("C1", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 100.0)

    r = p.periodic_steady_state(b, period=T, control=ch, engine=engine, rtol=1e-11)
    assert r.stable
    assert abs(float(r.x0[r.state_labels.index("v(out)")]) - 5.0) < 1e-9
    # Steady-state source voltage needed for 5 V across 100/(10+100) divider is 5.5 V
    assert r.controller_state[0] == pytest.approx(5.5, rel=1e-6)


def test_failure_preserves_caller_controller_state():
    b = _buck()
    lp = _pi_loop(b)
    cxx = lp.step_observer._cxx_chain
    # Advance the controller once so it has non-zero state before failing PSS
    cxx.make_tick_observer()(0.0, np.zeros(b.pool.state_size(b.graph)))
    before = cxx.dump_state(cxx.save_state())

    with pytest.raises(p.ConvergenceError):
        p.periodic_steady_state(b, period=T, closed_loops=[lp], max_period_runs=2)

    after = cxx.dump_state(cxx.save_state())
    assert after == before


def test_incommensurate_controller_rate_is_refused():
    b = _buck()
    ch = p.MixedDomainBlockChain(rate=p.Rate(0.7 * T))
    ch.add(
        "pi",
        p.PIController(Kp=0.02, Ki=20.0, output_min=0.05, output_max=0.95),
        inputs=dict(setpoint=V_REF, measured="out", dt="dt"),
        output="duty",
    )
    obs = ch.make_step_observer(b)
    gate = ch._cxx_chain.make_pwm_gate(
        b.graph.num_switches, [("duty", b.switch_index_of("Q"), T, 0.0)]
    )
    gate._graph = ch
    gate._closed_loop = "MixedDomainBlockChain"
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=gate, step_observer=obs)
    assert ei.value.feature == "period"


def test_ki_zero_singular_integrator_state_is_refused():
    b = _buck()
    lp = _pi_loop(b, Kp=0.02, Ki=0.0)
    with pytest.raises(p.ConvergenceError, match="not isolated.*ctrl\\(pi\\.integrator_state\\)"):
        p.periodic_steady_state(b, period=T, closed_loops=[lp])


# ---------------------------------------------------------------------------
# 3. Refusal for Python closures / objects / PythonBlocks with hidden state
# ---------------------------------------------------------------------------

class _StatefulObserver:
    def __init__(self):
        self.integral = 0.0

    def __call__(self, t, x):
        self.integral += 1.0


class _StatefulSwitch:
    def __init__(self, n, k):
        self.n, self.k = n, k
        self.duty = 0.5
        self.calls = 0

    def __call__(self, t):
        self.calls += 1                 # state that moves on every call
        m = p.SwitchStateMask(self.n)
        m.set(self.k, (t / T) % 1.0 < self.duty)
        return m

    def next_edge_after(self, t):
        return t + 0.5 * T


def _hidden_state_cases(b):
    n = b.graph.num_switches
    k = b.switch_index_of("Q")
    pwm = p.make_pwm_switch_fn(1.0 / T, 0.5, k, n)

    # 1. Closure observer over mutable list
    acc = [0.0]

    def list_obs(t, x):
        acc[0] += float(x[0])

    # 2. Closure observer mutating nonlocal scalar
    scalar_acc = 0.0

    def nonlocal_obs(t, x):
        nonlocal scalar_acc
        scalar_acc += 1.0

    # 3. Stateful callable object as step_observer
    obj_obs = _StatefulObserver()

    # 4. Switch closure over a list it appends to
    calls_log = []

    def list_sw(t):
        m = p.SwitchStateMask(n)
        calls_log.append(t)
        m.set(k, (t / T) % 1.0 < 0.5)
        return m

    # 5. Stateful callable switch object
    obj_sw = _StatefulSwitch(n, k)

    # 6. PythonBlock wrapping mutable closure inside a control graph
    py_acc = [0.0]

    def py_blk_fn(x):
        py_acc[0] += x
        return max(0.05, min(0.95, 0.5 - 0.01 * py_acc[0]))

    ch_closure = p.MixedDomainBlockChain(rate=p.Rate(T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ch_closure.add(
            "py_ctrl",
            p.PythonBlock(py_blk_fn),
            inputs=dict(x="node:out"),
            output="duty",
        )
    obs_py_closure = ch_closure.make_step_observer(b)

    # 7. PythonBlock wrapping stateful user class inside a control graph
    class _ObjBlock:
        def __init__(self):
            self.s = 0.0

        def update(self, x):
            self.s += x
            return 0.5

    ch_obj = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch_obj.add("py_obj", p.PythonBlock(_ObjBlock()), inputs=dict(x="node:out"), output="duty")
    obs_py_obj = ch_obj.make_step_observer(b)

    return {
        "closure_list_observer": dict(switch_fn=pwm, step_observer=list_obs),
        "closure_nonlocal_observer": dict(switch_fn=pwm, step_observer=nonlocal_obs),
        "stateful_object_observer": dict(switch_fn=pwm, step_observer=obj_obs),
        "closure_list_switch_fn": dict(switch_fn=list_sw),
        "stateful_object_switch_fn": dict(switch_fn=obj_sw),
        "python_block_closure": dict(switch_fn=pwm, step_observer=obs_py_closure),
        "python_block_object": dict(switch_fn=pwm, step_observer=obs_py_obj),
    }


@pytest.mark.parametrize(
    "engine,ek",
    [("exact", {}), ("pwl", {"dt": T / 100}), ("trbdf2", {})],
)
@pytest.mark.parametrize(
    "case",
    [
        "closure_list_observer",
        "closure_nonlocal_observer",
        "stateful_object_observer",
        "closure_list_switch_fn",
        "stateful_object_switch_fn",
        "python_block_closure",
        "python_block_object",
    ],
)
def test_hidden_controller_state_is_refused_by_name(engine, ek, case):
    b = _buck()
    kw = _hidden_state_cases(b)[case]
    ok, why = p.periodic_steady_state_supported(b, T, engine=engine, **ek, **kw)
    assert not ok and "hidden" in why
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, engine=engine, **ek, **kw)
    assert (ei.value.engine, ei.value.feature) == (
        "periodic_steady_state",
        "hidden_controller_state",
    )


# ---------------------------------------------------------------------------
# 3b. Open-loop gate state: refused when it MOVES, accepted when only read
# ---------------------------------------------------------------------------

class _ReadOnlyPwm:
    """A PWM gate object holding its duty and frequency, which it only reads."""

    def __init__(self, n, k, duty, f_sw):
        self.n, self.k, self.duty, self.f_sw = n, k, duty, f_sw

    def __call__(self, t):
        m = p.SwitchStateMask(self.n)
        ph = t * self.f_sw
        m.set(self.k, bool(ph - np.floor(ph) < self.duty))
        return m

    def next_edge_after(self, t):
        ts = 1.0 / self.f_sw
        c = np.floor(t / ts)
        for e in (c * ts + self.duty * ts, (c + 1) * ts, (c + 1) * ts + self.duty * ts):
            if e > t * (1 + 1e-15) + 1e-18:
                return float(e)


class _CountingPwm(_ReadOnlyPwm):
    def __call__(self, t):
        self.n_calls += 1
        return super().__call__(t)


def test_read_only_gate_object_is_accepted_and_its_orbit_is_the_transients():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("Q")
    gate = _ReadOnlyPwm(n, k, 0.4, 1.0 / T)
    r = p.periodic_steady_state(b, period=T, switch_fn=gate, engine="exact")
    assert r.stable
    assert (gate.duty, gate.f_sw) == (0.4, 1.0 / T)
    # truth 1: the same schedule from the C++ helper (an independent gate)
    sf = p.make_pwm_switch_fn(1.0 / T, 0.4, k, n)
    r_ref = p.periodic_steady_state(_buck(), period=T, switch_fn=sf, engine="exact")
    assert np.abs(r.x0 - r_ref.x0).max() < 1e-12
    # truth 2: a long trbdf2 transient driven by the object itself (an
    # independent engine; 2000 periods = 42 decay constants 2RC; measured
    # agreement 4e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tr = p.simulate(_buck(), 2000 * T, engine="trbdf2", rtol=1e-10,
                        atol=1e-13, switch_fn=gate)
    x_tr = {"v(out)": float(np.asarray(tr.v("out"))[-1]),
            "i(L)": float(np.asarray(tr.i("L"))[-1])}
    for lbl, v in zip(r.state_labels, r.x0):
        assert abs(v - x_tr[lbl]) < 1e-6, lbl
    # the fixed and the trbdf2 shooting take it as well
    for eng, ek in (("pwl", {"dt": T / 100}), ("trbdf2", {})):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert p.periodic_steady_state(_buck(), period=T, switch_fn=gate,
                                           engine=eng, **ek).stable


@pytest.mark.parametrize("engine,ek", [("exact", {}), ("pwl", {"dt": T / 100})])
def test_gate_object_with_a_counter_is_refused_naming_it(engine, ek):
    b = _buck()
    gate = _CountingPwm(b.graph.num_switches, b.switch_index_of("Q"), 0.4, 1.0 / T)
    gate.n_calls = 0
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=gate, engine=engine, **ek)
    assert ei.value.feature == "hidden_controller_state"
    assert "'switch_fn.n_calls' changed" in str(ei.value)


def test_closure_appending_to_a_list_is_refused_naming_it():
    b = _buck()
    sf = p.make_pwm_switch_fn(1.0 / T, 0.4, b.switch_index_of("Q"), b.graph.num_switches)
    seen = []

    def gate(t):
        seen.append(t)
        return sf(t)

    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=gate, engine="pwl", dt=T / 100)
    assert ei.value.feature == "hidden_controller_state"
    assert "'switch_fn.<closure seen>' changed" in str(ei.value)


def test_closure_reading_a_float_is_accepted():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("Q")
    duty = 0.4

    def gate(t):
        m = p.SwitchStateMask(n)
        ph = t / T
        m.set(k, bool(ph - np.floor(ph) < duty))
        return m

    r = p.periodic_steady_state(b, period=T, switch_fn=gate, engine="pwl", dt=T / 100)
    sf = p.make_pwm_switch_fn(1.0 / T, 0.4, k, n)
    r_ref = p.periodic_steady_state(_buck(), period=T, switch_fn=sf, engine="pwl", dt=T / 100)
    # truth: the C++ schedule of the same duty on the same engine
    assert r.stable and np.abs(r.x0 - r_ref.x0).max() < 1e-12


def test_gate_over_an_opaque_object_is_refused_naming_it():
    import random

    b = _buck()
    sf = p.make_pwm_switch_fn(1.0 / T, 0.4, b.switch_index_of("Q"), b.graph.num_switches)
    rng = random.Random(1)

    def gate(t):
        rng.random()
        return sf(t)

    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=gate, engine="pwl", dt=T / 100)
    assert "uncomparable state at switch_fn.<closure rng>" in str(ei.value)


# ---------------------------------------------------------------------------
# 4. Stateless schedules retain 100% parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "engine,ek",
    [("exact", {}), ("pwl", {"dt": T / 100}), ("trbdf2", {})],
)
def test_stateless_pwm_schedule_parity(engine, ek):
    b = _buck()
    sf = p.make_pwm_switch_fn(1.0 / T, 0.5, b.switch_index_of("Q"), b.graph.num_switches)
    r = p.periodic_steady_state(b, period=T, switch_fn=sf, engine=engine, **ek)
    assert r.stable
    assert r.controller_state.size == 0
    assert r.controller_labels == []
    assert r.circuit_state.shape == (len(r.state_labels),) == (2,)


def test_stateless_python_lambda_keeps_existing_behavior():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("Q")

    def stateless_gate(t):
        m = p.SwitchStateMask(n)
        ph = t / T
        m.set(k, bool(ph - np.floor(ph + 1e-9) < 0.5 - 1e-9))
        return m

    # Stateless Python gate still works on fixed engine (pwl)
    r = p.periodic_steady_state(b, period=T, switch_fn=stateless_gate, engine="pwl", dt=T / 100)
    assert r.stable and r.controller_state.size == 0

    # And on exact/trbdf2 it is refused as edgeless switch_fn, NOT hidden_controller_state
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=stateless_gate, engine="exact")
    assert ei.value.feature == "switch_fn"


# ---------------------------------------------------------------------------
# 5. Adversarial regression tests (sequence control, bypasses, frozen dc,
#    sampled_model refusal)
# ---------------------------------------------------------------------------

def test_control_sequence_with_chain_and_multiple_observers():
    # control=[ch] with a MixedDomainBlockChain using drive_source
    b = p.CircuitBuilder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add(
        "pi",
        p.PIController(Kp=0.1, Ki=500.0, output_min=-50.0, output_max=50.0),
        inputs=dict(setpoint=5.0, measured="out", dt="dt"),
        output="vin_cmd",
    )
    ch.drive_source(b, "vin_cmd", ("v", "src", "gnd"))
    b.add_resistor("R1", "src", "out", 10.0)
    b.add_capacitor("C1", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 100.0)

    r_list = p.periodic_steady_state(b, period=T, control=[ch], engine="exact", rtol=1e-11)
    assert r_list.stable
    assert abs(float(r_list.x0[r_list.state_labels.index("v(out)")]) - 5.0) < 1e-9

    # control=[obs1, obs2] with two compiled observers (including one Monitor)
    b2 = _buck()
    lp2 = _pi_loop(b2)
    mon_ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    mon_ch.add(
        "del",
        p.DelayBlock(samples=1, initial_value=0.0),
        inputs=dict(x="node:out"),
        output="v_del",
    )
    obs2 = mon_ch.make_step_observer(b2)
    r_multi = p.periodic_steady_state(
        b2,
        period=T,
        switch_fn=lp2.switch_fn,
        control=[lp2.step_observer, obs2],
        engine="exact",
    )
    assert r_multi.stable
    assert "ctrl(pi.integrator_state)" in r_multi.controller_labels
    assert "ctrl(del.buf[0])" in r_multi.controller_labels
    assert abs(float(r_multi.x0[r_multi.state_labels.index("v(out)")]) - V_REF) < 1e-9


def test_hidden_state_function_attr_and_slotless_class_bypasses_refused():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("Q")
    pwm = p.make_pwm_switch_fn(1.0 / T, 0.5, k, n)

    # Bypass A: function attribute mutation
    def attr_sw(t):
        attr_sw.cnt += 1
        return pwm(t)

    attr_sw.cnt = 0

    with pytest.raises(p.UnsupportedError) as ei_a:
        p.periodic_steady_state(b, period=T, switch_fn=attr_sw, engine="pwl", dt=T / 100)
    assert ei_a.value.feature == "hidden_controller_state"

    def attr_obs(t, x):
        attr_obs.calls += 1

    attr_obs.calls = 0

    with pytest.raises(p.UnsupportedError) as ei_a2:
        p.periodic_steady_state(
            b, period=T, switch_fn=pwm, step_observer=attr_obs, engine="pwl", dt=T / 100
        )
    assert ei_a2.value.feature == "hidden_controller_state"

    # Bypass B1: slotless callable instance with mutable class attribute
    class _SlotlessMutableClassAttr:
        __slots__ = ()
        acc = [0.0]

        def __call__(self, t, x):
            self.acc[0] += float(x[0])

    with pytest.raises(p.UnsupportedError) as ei_b1:
        p.periodic_steady_state(
            b,
            period=T,
            switch_fn=pwm,
            step_observer=_SlotlessMutableClassAttr(),
            engine="pwl",
            dt=T / 100,
        )
    assert ei_b1.value.feature == "hidden_controller_state"

    # Bypass B2: slotless callable instance mutating immutable class scalar
    class _SlotlessMutatingScalarAttr:
        __slots__ = ()
        cnt = 0

        def __call__(self, t, x):
            type(self).cnt += 1

    with pytest.raises(p.UnsupportedError) as ei_b2:
        p.periodic_steady_state(
            b,
            period=T,
            switch_fn=pwm,
            step_observer=_SlotlessMutatingScalarAttr(),
            engine="pwl",
            dt=T / 100,
        )
    assert ei_b2.value.feature == "hidden_controller_state"


@dataclass(frozen=True)
class _FrozenCfg:
    duty: float
    period: float


@dataclass(frozen=True)
class _FrozenWithMutableBox:
    duty: float
    box: list


_FROZEN_CFG = _FrozenCfg(duty=0.5, period=T)
_FROZEN_MUTABLE_CFG = _FrozenWithMutableBox(duty=0.5, box=[0.0])


def test_frozen_dataclass_global_in_stateless_lambda_allowed():
    b = _buck()
    n, k = b.graph.num_switches, b.switch_index_of("Q")

    def frozen_cfg_gate(t):
        m = p.SwitchStateMask(n)
        ph = t / _FROZEN_CFG.period
        m.set(k, bool(ph - np.floor(ph + 1e-9) < _FROZEN_CFG.duty - 1e-9))
        return m

    r = p.periodic_steady_state(b, period=T, switch_fn=frozen_cfg_gate, engine="pwl", dt=T / 100)
    assert r.stable and r.controller_state.size == 0

    # Frozen dataclass carrying a mutable field is still refused
    def dirty_frozen_gate(t):
        _FROZEN_MUTABLE_CFG.box[0] = t
        return frozen_cfg_gate(t)

    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, switch_fn=dirty_frozen_gate, engine="pwl", dt=T / 100)
    assert ei.value.feature == "hidden_controller_state"


def test_closed_loop_exact_sampled_model_is_refused_by_name():
    b = _buck()
    lp = _pi_loop(b)
    r = p.periodic_steady_state(b, period=T, closed_loops=[lp], engine="exact")
    with pytest.raises(p.UnsupportedError) as ei:
        r.sampled_model(inputs=[("source", "Vin")], outputs=["v(out)"])
    assert ei.value.feature == "sampled_model"
    assert ei.value.kind == "closed_loop"


# ---------------------------------------------------------------------------
# 6. The controller-state byte layout is checked, not assumed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("const,wrong,what", [
    ("_GROUP_REC_BYTES", 32, "byte"),            # one byte short per group
    ("_COUNTERS_BYTES", 16, "byte"),             # a counter dropped
    ("_STATE_MAGIC", 0x5053474300000002, "layout tag"),   # another version
])
def test_wrong_layout_constant_is_refused_by_name(monkeypatch, const, wrong, what):
    import pulsim._control_pss as cp

    b = _buck()
    lp = _pi_loop(b)
    monkeypatch.setattr(cp, const, wrong)
    monkeypatch.setattr(cp, "_EMPTY_CHAIN_LEN", None)
    with pytest.raises(p.UnsupportedError) as ei:
        p.periodic_steady_state(b, period=T, closed_loops=[lp])
    assert ei.value.feature == "controller_state_layout"
    assert what in str(ei.value)


def test_layout_check_survives_python_O():
    # the check is a raise, not an assert: `python -O` keeps it
    import subprocess
    import sys

    code = (
        "import pulsim._control_pss as cp, pulsim as p\n"
        "try:\n"
        "    cp._layout(False, 'x')\n"
        "except p.UnsupportedError as e:\n"
        "    print(e.feature)\n"
    )
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "controller_state_layout"
