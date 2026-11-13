"""Blocks ported to the control graph against their Python classes
(add-compiled-control-graph, task 5): 10^4 random inputs per block,
compared byte for byte; the Python evaluator and its fallback are gone."""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import pulsim as p

N = 10_000


def _rng(seed):
    return np.random.default_rng(seed)


# name -> (block factory, {kwarg: sampler(rng) -> value}, output arity)
def _u(lo, hi):
    return lambda r: float(r.uniform(lo, hi))


def _dt(r):
    return float(r.choice([0.0, 1e-6, 1e-5, 3.3e-4]))


CASES = {
    "Comparator": (lambda: p.Comparator(threshold=0.1, hysteresis=0.4,
                                        output_high=5.0, output_low=-1.0),
                   {"input_value": _u(-1, 1)}),
    "RateLimiter": (lambda: p.RateLimiter(slew_up=1e3, slew_down=3e3),
                    {"target": _u(-2, 2), "dt": _dt}),
    "RateLimiter_inf": (lambda: p.RateLimiter(),
                        {"target": _u(-2, 2), "dt": _dt}),
    "SampleHold": (lambda: p.SampleHold(sample_period=2.5e-5),
                   {"input_value": _u(-1, 1), "t": None}),
    "TransferFunction": (lambda: p.TransferFunction(
                             num=(0.2, 0.3, -0.1), den=(-0.5, 0.06)),
                         {"x": _u(-1, 1)}),
    "TransferFunction_fir": (lambda: p.TransferFunction(num=(0.5, 0.5)),
                             {"x": _u(-1, 1)}),
    "StateMachine_toggle": (lambda: p.StateMachine(mode="toggle",
                                                   high_value=2.0),
                            {"trigger": _u(0, 1)}),
    "StateMachine_level": (lambda: p.StateMachine(mode="level"),
                           {"trigger": _u(0, 1)}),
    "StateMachine_sr": (lambda: p.StateMachine(mode="sr_latch"),
                        {"set_": _u(0, 0.7), "reset_": _u(0, 0.7)}),
    "OpAmp": (lambda: p.OpAmp(gain=1e4, rail_min=-12.0, rail_max=12.0),
              {"in_pos": _u(-1e-3, 1e-3), "in_neg": _u(-1e-3, 1e-3)}),
    "LookupTable1D": (lambda: p.LookupTable1D(xs=(-1.0, 0.0, 0.3, 2.0),
                                              ys=(4.0, -1.0, 0.5, 0.25)),
                      {"x": _u(-1.5, 2.5)}),
    "MathBlock_div": (lambda: p.MathBlock(op="div"),
                      {"a": _u(-3, 3), "b": lambda r: float(
                          r.choice([0.0, r.uniform(-2, 2)]))}),
    "MathBlock_sqrt": (lambda: p.MathBlock(op="sqrt"), {"a": _u(-1, 4)}),
    "MathBlock_mul": (lambda: p.MathBlock(op="mul"),
                      {"a": _u(-3, 3), "b": _u(-3, 3)}),
    "MovingAverageFilter": (lambda: p.MovingAverageFilter(alpha=0.83),
                            {"input_value": _u(-1, 1)}),
    "Differentiator": (lambda: p.Differentiator(alpha=0.35),
                       {"x": _u(-1, 1), "dt": _dt}),
    "Integrator": (lambda: p.Integrator(gain=3.0, output_min=-0.5,
                                        output_max=0.5),
                   {"x": _u(-1, 1), "dt": _dt}),
    "FirstOrderLowPass": (lambda: p.FirstOrderLowPass(tau=1e-3),
                          {"input_value": _u(-1, 1), "dt": _dt}),
    "FirstOrderLowPass_state": (lambda: p.FirstOrderLowPass(tau=1e-3,
                                                            state=0.4),
                                {"input_value": _u(-1, 1), "dt": _dt}),
    "FirstOrderLowPass_tau0": (lambda: p.FirstOrderLowPass(tau=0.0),
                               {"input_value": _u(-1, 1), "dt": _dt}),
    # phase in seconds (the carrier starts at t = +phase); t = k*7.3 us
    # walks every carrier position
    "PwmGenerator": (lambda: p.PwmGenerator(frequency=1e5, phase=2.5e-6),
                     {"duty": _u(-0.1, 1.1), "t": None}),
    "PwmGenerator_f0": (lambda: p.PwmGenerator(frequency=0.0),
                        {"duty": _u(-0.1, 1.1), "t": None}),
    "DelayBlock": (lambda: p.DelayBlock(samples=3, initial_value=0.25),
                   {"x": _u(-1, 1)}),
    "PIController": (lambda: p.PIController(Kp=0.3, Ki=40.0,
                                            output_min=-0.2, output_max=0.7),
                     {"setpoint": _u(-1, 1), "measured": _u(-1, 1),
                      "dt": _dt}),
    "Gain": (lambda: p.Gain(k=-2.7), {"x": _u(-1, 1)}),
    "Subtract": (lambda: p.Subtract(), {"a": _u(-1, 1), "b": _u(-1, 1)}),
    "Limiter": (lambda: p.Limiter(output_min=-0.3, output_max=0.6),
                {"x": _u(-1, 1)}),
    "PIDController": (lambda: p.PIDController(
                          Kp=0.7, Ki=50.0, Kd=1e-3, derivative_alpha=0.2,
                          output_min=-0.4, output_max=0.9),
                      {"setpoint": _u(-1, 1), "measured": _u(-1, 1),
                       "dt": _dt}),
}


def _python(block, kwargs_seq, tuple_out=False):
    out = []
    for kw in kwargs_seq:
        y = block.update(**kw)
        out.append(tuple(y) if tuple_out else float(y))
    return out


def _kernel(block, names, kwargs_seq, outputs="y"):
    """Drive the block through a compiled chain: each input is a channel
    the test sets before the evaluation (t from the time input)."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ins = {n: ("time" if n == "t" else f"channel:in_{n}") for n in names}
    ch.add("blk", block, inputs=ins, output=outputs)
    obs = ch.make_step_observer(b, dt=1e-6)
    cxx = ch._cxx_chain
    x = np.zeros(1)
    res = []
    outs = outputs if isinstance(outputs, tuple) else (outputs,)
    for k, kw in enumerate(kwargs_seq):
        for n in names:
            if n != "t":
                cxx.set_channel(f"in_{n}", kw[n])
        obs(kw.get("t", k * 1e-6), x)
        vals = tuple(cxx.get_channel(o) for o in outs)
        res.append(vals if isinstance(outputs, tuple) else vals[0])
    return res


@pytest.mark.parametrize("case", list(CASES))
def test_ported_block_is_bit_identical_to_its_python_class(case):
    make, samplers = CASES[case]
    r = _rng(zlib.crc32(case.encode()))
    seq = []
    for k in range(N):
        kw = {}
        for n, f in samplers.items():
            kw[n] = k * 7.3e-6 if f is None else f(r)
        seq.append(kw)
    want = _python(make(), seq)
    got = _kernel(make(), list(samplers), seq)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


def test_sum_with_weights_and_a_tuple_of_inputs():
    r = _rng(3)
    seq = [{"a": r.uniform(-1, 1), "b": r.uniform(-1, 1),
            "c": r.uniform(-1, 1)} for _ in range(N)]
    blk = p.Sum(weights=(0.3, -1.7, 2.1))
    want = [blk.update(inputs=(kw["a"], kw["b"], kw["c"])) for kw in seq]
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("s", p.Sum(weights=(0.3, -1.7, 2.1)),
           inputs=dict(inputs=("channel:a", "channel:b", "channel:c")),
           output="y")
    obs = ch.make_step_observer(b, dt=1e-6)
    got = []
    for kw in seq:
        for n in "abc":
            ch._cxx_chain.set_channel(n, kw[n])
        obs(0.0, np.zeros(1))
        got.append(ch._cxx_chain.get_channel("y"))
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()


def test_sum_weights_must_match_its_inputs():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    # 2.0 silently ignored a third input c (w2 = 0); now a build error.
    ch.add("s", p.Sum(), inputs=dict(a=1.0, b=2.0, c=3.0), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="2 weights but 3"):
        ch.make_step_observer(b, dt=1e-6)


def test_signal_mux_and_demux():
    r = _rng(5)
    seq = [{"selector": float(r.uniform(-1.5, 4.5)),
            "v": [float(r.uniform(-1, 1)) for _ in range(3)]}
           for _ in range(N)]
    mux = p.SignalMux()
    want = [mux.update(selector=kw["selector"], inputs=tuple(kw["v"]))
            for kw in seq]
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("m", p.SignalMux(),
           inputs=dict(selector="channel:sel",
                       inputs=("channel:v0", "channel:v1", "channel:v2")),
           output="y")
    ch.add("d", p.SignalDemux(n_outputs=2), inputs=dict(x="channel:y"),
           output=("y0", "y1"))
    obs = ch.make_step_observer(b, dt=1e-6)
    cx = ch._cxx_chain
    got, fan = [], []
    for kw in seq:
        cx.set_channel("sel", kw["selector"])
        for i, v in enumerate(kw["v"]):
            cx.set_channel(f"v{i}", v)
        obs(0.0, np.zeros(1))
        got.append(cx.get_channel("y"))
        fan.append((cx.get_channel("y0"), cx.get_channel("y1")))
    assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
    assert all(a == b == g for (a, b), g in zip(fan, got))


def test_initial_block_state_reaches_the_kernel():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("pi", p.PIController(Kp=0.0, Ki=1.0, integrator_state=0.25),
           inputs=dict(setpoint=0.0, measured=0.0, dt="dt"), output="u")
    ch.add("cmp", p.Comparator(state=True),
           inputs=dict(input_value=0.0), output="c")
    obs = ch.make_step_observer(b, dt=1e-6)
    obs(0.0, np.zeros(1))
    assert ch.get("u") == 0.25 and ch.get("c") == 1.0


def test_an_unknown_block_kind_is_a_build_error_naming_python_block():
    class Doubler:
        def update(self, *, x):
            return 2 * x

    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("d", Doubler(), inputs=dict(x=1.0), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="PythonBlock"):
        ch.make_step_observer(b, dt=1e-6)


def test_python_block_runs_a_callable_in_the_graph():
    class Doubler:
        def update(self, *, x):
            return 2 * x

    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=3.0), inputs=dict(x=1.0), output="u")
    ch.add("d", p.PythonBlock(Doubler()), inputs=dict(x="channel:u"),
           output="y")
    ch.add("split", p.PythonBlock(lambda x: (x, -x)),
           inputs=dict(x="channel:y"), output=("p", "n"))
    obs = ch.make_step_observer(b, dt=1e-6)
    obs(0.0, np.zeros(1))
    assert (ch.get("y"), ch.get("p"), ch.get("n")) == (6.0, 6.0, -6.0)


def test_use_kernel_false_is_a_deprecation_error_naming_the_replacement():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=2.0), inputs=dict(x=1.0), output="y")
    with pytest.raises(DeprecationWarning, match="record_channel.*PythonBlock"):
        ch.make_step_observer(b, dt=1e-6, use_kernel=False)


def test_channels_registered_before_the_build_are_recorded():
    # 2.0 dropped a record_channel made before make_step_observer on the
    # kernel path (the converter notebooks used use_kernel=False for it).
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=2.0), inputs=dict(x=1.0), output="y")
    ch.record_channel("y")
    obs = ch.make_step_observer(b, dt=1e-6)
    for k in range(3):
        obs(k * 1e-6, np.zeros(1))
    assert list(ch.get_channel_history("y")) == [2.0, 2.0, 2.0]


def test_comparator_hysteresis_is_python_exact_at_the_band_edge():
    # threshold + half band exactly: Python keeps LOW (strict >).
    blk = p.Comparator(threshold=0.5, hysteresis=0.2)
    assert _kernel(blk, ["input_value"], [{"input_value": 0.6}]) == [0.0]


@pytest.mark.parametrize("engine,kw", [("pwl", {"dt": 1e-7}),
                                       ("trbdf2", {})])
def test_python_calls_per_run_are_the_python_blocks_evaluations(engine, kw):
    """Counter gate (task 9.1): a 10 kHz group over 2 ms ticks 20 times
    on either engine (one evaluation per tick, also at dt = 0.1 us); a
    graph with no PythonBlock makes no Python call during the run, and
    one PythonBlock adds exactly one call per tick."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from test_control_graph_characterisation import PyCallsDuringRun
    for with_py in (False, True):
        b = p.CircuitBuilder()
        b.add_voltage_source("V1", "in", "gnd", 1.0)
        b.add_resistor("R", "in", "out", 1e3)
        b.add_capacitor("C", "out", "gnd", 1e-6)
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
        ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
        ch.add("i", p.Integrator(), inputs=dict(x="channel:u", dt="dt"),
               output="y")
        calls = []
        if with_py:
            ch.add("py", p.PythonBlock(lambda y: calls.append(y) or y),
                   inputs=dict(y="channel:y"), output="z")
        # A driven source: its b_extra and next_jump_after run in C++.
        b.add_resistor("Rd", "drv", "gnd", 1e3)
        ch.drive_source(b, "y", ("v", "drv", "gnd"))
        obs = ch.make_step_observer(b)
        with PyCallsDuringRun() as c:
            p.simulate(b, 2e-3, engine=engine, step_observer=obs, **kw)
        n_blocks = 3 if with_py else 2
        assert ch.tick_counts == [20]
        assert ch._cxx_chain.n_block_evals == 20 * n_blocks
        assert len(calls) == (20 if with_py else 0)
        assert (c.n > 0) == with_py, (engine, with_py, c.n)


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


def test_a_reused_observer_restarts_its_tick_grid():
    # Before: the second simulate() saw tick instants from the first run
    # (k = 20 at t >= 2 ms) and never ticked again.
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
    obs = ch.make_step_observer(b)
    for _ in range(2):
        p.simulate(b, 2e-3, engine="pwl", dt=1e-6, step_observer=obs)
    assert ch.tick_counts == [40]


def test_a_rate_period_below_the_fixed_step_is_refused():
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(5e-8))
    ch.add("i", p.Integrator(), inputs=dict(x="out", dt="dt"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="shorter than the fixed"):
        p.simulate(b, 1e-5, engine="pwl", dt=1e-7,
                   step_observer=ch.make_step_observer(b))


def test_a_pwm_inheriting_a_controller_rate_is_refused():
    def chain(pwm_rate):
        ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
        ch.add("pi", p.PIController(Kp=0.1, Ki=10.0),
               inputs=dict(setpoint=1.0, measured="out", dt="dt"),
               output="d")
        ch.add("pwm", p.PwmGenerator(frequency=1e5),
               inputs=dict(duty="channel:d", t="time"), output="g",
               rate=pwm_rate)
        return ch
    b = _rc()
    with pytest.raises(p.ControlGraphBuildError, match="PwmGenerator 'pwm'"):
        chain(None).make_step_observer(b)
    chain(p.Rate(1e-7)).make_step_observer(b)   # its own rate: fine


def test_controller_period_with_a_rated_chain_is_refused():
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="u")
    with pytest.raises(ValueError, match="Drop controller_period"):
        p.simulate(b, 1e-3, engine="pwl", dt=1e-6, controller_period=1e-4,
                   step_observer=ch.make_step_observer(b))


@pytest.mark.parametrize("delay,on", [(0.0, True), ("one_sample", False)])
def test_a_switch_gate_honours_compute_delay(delay, on):
    # The gate is a circuit sink: a one_sample group's output reaches it
    # at the next tick; the signal itself is current.
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4, compute_delay=delay))
    ch.add("g", p.Gain(k=1.0), inputs=dict(x=1.0), output="gate")
    obs = ch.make_step_observer(b)
    sw = ch.make_pwm_switch_fn("gate", num_switches=1)
    obs(0.0, np.zeros(3))
    assert ch.get("gate") == 1.0
    assert sw(0.0).get(0) is on


def test_runtime_faults_are_value_errors():
    b = _rc()
    ch = p.MixedDomainBlockChain()
    ch.add("m", p.SignalMux(),
           inputs=dict(selector="channel:sel", inputs=(1.0, 2.0)), output="y")
    obs = ch.make_step_observer(b, dt=1e-6)
    ch._cxx_chain.set_channel("sel", float("nan"))
    with pytest.raises(ValueError, match="selector is not finite"):
        obs(0.0, np.zeros(3))   # Python's int(nan) raises too
    ch = p.MixedDomainBlockChain()
    ch.add("py", p.PythonBlock(lambda: (1.0, 2.0, 3.0)), inputs={},
           output=("a", "b"))
    obs = ch.make_step_observer(b, dt=1e-6)
    with pytest.raises(ValueError, match="3 values for 2 outputs") as e:
        obs(0.0, np.zeros(3))
    assert not isinstance(e.value, p.ControlGraphBuildError)
