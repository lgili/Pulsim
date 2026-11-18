"""Fixes from the fresh review of the compiled control graph (integration
batch 8): Python-order clamps, the PythonBlock arity fault reaching the
caller as ValueError, the C++ b_extra path of a driven source, one
closed-form pin per law the port changed, the kernel PWM phase unit, and
PythonBlock.reset()."""
from __future__ import annotations

import pathlib
import struct
import sys

import numpy as np
import pytest

import pulsim as p

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_control_graph_blocks import _kernel, _python  # noqa: E402


def _bits(seq):
    return [struct.pack("<d", float(v)) for v in seq]


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V1", "in", "gnd", 1.0)
    b.add_resistor("R", "in", "out", 1e3)
    b.add_capacitor("C", "out", "gnd", 1e-6)
    return b


# ---------------------------------------------------------------- (a) clamps
NAN = float("nan")


@pytest.mark.parametrize("make,names,seq", [
    # max(0, min(1, -0.0)) is +0.0 in Python; std::clamp returned -0.0.
    # NaN goes to the upper bound in Python; std::clamp passed it through.
    (lambda: p.Limiter(output_min=0.0, output_max=1.0), ("x",),
     [dict(x=v) for v in (-0.0, NAN, 0.0, 2.0, -3.0)]),
    (lambda: p.PIController(Kp=1.0, Ki=0.0, output_min=0.0,
                            output_max=1.0), ("setpoint", "measured", "dt"),
     [dict(setpoint=0.0, measured=0.0, dt=1e-6),     # error +0.0
      dict(setpoint=-0.0, measured=0.0, dt=1e-6),    # error -0.0
      dict(setpoint=NAN, measured=0.0, dt=1e-6)]),
])
def test_kernel_clamp_is_pythons_for_nan_and_signed_zero(make, names, seq):
    py = _python(make(), seq)
    kr = _kernel(make(), names, seq)
    assert _bits(kr) == _bits(py), (kr, py)


@pytest.mark.parametrize("block", [
    p.Limiter(output_min=1.0, output_max=0.0),
    p.PIController(Kp=1.0, output_min=1.0, output_max=0.0),
    p.PIDController(Kp=1.0, output_min=1.0, output_max=0.0),
    p.Integrator(output_min=1.0, output_max=0.0),
    p.OpAmp(rail_min=1.0, rail_max=0.0),
])
def test_swapped_output_bounds_are_refused_by_name(block):
    ch = p.MixedDomainBlockChain()
    names = {"Limiter": ("x",), "Integrator": ("x", "dt"),
             "OpAmp": ("in_pos", "in_neg")}.get(
        type(block).__name__, ("setpoint", "measured", "dt"))
    ch.add("blk", block, inputs={n: 0.0 for n in names}, output="y")
    with pytest.raises(p.ControlGraphBuildError,
                       match=rf"block 'blk': {type(block).__name__} "
                             r"\w+_min=1\.0 > \w+_max=0\.0"):
        ch.make_step_observer(_rc(), dt=1e-6)


# --------------------------------------------- (b) PythonBlock arity fault
@pytest.mark.parametrize("engine,kw", [("pwl", {"dt": 1e-6}), ("trbdf2", {})])
def test_python_block_arity_fault_reaches_simulate_as_value_error(engine, kw):
    b = _rc()
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("py", p.PythonBlock(lambda: (1.0, 2.0, 3.0)), inputs={},
           output=("a", "b"))
    with pytest.raises(ValueError, match="3 values for 2 outputs") as e:
        p.simulate(b, 1e-3, engine=engine,
                   step_observer=ch.make_step_observer(b), **kw)
    # It was retried as a step failure and came back SimulationAborted.
    assert type(e.value) is ValueError, type(e.value)


# ------------------------------------- (c) the driven source's C++ b_extra
@pytest.mark.parametrize("engine,kw", [("pwl", {"dt": 1e-6}), ("trbdf2", {})])
def test_driven_source_b_extra_makes_no_python_call(engine, kw, monkeypatch):
    K = p._pulsim.CxxChainBExtra
    calls = {"call": 0, "jump": 0}
    orig_call, orig_jump = K.__call__, K.next_jump_after

    def counting_call(self, t):
        calls["call"] += 1
        return orig_call(self, t)

    def counting_jump(self, t):
        calls["jump"] += 1
        return orig_jump(self, t)

    monkeypatch.setattr(K, "__call__", counting_call)
    monkeypatch.setattr(K, "next_jump_after", counting_jump)

    b = _rc()
    b.add_resistor("Rd", "drv", "gnd", 1e3)
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="out"), output="y")
    ch.drive_source(b, "y", ("v", "drv", "gnd"))
    obs = ch.make_step_observer(b)
    r = p.simulate(b, 2e-3, engine=engine, step_observer=obs, **kw)
    # The source is driven (2 * v(out), held between ticks) ...
    vd, vo = np.asarray(r.v("drv")), np.asarray(r.v("out"))
    assert vd[-1] == pytest.approx(2.0 * vo[-1], rel=0.1)
    assert vd[-1] > 1.0
    # ... and the engine called it in C++: the Python methods never ran.
    assert calls == {"call": 0, "jump": 0}
    # Non-vacuous: the engine got this object, and a Python call to it
    # reaches the counting method.
    assert isinstance(obs._b_extra_fn, K)
    obs._b_extra_fn(0.0)
    assert calls["call"] == 1


# ------------------------------- (d) the five laws the port changed, pinned
def test_pid_derivative_acts_on_the_error():
    # Kp = Ki = 0, Kd = 1, alpha = 1: u = (e_k - e_{k-1}) / dt, e_0 = 0.
    # A setpoint step moves u (on the measurement it would not).
    seq = [dict(setpoint=1.0, measured=0.0, dt=0.5),    # (1 - 0) / .5 = 2
           dict(setpoint=1.0, measured=0.5, dt=0.5),    # (.5 - 1) / .5 = -1
           dict(setpoint=3.0, measured=0.5, dt=0.5),    # (2.5 - .5) / .5 = 4
           dict(setpoint=3.0, measured=0.5, dt=0.0)]    # dt = 0: D = 0
    blk = p.PIDController(Kp=0.0, Ki=0.0, Kd=1.0, derivative_alpha=1.0,
                          output_min=-10.0, output_max=10.0)
    assert _kernel(blk, ("setpoint", "measured", "dt"), seq) == \
        [2.0, -1.0, 4.0, 0.0]


def test_integrator_holds_when_pushed_past_its_limit():
    # gain 1, dt 1, limits +-1: 0.75; 0.75 + 0.5 = 1.25 > 1 with x > 0
    # holds 0.75 (a clamp would give 1.0); 0.75 - 0.25 = 0.5.
    seq = [dict(x=0.75, dt=1.0), dict(x=0.5, dt=1.0), dict(x=-0.25, dt=1.0)]
    blk = p.Integrator(gain=1.0, output_min=-1.0, output_max=1.0)
    assert _kernel(blk, ("x", "dt"), seq) == [0.75, 0.75, 0.5]


def test_sum_applies_its_weights():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ch = p.MixedDomainBlockChain()
    ch.add("s", p.Sum(weights=(2.0, -0.5, 0.25)),
           inputs=dict(inputs=("channel:x0", "channel:x1", "channel:x2")),
           output="y")
    obs = ch.make_step_observer(b, dt=1e-6)
    out = []
    for k, xs in enumerate([(1.0, 4.0, 8.0), (3.0, 2.0, -4.0)]):
        for i, v in enumerate(xs):
            ch._cxx_chain.set_channel(f"x{i}", v)
        obs(k * 1e-6, np.zeros(1))
        out.append(ch.get("y"))
    assert out == [2.0 - 2.0 + 2.0, 6.0 - 1.0 - 1.0]


def test_low_pass_holds_at_dt_zero_and_starts_from_its_state():
    # tau 1, state 0.5: dt 0 -> alpha 0, holds 0.5 (it passed 3.0 through);
    # dt 1 -> alpha 1/2: 0.5 + (2.5 - 0.5)/2 = 1.5; dt 3 -> alpha 3/4 on
    # a zero error: 1.5.
    seq = [dict(input_value=3.0, dt=0.0), dict(input_value=2.5, dt=1.0),
           dict(input_value=1.5, dt=3.0)]
    blk = p.FirstOrderLowPass(tau=1.0, state=0.5)
    assert _kernel(blk, ("input_value", "dt"), seq) == [0.5, 1.5, 1.5]


def test_pwm_phase_is_a_delay_in_seconds():
    # 100 kHz, phase 2.5 us, duty 0.5: a period starts at t = +2.5 us, so
    # (t - 2.5e-6) / 1e-5 mod 1 = 0.75, 0.05, 0.45, 0.55 -> 0, 1, 1, 0.
    ts = (0.0, 3e-6, 7e-6, 8e-6)
    seq = [dict(duty=0.5, t=t) for t in ts]
    blk = p.PwmGenerator(frequency=1e5, phase=2.5e-6)
    assert _kernel(blk, ("duty", "t"), seq) == [0.0, 1.0, 1.0, 0.0]


# ------------------------------------------ (e) the kernel PWM phase unit
def test_kernel_pwm_generator_reads_phase_in_seconds_like_the_class():
    g = p._pulsim.CxxPwmGenerator()
    g.frequency, g.phase = 1e5, 2.5e-6
    ts = (0.0, 3e-6, 7e-6, 8e-6)
    assert [g.update(duty=0.5, t=t) for t in ts] == [0.0, 1.0, 1.0, 0.0]
    cls = p.PwmGenerator(frequency=1e5, phase=2.5e-6)
    assert [g.update(duty=0.5, t=t) for t in ts] == \
        [cls.update(duty=0.5, t=t) for t in ts]


@pytest.mark.parametrize("kw,expect", [
    (dict(phase=2.5e-6), [0.0, 1.0, 1.0, 0.0]),
    # the old reading, spelled out: an advance of a quarter period
    (dict(phase_fraction=0.25), [1.0, 0.0, 0.0, 1.0]),
])
def test_chain_add_pwm_generator_phase_units(kw, expect):
    k = p._pulsim
    ch = k.CxxBlockChain()
    ch.add_pwm_generator(1e5, duty=k.CxxInputRef.const_value(0.5),
                         t=k.CxxInputRef.time(), output="y", **kw)
    obs = ch.make_step_observer(1e-6)
    out = []
    for t in (0.0, 3e-6, 7e-6, 8e-6):
        obs(t, np.zeros(1))
        out.append(ch.get_channel("y"))
    assert out == expect


def test_chain_add_pwm_generator_refuses_both_phase_units():
    k = p._pulsim
    with pytest.raises(ValueError, match="not both"):
        k.CxxBlockChain().add_pwm_generator(
            1e5, 1e-6, phase_fraction=0.1,
            duty=k.CxxInputRef.const_value(0.5), t=k.CxxInputRef.time(),
            output="y")


# ------------------------------------------------- (f) PythonBlock.reset()
def test_python_block_reset_reaches_the_wrapped_object():
    class Acc:
        def __init__(self):
            self.n = 0

        def update(self, *, x):
            self.n += 1
            return float(self.n)

        def reset(self):
            self.n = 0

    acc = Acc()
    b = _rc()
    ch = p.MixedDomainBlockChain()
    ch.add("acc", p.PythonBlock(acc), inputs=dict(x=0.0), output="y")
    obs = ch.make_step_observer(b, dt=1e-6)
    for k in range(3):
        obs(k * 1e-6, np.zeros(3))
    assert acc.n == 3
    ch.reset()
    assert acc.n == 0
    p.PythonBlock(lambda x: x).reset()   # a function: nothing to forward
