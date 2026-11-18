"""Group 8 of add-compiled-control-graph: pulsim.export_c renders a rated
control graph to C99 from the kernel's own law table (block_laws.h).
Compiled with -ffp-contract=off it reproduces the kernel graph bit for
bit: every block kind tick by tick through ctypes, and a PI + limiter +
PWM graph loaded back through add_c_block(lib=...) in a circuit."""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

import pulsim as p
from pulsim.c_block import _cache_dir, _check_owned, _find_compiler
from pulsim.c_export import export_c

T = 1e-5
FLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-ffp-contract=off", "-O2", "-shared", "-fPIC"]


def _cc():
    cc = _find_compiler("c")
    if cc is None:
        if sys.platform == "win32":
            pytest.skip("no C compiler on PATH (Windows runner)")
        pytest.fail("no C compiler found (Linux and macOS runners have one)")
    return cc


def _build(tmp_path, ex, name="graph"):
    src = ex.write(tmp_path / f"{name}.c")
    so = tmp_path / f"{name}.so"
    r = subprocess.run([_cc(), *FLAGS, str(src), "-o", str(so), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return so


def _all_kinds(chain):
    R = chain.rate
    add = chain.add
    add("g", p.Gain(k=1.7), inputs=dict(x="channel:u1"), output="g")
    add("sub", p.Subtract(), inputs=dict(a="a", b="channel:u2"),
        output="sub")
    add("sum", p.Sum(weights=(0.3, -1.2, 0.5)),
        inputs=dict(inputs=("channel:g", "channel:sub", 0.25)), output="sum")
    add("pi", p.PIController(Kp=0.3, Ki=40.0, output_min=-0.5,
                             output_max=0.7),
        inputs=dict(setpoint="channel:u1", measured="channel:u2", dt="dt"),
        output="pi")
    add("pid", p.PIDController(Kp=0.7, Ki=50.0, Kd=1e-3,
                               derivative_alpha=0.2, output_min=-0.4,
                               output_max=0.9),
        inputs=dict(setpoint="channel:u1", measured="channel:sum", dt="dt"),
        output="pid")
    add("lp", p.FirstOrderLowPass(tau=1e-4),
        inputs=dict(input_value="channel:u2", dt="dt"), output="lp")
    add("integ", p.Integrator(gain=3e3, output_min=-0.5, output_max=0.5),
        inputs=dict(x="channel:u1", dt="dt"), output="integ")
    add("diff", p.Differentiator(alpha=0.35),
        inputs=dict(x="channel:u2", dt="dt"), output="diff")
    add("lim", p.Limiter(output_min=-0.3, output_max=0.6),
        inputs=dict(x="channel:pid"), output="lim")
    add("ema", p.MovingAverageFilter(alpha=0.83),
        inputs=dict(input_value="channel:u1"), output="ema")
    add("cmp", p.Comparator(threshold=0.1, hysteresis=0.4, output_high=5.0,
                            output_low=-1.0),
        inputs=dict(input_value="channel:u2"), output="cmp")
    add("rl", p.RateLimiter(slew_up=1e4, slew_down=3e4),
        inputs=dict(target="channel:u1", dt="dt"), output="rl")
    add("sh", p.SampleHold(sample_period=2.5e-5),
        inputs=dict(input_value="channel:u2", t="time"), output="sh")
    add("tf", p.TransferFunction(num=(0.2, 0.3, -0.1), den=(-0.5, 0.06)),
        inputs=dict(x="channel:u1"), output="tf")
    add("sm", p.StateMachine(mode="toggle", high_value=2.0),
        inputs=dict(trigger="channel:u3"), output="sm")
    add("op", p.OpAmp(gain=30.0, rail_min=-12.0, rail_max=12.0),
        inputs=dict(in_pos="channel:u1", in_neg="channel:u2"), output="op")
    add("lut", p.LookupTable1D(xs=(-1.0, 0.0, 0.3, 2.0),
                               ys=(4.0, -1.0, 0.5, 0.25)),
        inputs=dict(x="channel:u1"), output="lut")
    add("m", p.MathBlock(op="div"),
        inputs=dict(a="channel:u1", b="channel:u2"), output="m")
    add("mux", p.SignalMux(),
        inputs=dict(selector="channel:sel",
                    inputs=("channel:g", "channel:sub", "channel:lp")),
        output="mux")
    add("dmx", p.SignalDemux(n_outputs=2), inputs=dict(x="channel:mux"),
        output=("d0", "d1"))
    add("pwm", p.PwmGenerator(frequency=3e4, phase=2.5e-6),
        inputs=dict(duty="channel:u3", t="time"), output="pwm", rate=R)
    add("svm", p.SpaceVectorModulator(v_dc=2.0),
        inputs=dict(v_alpha="channel:u1", v_beta="channel:u2"),
        output=("da", "db", "dc"))
    add("cl", p.ClarkeTransform(),
        inputs=dict(a="channel:u1", b="channel:u2", c="channel:u3"),
        output=("al", "be", "z"))
    add("icl", p.InverseClarkeTransform(),
        inputs=dict(alpha="channel:al", beta="channel:be", zero="channel:z"),
        output=("ia", "ib", "ic"))
    add("pk", p.ParkTransform(),
        inputs=dict(alpha="channel:al", beta="channel:be",
                    theta="channel:th"), output=("pd", "pq"))
    add("ipk", p.InverseParkTransform(),
        inputs=dict(d="channel:pd", q="channel:pq", theta="channel:th"),
        output=("pa", "pb"))
    add("pll", p.PLL(f_nominal=50.0, Kp=80.0, Ki=900.0),
        inputs=dict(v_alpha="channel:al", v_beta="channel:be", dt="dt"),
        output=("th", "w", "f"))
    add("dly", p.DelayBlock(samples=3, initial_value=0.25),
        inputs=dict(x="channel:u1"), output="dly")
    return chain


def _builder():
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    return b


def test_every_block_kind_is_bit_identical_in_c(tmp_path):
    b = _builder()
    ex = export_c(_all_kinds(p.MixedDomainBlockChain(rate=p.Rate(T))), b,
                  outputs=["lim"])
    assert sorted(ex.inputs) == ["a", "channel:sel", "channel:u1",
                                 "channel:u2", "channel:u3"]
    lib = ctypes.CDLL(str(_build(tmp_path, ex)))
    P = ex.prefix
    getattr(lib, f"{P}_state_size").restype = ctypes.c_size_t
    getattr(lib, f"{P}_get").restype = ctypes.c_double
    st = ctypes.create_string_buffer(getattr(lib, f"{P}_state_size")())
    getattr(lib, f"{P}_init")(st)
    step = getattr(lib, f"{P}_step_rate0")
    step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double),
                     ctypes.c_double]
    get = getattr(lib, f"{P}_get")
    get.argtypes = [ctypes.c_void_p, ctypes.c_int]

    # the kernel graph: the same blocks on every call (dt = T)
    kern = _all_kinds(p.MixedDomainBlockChain(rate=None))
    obs = kern.make_step_observer(b, dt=T)
    cx = kern._cxx_chain
    rng = np.random.default_rng(8)
    n = len(ex.signals)
    got = np.empty((10_000, n))
    want = np.empty((10_000, n))
    vin = (ctypes.c_double * len(ex.inputs))()
    for k in range(10_000):
        t = k * T
        vals = {"a": rng.uniform(-1, 1), "channel:u1": rng.uniform(-1, 1),
                "channel:u2": rng.choice([0.0, rng.uniform(-1, 1)]),
                "channel:u3": rng.uniform(0, 1),
                "channel:sel": rng.uniform(-1.5, 3.5)}
        for j, name in enumerate(ex.inputs):
            vin[j] = vals[name]
            if name.startswith("channel:"):
                cx.set_channel(name[8:], vals[name])
        obs(t, np.array([vals["a"]]))
        step(st, vin, t)
        want[k] = [kern.get(s) for s in ex.signals]
        got[k] = [get(st, j) for j in range(n)]
    assert got.tobytes() == want.tobytes()
    # every kind did something
    assert all(np.unique(want[:, j]).size > 1 for j in range(n))


def _pi_pwm(rate):
    ch = p.MixedDomainBlockChain(rate=rate, label_time=True)
    ch.add("pi", p.PIController(Kp=0.05, Ki=400.0, output_min=0.05,
                                output_max=0.95),
           inputs=dict(setpoint=4.0, measured="out", dt="dt"), output="u")
    ch.add("lim", p.Limiter(output_min=0.1, output_max=0.9),
           inputs=dict(x="channel:u"), output="d")
    ch.add("pwm", p.PwmGenerator(frequency=2e4),
           inputs=dict(duty="channel:d", t="time"), output="g", rate=rate)
    ch.add("k", p.Gain(k=10.0), inputs=dict(x="channel:g"), output="y")
    return ch


def _rc():
    b = p.CircuitBuilder()
    b.add_resistor("R", "sw", "out", 10.0)
    b.add_capacitor("C", "out", "gnd", 10e-6)
    b.add_resistor("RL", "out", "gnd", 100.0)
    return b


def test_pi_limiter_pwm_export_runs_as_a_c_block_bit_for_bit(tmp_path):
    dt = 1e-6
    rate = p.Rate(dt)
    b0 = _rc()
    ex = export_c(_pi_pwm(rate), b0, outputs=["y"])
    so = _build(tmp_path, ex, "pipwm")

    b1 = _rc()
    h = p.add_c_block(b1, inputs=[("v", "out")], outputs=[("v", "sw", "gnd")],
                      dt=dt, lib=str(so))
    r1 = p.simulate(b1, 1e4 * dt, dt=dt)

    b2 = _rc()
    ch = _pi_pwm(rate)
    ch.drive_source(b2, "y", ("v", "sw", "gnd"))
    r2 = p.simulate(b2, 1e4 * dt, dt=dt,
                    step_observer=ch.make_step_observer(b2))
    assert h.n_fires == 10_000
    x1, x2 = np.asarray(r1.states), np.asarray(r2.states)
    assert x1.shape == x2.shape
    assert x1.tobytes() == x2.tobytes()
    assert 3.0 < x1[-1, b1.node_id_of("out")] < 5.0      # the loop regulates


def test_tick_counters_are_64_bit(tmp_path):
    b = _builder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(T, offset=2e-6))
    ch.add("i", p.Integrator(), inputs=dict(x=1.0, dt="dt"), output="y")
    ex = export_c(ch, b, outputs=["y"])
    lib = ctypes.CDLL(str(_build(tmp_path, ex, "ticks")))
    P = ex.prefix
    getattr(lib, f"{P}_state_size").restype = ctypes.c_size_t
    st = ctypes.create_string_buffer(getattr(lib, f"{P}_state_size")())
    getattr(lib, f"{P}_init")(st)
    ticks = ctypes.c_uint64.from_buffer(st, 8 * len(ex.signals))
    ticks.value = 10**9                         # a simulated 10^9 ticks
    getattr(lib, f"{P}_step_rate0")(st, None, ctypes.c_double(0.0))
    tk = getattr(lib, f"{P}_ticks")
    tk.restype = ctypes.c_uint64
    nxt = getattr(lib, f"{P}_next_tick")
    nxt.restype = ctypes.c_double
    assert tk(st, 0) == 10**9 + 1
    assert nxt(st, 0) == 2e-6 + float(10**9 + 1) * T
    # a 32-bit counter would have wrapped past 2^32
    ticks.value = 2**32 + 5
    assert tk(st, 0) == 2**32 + 5


def test_export_refuses_by_name():
    b = _builder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("py", p.PythonBlock(lambda x: x), inputs=dict(x="a"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="'py' .PythonBlock"):
        export_c(ch, b, outputs=["y"])
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        h = p.add_c_block(b, inputs=[("v", "a")], outputs=[("v", "a", "gnd")],
                          dt=T, fn=lambda t, dt, i, o, s: None)
    ch.add("cb", h.block, inputs=dict(x="a"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="'cb' is a C block"):
        export_c(ch, b, outputs=["y"])
    ch = p.MixedDomainBlockChain()
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="no Rate"):
        export_c(ch, _builder(), outputs=["y"])
    ch = p.MixedDomainBlockChain(rate=p.Rate(T, compute_delay="one_sample"))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="compute_delay"):
        export_c(ch, _builder(), outputs=["y"])
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    with pytest.raises(p.ControlGraphBuildError, match="'nope'"):
        export_c(ch, _builder(), outputs=["nope"])


def test_multi_rate_export_has_a_rate_table_and_no_shim():
    ch = p.MixedDomainBlockChain(rate=p.Rate(1e-4))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    ch.add("lp", p.FirstOrderLowPass(tau=1e-3),
           inputs=dict(input_value="channel:y", dt="dt"), output="z",
           rate=p.Rate(1e-3))
    ex = export_c(ch, _builder(), outputs=["z"], prefix="ctl")
    assert ex.periods == [1e-4, 1e-3]
    assert "ctl_step_rate0" in ex.source and "ctl_step_rate1" in ex.source
    assert "pulsim_cblock_step" not in ex.source
    assert "malloc" not in ex.source


@pytest.mark.skipif(os.name != "posix", reason="POSIX owners and modes")
def test_cblock_cache_is_per_user_and_private(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    d = _cache_dir()
    base = ("Library/Caches" if sys.platform == "darwin" else ".cache")
    assert d == tmp_path / base / "pulsim" / "cblock"
    assert d.stat().st_mode & 0o777 == 0o700
    f = d / "cblk_planted.so"
    f.write_bytes(b"")
    os.chmod(f, 0o666)                       # writable by others
    with pytest.raises(RuntimeError, match="refusing"):
        _check_owned(f)
    os.chmod(f, 0o600)
    _check_owned(f)
    monkeypatch.setattr(os, "getuid", lambda: os.stat(f).st_uid + 1)
    with pytest.raises(RuntimeError, match="owned by uid"):
        _check_owned(f)


def test_a_delay_that_breaks_a_loop_pushes_after_the_pass(tmp_path):
    # y = u - 0.5 * delay(y): the delay's producer runs after it; the
    # kernel pushes x after the whole pass (two-phase), so must the C.
    def mk(rate):
        ch = p.MixedDomainBlockChain(rate=rate)
        ch.add("d", p.DelayBlock(samples=2, initial_value=0.25),
               inputs=dict(x="channel:y"), output="z")
        ch.add("g", p.Gain(k=0.5), inputs=dict(x="channel:z"), output="w")
        ch.add("s", p.Subtract(), inputs=dict(a="channel:u", b="channel:w"),
               output="y")
        return ch
    b = _builder()
    ex = export_c(mk(p.Rate(T)), b, outputs=["y"])
    lib = ctypes.CDLL(str(_build(tmp_path, ex, "dly")))
    P = ex.prefix
    getattr(lib, f"{P}_state_size").restype = ctypes.c_size_t
    get = getattr(lib, f"{P}_get")
    get.restype = ctypes.c_double
    st = ctypes.create_string_buffer(getattr(lib, f"{P}_state_size")())
    getattr(lib, f"{P}_init")(st)
    k = mk(None)
    obs = k.make_step_observer(b, dt=T)
    vin = (ctypes.c_double * 1)()
    for i in range(50):
        k._cxx_chain.set_channel("u", float(i % 7))
        vin[0] = float(i % 7)
        obs(i * T, np.zeros(1))
        getattr(lib, f"{P}_step_rate0")(st, vin, ctypes.c_double(i * T))
        assert get(st, ex.signals.index("y")) == k.get("y")


def test_export_leaves_the_chains_observer_alone():
    b = _builder()
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    obs = ch.make_step_observer(b)
    cxx = ch._cxx_chain
    export_c(ch, b, outputs=["y"])
    assert ch._cxx_chain is cxx and obs._cxx_chain is cxx


def test_the_shim_writes_nan_when_wired_or_clocked_unlike_the_export(tmp_path):
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("g", p.Gain(k=2.0), inputs=dict(x="a"), output="y")
    lib = ctypes.CDLL(str(_build(tmp_path, export_c(ch, _builder(),
                                                    outputs=["y"]), "nan")))
    step = lib.pulsim_cblock_step
    step.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                     ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                     ctypes.c_double, ctypes.c_double,
                     ctypes.POINTER(ctypes.c_void_p)]
    inp = (ctypes.c_double * 1)(1.5)
    out = (ctypes.c_double * 1)(0.0)
    state = ctypes.c_void_p(None)
    step(inp, 1, out, 1, 0.0, T, ctypes.byref(state))
    assert out[0] == 3.0
    step(inp, 1, out, 1, 0.0, 2 * T, ctypes.byref(state))
    assert math.isnan(out[0])
