"""Reviewer leftovers of add-compiled-control-graph (task 9.3 review):
the compile cache's parents, the cache check on Windows, warnings that
point at the caller, the multi-rate export's call order, and one live
block per circuit for an exported library (its state is static)."""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import pulsim as p
from pulsim.c_block import _cache_dir, _check_in_profile, _find_compiler
from pulsim.c_export import export_c

T = 1e-4
FLAGS = ["-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
         "-ffp-contract=off", "-O2", "-shared", "-fPIC"]


def _cc():
    cc = _find_compiler("c")
    if cc is None:
        if sys.platform == "win32":
            pytest.skip("no C compiler on PATH (Windows runner)")
        pytest.fail("no C compiler found (Linux and macOS runners have one)")
    return cc


def _build(tmp_path, ex, name):
    src = ex.write(tmp_path / f"{name}.c")
    so = tmp_path / f"{name}.so"
    r = subprocess.run([_cc(), *FLAGS, str(src), "-o", str(so), "-lm"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return so


def _rc():
    b = p.CircuitBuilder()
    b.add_voltage_source("V", "u", "gnd", 1.0)
    b.add_resistor("R", "u", "v", 1e3)
    b.add_capacitor("C", "v", "gnd", 1e-6)
    return b


# ---- cache -----------------------------------------------------------------

@pytest.mark.skipif(os.name != "posix", reason="POSIX owners and modes")
def test_the_cache_parent_is_checked_too(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    d = _cache_dir()
    assert d.parent.stat().st_mode & 0o777 == 0o700
    os.chmod(d.parent, 0o777)            # ours: tightened again
    _cache_dir()
    assert d.parent.stat().st_mode & 0o777 == 0o700
    uid = os.stat(d.parent).st_uid
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)   # someone else's
    with pytest.raises(RuntimeError, match=r"refusing .*pulsim"):
        _cache_dir()


def test_windows_cache_must_sit_in_the_profile(tmp_path, monkeypatch):
    home = tmp_path / "me"
    (home / "AppData").mkdir(parents=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    _check_in_profile(home / "AppData")
    with pytest.raises(RuntimeError, match="outside your profile"):
        _check_in_profile(tmp_path / "shared")


# ---- warnings point at the caller ------------------------------------------

_OURS = ("<callable>", "clamping the block", "PythonBlock adapter",
         "not a multiple of dt")


def _where(fn):
    """(category, file) of the warnings under test, others ignored."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fn()
    return [(x.category, Path(x.filename).name) for x in w
            if any(s in str(x.message) for s in _OURS)]


def test_warnings_point_at_the_callers_line():
    here = Path(__file__).name
    b = _rc()
    b.add_switch("S", "u", "v", 1.0, 1e-9)
    assert _where(lambda: p.bind_pi_to_switch(
        b, pi=p.PIController(Kp=1.0, Ki=0.0), measured=lambda x: 0.0,
        setpoint=1.0, switch="S", freq=1e4)) == [(DeprecationWarning, here)]

    def step(t, dt, i, o, s):
        o[0] = 0.0
    got = _where(lambda: p.add_c_block(
        _rc(), inputs=[("v", "v")], outputs=[("v", "x", "gnd")], dt=1e-7,
        fn=step, sim_dt=1e-6))
    assert got == [(UserWarning, here), (DeprecationWarning, here)]
    b = _rc()
    b.add_switch("S", "u", "v", 1.0, 1e-9)
    loop = p.bind_pi_to_switch(b, pi=p.PIController(Kp=1.0, Ki=0.0),
                               measured="v", setpoint=1.0, switch="S",
                               freq=3e4)        # T = 33.3 us, not k * dt
    assert _where(lambda: p.simulate(b, 1e-4, dt=1e-6, closed_loops=[loop])) \
        == [(UserWarning, here)]
    ticks = []
    assert _where(lambda: p.simulate(
        _rc(), 1e-4, dt=1e-6, controller_period=2.5e-6,
        step_observer=lambda t, x: ticks.append(t))) == [(UserWarning, here)]
    assert len(ticks) == 40


# ---- multi-rate export: call order -----------------------------------------

def _two_rates(slow_reads_fast: bool):
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    if slow_reads_fast:
        ch.add("g", p.Gain(k=2.0), inputs=dict(x="channel:u"), output="y")
        ch.add("lp", p.FirstOrderLowPass(tau=1e-3),
               inputs=dict(input_value="channel:y", dt="dt"), output="z",
               rate=p.Rate(10 * T))
    else:
        ch.add("lp", p.FirstOrderLowPass(tau=1e-3),
               inputs=dict(input_value="channel:u", dt="dt"), output="z",
               rate=p.Rate(10 * T))
        ch.add("g", p.Gain(k=2.0), inputs=dict(x="channel:z"), output="y",
               rate=p.Rate(T))
    return ch


@pytest.mark.parametrize("slow_reads_fast", [True, False])
def test_multi_rate_export_called_in_rate_order(tmp_path, slow_reads_fast):
    """Due together, step_rate0 then step_rate1 reproduce the kernel's
    single pass bit for bit when no block reads a signal a later-called
    rate writes (the slow rate reads the fast one); a fast block reading
    the slow rate's output at a shared instant sees the previous value
    in C (docs/control-graph.md)."""
    b = p.CircuitBuilder()
    b.add_resistor("R", "a", "gnd", 1.0)
    ex = export_c(_two_rates(slow_reads_fast), b, outputs=["y", "z"])
    lib = ctypes.CDLL(str(_build(tmp_path, ex, f"mr{int(slow_reads_fast)}")))
    P = ex.prefix
    getattr(lib, f"{P}_state_size").restype = ctypes.c_size_t
    get = getattr(lib, f"{P}_get")
    get.restype = ctypes.c_double
    st = ctypes.create_string_buffer(getattr(lib, f"{P}_state_size")())
    getattr(lib, f"{P}_init")(st)
    k = _two_rates(slow_reads_fast)
    # the kernel's state-time tick: runs the groups due at t
    tick = k.make_step_observer(b, dt=T)._rate_ticks[0][0]
    vin = (ctypes.c_double * len(ex.inputs))()
    same = True
    for i in range(60):
        u = float((i * 7) % 5) - 1.0
        k._cxx_chain.set_channel("u", u)
        vin[ex.inputs.index("channel:u")] = u
        tick(i * T, np.zeros(1))
        getattr(lib, f"{P}_step_rate0")(st, vin, ctypes.c_double(i * T))
        if i % 10 == 0:
            getattr(lib, f"{P}_step_rate1")(st, vin, ctypes.c_double(i * T))
        for s in ("y", "z"):
            same = same and get(st, ex.signals.index(s)) == k.get(s)
    assert same is slow_reads_fast


# ---- one block per circuit for an exported library ---------------------------------

def test_an_exported_library_runs_one_block_per_circuit(tmp_path):
    ch = p.MixedDomainBlockChain(rate=p.Rate(T))
    ch.add("i", p.Integrator(), inputs=dict(x="channel:u", dt="dt"),
           output="y")
    b = _rc()
    ex = export_c(ch, b, outputs=["y"])
    so = str(_build(tmp_path, ex, "one"))
    wire = dict(inputs=[("v", "v")], outputs=[("v", "x", "gnd")], dt=T)
    b1 = _rc()
    p.add_c_block(b1, lib=so, **wire)
    with pytest.raises(ValueError, match="one static state"):
        p.add_c_block(b1, lib=so, **wire)
    # another circuit may load it (a sweep): its init resets the state
    p.add_c_block(_rc(), lib=so, **wire)
