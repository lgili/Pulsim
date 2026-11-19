"""add-kernel-simulation-object task 7.2: the fixed overhead of one
simulate() call, as Python bytecodes executed on a tiny 2-switch buck
(sweeps and optimisation loops call simulate() thousands of times).

A ratchet, not a timing test. LOCAL ONLY: the count depends on the
interpreter's bytecode (it differs between Python versions) and on numpy's
Python wrappers, so it runs on the developer Mac with CPython 3.13 and is
skipped on CI. Counted with `sys.settrace` opcode events on a warm (second)
call, in a fresh interpreter: inside the suite, state earlier tests leave
behind adds ~200. Before Stage A (214947c1): pwl 2551, trbdf2 3380; Stage A
as first landed: 3609 / 4413; now: see the ceilings. Lower them when the
count drops; never raise them to pass."""

import os
import pathlib
import subprocess
import sys

import pytest

import pulsim as p

pytestmark = pytest.mark.skipif(
    bool(os.environ.get("CI")) or sys.platform != "darwin"
    or sys.version_info[:2] != (3, 13),
    reason="local bytecode gate (CPython 3.13 on the developer Mac)")

# bytecodes per warm call; measured 2834 / 3683 (2026-11-19), ~2 % margin
_CEILING = {"pwl": 2890, "trbdf2": 3760}

_SCRIPT = r"""
import gc, sys, warnings
import pulsim as p
warnings.simplefilter("ignore")
b = p.CircuitBuilder()
b.add_voltage_source("V", "in", "gnd", 12.0)
b.add_switch("S", "in", "sw", 1e3, 1e-9)
b.add_diode("D", "gnd", "sw", 1e3, 1e-9)
b.add_inductor("L", "sw", "out", 100e-6)
b.add_capacitor("C", "out", "gnd", 10e-6)
b.add_resistor("R", "out", "gnd", 5.0)
sw = p.make_pwm_switch_fn(frequency=50e3, duty=0.4, switch_idx=0, num_switches=2)
if sys.argv[1] == "pwl":
    call = lambda: p.simulate(b, 2e-6, dt=1e-7, switch_fn=sw)
else:
    call = lambda: p.simulate(b, 4e-5, switch_fn=sw)
n = [0]
def tr(frame, event, arg):
    frame.f_trace_opcodes = True
    if event == "opcode":
        n[0] += 1
    return tr
call()          # warm: first-call imports and caches
gc.collect(); gc.disable()
sys.settrace(tr)
call()
sys.settrace(None)
print(n[0])
"""


@pytest.mark.parametrize("engine", ["pwl", "trbdf2"])
def test_simulate_fixed_overhead_in_bytecodes(engine):
    env = dict(os.environ,
               PYTHONPATH=str(pathlib.Path(p.__file__).resolve().parents[1]))
    out = subprocess.run([sys.executable, "-c", _SCRIPT, engine], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    n = int(out.stdout.split()[-1])
    assert n <= _CEILING[engine], (engine, n)
